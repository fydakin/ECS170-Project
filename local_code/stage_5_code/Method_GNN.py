import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from local_code.base_class.method import method


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x, edge_index, edge_weight):
        support = self.linear(x)
        output = torch.zeros_like(support)

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        messages = support[source_nodes] * edge_weight.unsqueeze(1)
        output.index_add_(0, target_nodes, messages)

        return output


class Method_GNN(method, nn.Module):
    data = None
    max_epoch = 200
    learning_rate = 1e-2
    weight_decay = 5e-4

    def __init__(
            self,
            mName=None,
            mDescription=None,
            input_size=None,
            hidden_size=64,
            output_size=None,
            dropout=0.5,
            normalize_edges=True):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout
        self.normalize_edges = normalize_edges

        self.gcn1 = None
        self.gcn2 = None

        self.epoch_numbers = []
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.y_probs = None

    def _configure_from_data(self):
        if self.data is None:
            raise ValueError('Method_GNN.data must be set before run().')

        if self.input_size is None:
            self.input_size = int(self.data['X'].shape[1])

        if self.output_size is None:
            if 'num_classes' in self.data:
                self.output_size = int(self.data['num_classes'])
            else:
                self.output_size = int(np.max(self.data['y'])) + 1

        if self.gcn1 is not None:
            return

        self.gcn1 = GraphConvolution(self.input_size, self.hidden_size)
        self.gcn2 = GraphConvolution(self.hidden_size, self.output_size)

    def forward(self, x, edge_index, edge_weight):
        x = self.gcn1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gcn2(x, edge_index, edge_weight)
        return x

    def _prepare_tensors(self):
        x = torch.FloatTensor(np.asarray(self.data['X'], dtype=np.float32))
        y = torch.LongTensor(np.asarray(self.data['y'], dtype=np.int64))

        edge_index = torch.LongTensor(np.asarray(self.data['edge_index']))
        if edge_index.ndim != 2:
            raise ValueError('edge_index must be a 2D array.')
        if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.T
        if edge_index.shape[0] != 2:
            raise ValueError('edge_index must have shape (2, num_edges).')
        edge_index = self._ensure_self_loops(edge_index, x.shape[0])

        edge_weight = self._build_edge_weight(edge_index, x.shape[0])

        train_idx = torch.LongTensor(np.asarray(self.data['train']['idx']))
        test_idx = torch.LongTensor(np.asarray(self.data['test']['idx']))

        return x, y, edge_index, edge_weight, train_idx, test_idx

    def _ensure_self_loops(self, edge_index, num_nodes):
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        has_self_loop = torch.zeros(num_nodes, dtype=torch.bool)
        self_loop_mask = source_nodes == target_nodes
        has_self_loop[source_nodes[self_loop_mask]] = True

        missing_nodes = torch.arange(num_nodes)[~has_self_loop]
        if missing_nodes.numel() == 0:
            return edge_index

        self_loops = torch.stack((missing_nodes, missing_nodes), dim=0)
        return torch.cat((edge_index, self_loops), dim=1)

    def _build_edge_weight(self, edge_index, num_nodes):
        if not self.normalize_edges:
            return torch.ones(edge_index.shape[1], dtype=torch.float32)

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        degrees = torch.bincount(
            target_nodes,
            minlength=num_nodes
        ).float()
        degrees[degrees == 0] = 1.0

        return 1.0 / degrees[target_nodes]

    def train_model(self, x, y, edge_index, edge_weight, train_idx, test_idx):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        loss_function = nn.CrossEntropyLoss()

        epoch_numbers = []
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(self.max_epoch):
            self.train()
            logits = self.forward(x, edge_index, edge_weight)
            loss = loss_function(logits[train_idx], y[train_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc = self._accuracy(logits[train_idx], y[train_idx])

            self.eval()
            with torch.no_grad():
                eval_logits = self.forward(x, edge_index, edge_weight)
                test_loss = loss_function(
                    eval_logits[test_idx],
                    y[test_idx]
                ).item()
                test_acc = self._accuracy(eval_logits[test_idx], y[test_idx])

            epoch_number = epoch + 1
            print(
                f'Epoch: {epoch_number}  '
                f'Train Accuracy: {train_acc:.4f}  '
                f'Test Accuracy: {test_acc:.4f}  '
                f'Train Loss: {loss.item():.4f}  '
                f'Test Loss: {test_loss:.4f}'
            )

            epoch_numbers.append(epoch_number)
            train_losses.append(loss.item())
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

        return (
            epoch_numbers,
            train_losses,
            test_losses,
            train_accuracies,
            test_accuracies
        )

    def test(self, x, edge_index, edge_weight, test_idx):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index, edge_weight)
            probs = torch.softmax(logits[test_idx], dim=1)
            pred_y = torch.argmax(probs, dim=1)

        return pred_y, probs

    def _accuracy(self, logits, true_y):
        pred_y = torch.argmax(logits, dim=1)
        return (pred_y == true_y).float().mean().item()

    def plot_metrics(
            self,
            epoch_numbers=None,
            train_losses=None,
            train_accuracies=None,
            test_losses=None,
            test_accuracies=None):
        import matplotlib.pyplot as plt

        epoch_numbers = epoch_numbers if epoch_numbers is not None else self.epoch_numbers
        train_losses = train_losses if train_losses is not None else self.train_losses
        train_accuracies = (
            train_accuracies if train_accuracies is not None else self.train_accuracies
        )
        test_losses = test_losses if test_losses is not None else self.test_losses
        test_accuracies = (
            test_accuracies if test_accuracies is not None else self.test_accuracies
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(epoch_numbers, train_losses, label='Train', color='tab:red')
        axes[0].plot(epoch_numbers, test_losses, label='Test', color='tab:orange')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].plot(epoch_numbers, train_accuracies, label='Train', color='tab:blue')
        axes[1].plot(epoch_numbers, test_accuracies, label='Test', color='tab:green')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, 1)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def run(self):
        print('method running...')
        self._configure_from_data()
        x, y, edge_index, edge_weight, train_idx, test_idx = self._prepare_tensors()

        print('--start training...')
        (
            self.epoch_numbers,
            self.train_losses,
            self.test_losses,
            self.train_accuracies,
            self.test_accuracies
        ) = self.train_model(x, y, edge_index, edge_weight, train_idx, test_idx)

        print('--start testing...')
        pred_y, y_probs = self.test(x, edge_index, edge_weight, test_idx)
        self.y_probs = y_probs.numpy()

        return {
            'pred_y': pred_y.numpy(),
            'true_y': y[test_idx].numpy(),
            'y_score': self.y_probs,
            'epoch_numbers': self.epoch_numbers,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies
        }
