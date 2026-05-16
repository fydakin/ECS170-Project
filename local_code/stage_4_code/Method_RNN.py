import html
import re

import numpy as np
import torch
from torch import nn

from local_code.base_class.method import method
from local_code.stage_4_code.Evaluate_Accuracy_Classification import Evaluate_Accuracy


class Method_RNN(method, nn.Module):
    #DO NOT EDIT THESE VALUES. Edit the ones in script_RNN.py
    data = None
    max_epoch = 10 
    learning_rate = 1e-3

    def __init__(
            self,
            mName=None,
            mDescription=None,
            vocab_size=None,
            output_size=None,
            embedding_dim=128,
            hidden_size=128,
            num_layers=1,
            dropout=0.3,
            batch_size=64,
            validation_split=0.2,
            task=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.task = task

        self.embedding = None
        self.rnn = None
        self.dropout = None
        self.fc = None

        self.epoch_numbers = []
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.y_probs = None

    def _configure_from_data(self):
        if self.data is None:
            raise ValueError('Method_RNN.data must be set before run().')

        if self.task is None:
            self.task = 'classification' if 'test' in self.data else 'generation'

        if self.vocab_size is None:
            self.vocab_size = len(self.data['vocab'])

        if self.output_size is None:
            if self.task == 'classification':
                self.output_size = len(self.data.get('label_map', {'neg': 0, 'pos': 1}))
            else:
                self.output_size = self.vocab_size

        if self.embedding is not None:
            return

        pad_index = self.data.get('pad_index', 0)
        rnn_dropout = self.dropout_rate if self.num_layers > 1 else 0

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=pad_index
        )
        self.rnn = nn.RNN( #change to nn.LSTM(...) or nn.GRU(...) later
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = x.long()
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)

        if self.task == 'classification':
            features = self._last_non_padding_output(output, x)
        else:
            features = hidden[-1]

        features = self.dropout(features)
        return self.fc(features)

    def _last_non_padding_output(self, output, x):
        pad_index = self.data.get('pad_index', 0)
        lengths = (x != pad_index).sum(dim=1)
        lengths = torch.clamp(lengths, min=1)
        batch_indices = torch.arange(output.size(0), device=output.device)
        return output[batch_indices, lengths - 1]

    def train_model(self, X, y, X_test=None, y_test=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = torch.LongTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))
        X_test_tensor = None
        y_test_tensor = None
        if X_test is not None and y_test is not None:
            X_test_tensor = torch.LongTensor(np.array(X_test))
            y_test_tensor = torch.LongTensor(np.array(y_test))

        if (
                self.task == 'generation'
                and X_test_tensor is None
                and y_test_tensor is None
                and 0 < self.validation_split < 1
                and len(X_tensor) > 1):
            validation_size = max(1, int(len(X_tensor) * self.validation_split))
            if validation_size < len(X_tensor):
                shuffled_indices = torch.randperm(len(X_tensor))
                validation_indices = shuffled_indices[:validation_size]
                train_indices = shuffled_indices[validation_size:]

                X_test_tensor = X_tensor[validation_indices]
                y_test_tensor = y_tensor[validation_indices]
                X_tensor = X_tensor[train_indices]
                y_tensor = y_tensor[train_indices]

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        epoch_numbers = []
        train_losses = []
        test_losses = []
        train_accuracies = []

        for epoch in range(self.max_epoch):
            self.train()
            epoch_loss = 0

            for X_batch, y_batch in dataloader:
                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            test_loss = None
            if X_test_tensor is not None and y_test_tensor is not None:
                test_loss = self._calculate_loss(X_test_tensor, y_test_tensor, loss_function)
                test_losses.append(test_loss)

            pred_y, _ = self.test(X_tensor, return_probs=False)
            if self.task == 'classification':
                accuracy_evaluator.data = {
                    'true_y': y_tensor,
                    'pred_y': pred_y
                }
                acc = accuracy_evaluator.evaluate()['acc']
            else:
                acc = (pred_y == y_tensor).float().mean().item()

            if test_loss is None:
                print(f'Epoch: {epoch + 1}  Accuracy: {acc:.4f}  Loss: {avg_loss:.4f}')
            else:
                loss_label = 'Test' if self.task == 'classification' else 'Validation'
                print(
                    f'Epoch: {epoch + 1}  Accuracy: {acc:.4f}  '
                    f'Train Loss: {avg_loss:.4f}  {loss_label} Loss: {test_loss:.4f}'
                )
            epoch_numbers.append(epoch + 1)
            train_losses.append(avg_loss)
            train_accuracies.append(acc)

        return epoch_numbers, train_losses, test_losses, train_accuracies

    def _calculate_loss(self, X_tensor, y_tensor, loss_function):
        self.eval()
        total_loss = 0
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def test(self, X, return_probs=True):
        self.eval()

        if isinstance(X, torch.Tensor):
            X_tensor = X.long()
        else:
            X_tensor = torch.LongTensor(np.array(X))

        all_pred_y = []
        all_y_probs = []
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        with torch.no_grad():
            for (X_batch,) in dataloader:
                y_pred = self.forward(X_batch)
                all_pred_y.append(y_pred.max(1)[1])
                if return_probs:
                    all_y_probs.append(torch.softmax(y_pred, dim=1))

        y_probs = torch.cat(all_y_probs) if return_probs else None
        return torch.cat(all_pred_y), y_probs

    def generate_text(self, start_words, max_new_words=30):
        if self.task != 'generation':
            raise ValueError('generate_text() is only available for text generation.')

        self.eval()
        vocab = self.data['vocab']
        idx_to_word = self.data['idx_to_word']
        unk_index = self.data.get('unk_index', 1)
        context_size = self.data.get('context_size', 3)

        tokens = self._clean_and_tokenize(start_words)
        if len(tokens) < context_size:
            raise ValueError(f'Please provide at least {context_size} starting words.')

        generated = list(tokens)
        with torch.no_grad():
            for _ in range(max_new_words):
                context = generated[-context_size:]
                context_ids = [vocab.get(token, unk_index) for token in context]
                X = torch.LongTensor([context_ids])
                y_pred = self.forward(X)
                next_idx = int(torch.argmax(y_pred, dim=1).item())
                next_word = idx_to_word.get(next_idx, '<UNK>')
                generated.append(next_word)

        return ' '.join(generated)

    def generate_joke(self, start_words, max_new_words=30):
        return self.generate_text(start_words, max_new_words)

    def _clean_and_tokenize(self, text):
        text = html.unescape(text)
        text = text.replace('<br />', ' ')
        text = text.lower()
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)

    def plot_metrics(self, epoch_numbers=None, train_losses=None, train_accuracies=None, *args, **kwargs):
        import matplotlib.pyplot as plt

        epoch_numbers = epoch_numbers if epoch_numbers is not None else self.epoch_numbers
        train_losses = train_losses if train_losses is not None else self.train_losses
        train_accuracies = (
            train_accuracies if train_accuracies is not None else self.train_accuracies
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(epoch_numbers, train_losses, color='tab:red')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

        axes[1].plot(epoch_numbers, train_accuracies, color='tab:blue')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

    def run(self):
        print('method running...')
        self._configure_from_data()

        print('--start training...')
        if self.task == 'classification':
            epoch_numbers, train_losses, test_losses, train_accuracies = self.train_model(
                self.data['train']['X'],
                self.data['train']['y'],
                self.data['test']['X'],
                self.data['test']['y']
            )
        else:
            epoch_numbers, train_losses, test_losses, train_accuracies = self.train_model(
                self.data['train']['X'],
                self.data['train']['y']
            )

        self.epoch_numbers = epoch_numbers
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.train_accuracies = train_accuracies

        print('--start testing...')
        if self.task == 'classification':
            pred_y, y_probs = self.test(self.data['test']['X'])
            true_y = self.data['test']['y']
            self.y_probs = y_probs.numpy()
        else:
            pred_y, y_probs = self.test(self.data['train']['X'], return_probs=False)
            true_y = self.data['train']['y']

        result = {
            'pred_y': pred_y,
            'true_y': true_y,
            'epoch_numbers': self.epoch_numbers,
            'train_losses': self.train_losses
        }
        if self.task == 'classification':
            result['y_score'] = self.y_probs
            result['test_losses'] = self.test_losses
        else:
            result['validation_losses'] = self.test_losses
            if self.test_losses:
                result['cross_entropy_loss'] = self.test_losses[-1]
            elif self.train_losses:
                result['cross_entropy_loss'] = self.train_losses[-1]

        return result
