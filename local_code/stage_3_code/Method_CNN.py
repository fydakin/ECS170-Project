from local_code.base_class import dataset
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 50 #change if necessary
    learning_rate = 1e-2 #change if necessary
    #changed __init__ to fit MNIST
    def __init__(self, mName, mDescription,
             input_channels=1,
             num_classes=10,
             input_height=28,
             input_width=28):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.epoch_numbers = []
        self.train_losses = []
        self.train_accuracies = []

        self.max_epoch = 20
        self.learning_rate = 0.001

         # CNN Layers
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.5)

        # MNIST size calculations:
        # 28x28
        # after pool -> 14x14
        # after second pool -> 7x7

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        #already put in datatset loader
         # x shape coming in: (batch, 3, H, W) — take just 1 channel
        #x = x[:, 0:1, :, :] # → (batch, 1, H, W)

        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # → (batch, 32, H/2, W/2)
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # → (batch, 64, H/4, W/4)

        x = torch.flatten(x, 1) # → (batch, 64*H/4*W/4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # → (batch, num_classes)
        return x

    def train_model(self, X, y):  #Name changed due to error in switching to eval mode
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) #Adam = optimization algorithm (updates weights)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))

        dataset    = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        epoch_numbers = []  #Data collection during training added for loss plots
        train_losses = []
        train_accuracies = []

        for epoch in range(self.max_epoch):
            self.train()  # set to train mode each epoch
            epoch_loss = 0

            for X_batch, y_batch in dataloader:
                y_pred = self.forward(X_batch)
                loss   = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # evaluate on full set each epoch
            self.eval()
            with torch.no_grad():
                y_pred_full = self.forward(X_tensor)

            accuracy_evaluator.data = {
                'true_y': y_tensor,
                'pred_y': y_pred_full.max(1)[1]
            }
            avg_loss = epoch_loss / len(dataloader)
            acc = accuracy_evaluator.evaluate()['acc']

            print(f'Epoch: {epoch}  Accuracy: {acc:.4f}  Loss: {avg_loss:.4f}')
            epoch_numbers.append(epoch + 1)
            train_losses.append(avg_loss)
            train_accuracies.append(acc)

        return epoch_numbers, train_losses, train_accuracies

    def test(self, X):
        self.eval()  # switch to evaluation mode
        X_tensor = torch.FloatTensor(np.array(X))
    
        with torch.no_grad():  #no gradients needed as during testing, we don’t update weights
            y_pred = self.forward(X_tensor)
            y_probs  = torch.softmax(y_pred, dim=1)

        return y_pred.max(1)[1], y_probs
    #changed classes in plot_metrics to 10
    def plot_metrics(self, epoch_numbers, train_losses, train_accuracies, y_true, y_probs, n_classes = 10):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epoch_numbers, train_losses, color='tab:red')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_xticks(np.arange(0, max(epoch_numbers) + 1, max(epoch_numbers)//10))

        axes[1].plot(epoch_numbers, train_accuracies, color='tab:blue')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, 1)
        axes[1].set_xticks(np.arange(0, max(epoch_numbers) + 1, max(epoch_numbers)//10))

        classes    = list(range(n_classes))
        y_true_bin = label_binarize(y_true, classes=classes)

        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for i, color in zip(classes, colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc     = auc(fpr, tpr)
            axes[2].plot(fpr, tpr, color=color, lw=1.2, alpha=0.6, label=f'Class {i} (AUC={roc_auc:.2f})')

        axes[2].plot([0, 1], [0, 1], 'k:', lw=1)
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1.05)
        axes[2].set_title('ROC Curve (One-vs-Rest)')
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        axes[2].legend(loc='lower right', fontsize=6, ncol=2)

        plt.tight_layout()
        #plt.savefig('training_metrics.png', dpi=150)
        plt.show()

    def run(self):
        print('method running...')
        print('--start training...')
        epoch_numbers, train_losses, train_accuracies = self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y, y_probs = self.test(self.data['test']['X'])

        self.epoch_numbers = epoch_numbers
        self.train_losses = train_losses
        self.train_accuracies = train_accuracies
        self.y_probs = y_probs.numpy()

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}