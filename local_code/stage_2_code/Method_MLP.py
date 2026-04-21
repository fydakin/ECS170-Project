
from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 20 #change if necessary
    learning_rate = 1e-3 #change if necessary

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.epoch_numbers = []
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracy = None

        # Input = 784 features
        # Output = 10 classes

        #Input (784) → Hidden (256) → Hidden (128) → Output (10)
        self.fc_layer_1 = nn.Linear(784, 256) #model input must accept 784 numbers (28 x 28 images)
        self.activation_func_1 = nn.ReLU()

        self.fc_layer_2 = nn.Linear(256, 128) #256 and 128 features (these can be changed)
        self.activation_func_2 = nn.ReLU()

        self.fc_layer_3 = nn.Linear(128, 10) #there are 10 classes

    def forward(self, x):
        '''Forward propagation'''
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.activation_func_2(self.fc_layer_2(h1))
        y_pred = self.fc_layer_3(h2)   #outputs raw scores (logits), not probabilities
        return y_pred

    def train_model(self, X, y):  #Name changed due to error in switching to eval mode
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) #Adam = optimization algorithm (updates weights)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))

        epoch_numbers = []  #Data collection during training added for loss plots
        train_losses = []
        train_accuracies = []

        for epoch in range(self.max_epoch):
            y_pred = self.forward(X_tensor)
            train_loss = loss_function(y_pred, y_tensor)

            #backpropogation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % 1 == 0:
                accuracy_evaluator.data = {
                    'true_y': y_tensor,
                    'pred_y': y_pred.max(1)[1]
                }
                print('Epoch:', epoch,
                      'Accuracy:', accuracy_evaluator.evaluate(),
                      'Loss:', train_loss.item())
            epoch_numbers.append(epoch + 1)
            train_losses.append(train_loss.item())
            train_accuracies.append(accuracy_evaluator.evaluate()['acc'])

        return epoch_numbers, train_losses, train_accuracies

    def test(self, X):
        self.eval()  # switch to evaluation mode
        X_tensor = torch.FloatTensor(np.array(X))
    
        with torch.no_grad():  #no gradients needed as during testing, we don’t update weights
            y_pred = self.forward(X_tensor)

        return y_pred.max(1)[1]
    
    def plot_metrics(self, epoch_numbers, train_losses, train_accuracies, test_accuracy):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

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

        plt.tight_layout()
        #plt.savefig('training_metrics.png', dpi=150)
        plt.show()

    def run(self):
        print('method running...')
        print('--start training...')
        epoch_numbers, train_losses, train_accuracies = self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        self.epoch_numbers = epoch_numbers
        self.train_losses = train_losses
        self.train_accuracies = train_accuracies

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}