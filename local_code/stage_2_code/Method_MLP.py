
from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 20 #change if necessary
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Input = 784 features
        # Output = 10 classes
        self.fc_layer_1 = nn.Linear(784, 256) #model input must accept 784 numbers
        self.activation_func_1 = nn.ReLU()

        self.fc_layer_2 = nn.Linear(256, 128)
        self.activation_func_2 = nn.ReLU()

        self.fc_layer_3 = nn.Linear(128, 10) #there are 10 classes

    def forward(self, x):
        '''Forward propagation'''
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.activation_func_2(self.fc_layer_2(h1))
        y_pred = self.fc_layer_3(h2)   # raw logits, no softmax here
        return y_pred

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))

        for epoch in range(self.max_epoch):
            y_pred = self.forward(X_tensor)
            train_loss = loss_function(y_pred, y_tensor)

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

    def test(self, X):
        X_tensor = torch.FloatTensor(np.array(X))
        y_pred = self.forward(X_tensor)
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}