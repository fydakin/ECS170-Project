'''
Concrete Evaluate class for binary text-classification metrics
'''

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import torch


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating classification performance...')

        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F1-score:', f1)
        print('Confusion Matrix:')
        print(cm)

        learning_curves = self._plot_learning_curves()

        return {
            'accuracy': accuracy,
            'acc': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'learning_curves': learning_curves
        }

    def _plot_learning_curves(self):
        train_losses = self.data.get('train_losses')
        test_losses = self.data.get('test_losses')
        if test_losses is None:
            test_losses = self.data.get('validation_losses')
        if test_losses is None:
            test_losses = self.data.get('val_losses')

        if train_losses is None or test_losses is None:
            return None

        epoch_numbers = self.data.get('epoch_numbers')
        if epoch_numbers is None:
            epoch_numbers = list(range(1, len(train_losses) + 1))

        train_losses = self._to_list(train_losses)
        test_losses = self._to_list(test_losses)
        epoch_numbers = self._to_list(epoch_numbers)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(epoch_numbers, train_losses, label='Training Loss')
        plt.plot(epoch_numbers, test_losses, label='Validation/Test Loss')
        plt.title('Training and Validation/Test Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

        return {
            'epoch_numbers': epoch_numbers,
            'train_losses': train_losses,
            'test_losses': test_losses
        }

    def _to_list(self, values):
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        if hasattr(values, 'tolist'):
            values = values.tolist()
        return list(values)
