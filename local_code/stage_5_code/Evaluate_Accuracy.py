'''
Concrete Evaluate class for stage 5 GNN node-classification metrics
'''

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score
)
import numpy as np
import torch


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating GNN node-classification performance...')

        y_true = self._to_numpy(self.data['true_y']).astype(np.int64)
        y_pred = self._to_numpy(self.data['pred_y']).astype(np.int64)

        labels = self._labels(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred, labels=labels)

        precision_macro = precision_score(
            y_true, y_pred, labels=labels, average='macro', zero_division=0
        )
        precision_weighted = precision_score(
            y_true, y_pred, labels=labels, average='weighted', zero_division=0
        )
        precision_micro = precision_score(
            y_true, y_pred, labels=labels, average='micro', zero_division=0
        )

        recall_macro = recall_score(
            y_true, y_pred, labels=labels, average='macro', zero_division=0
        )
        recall_weighted = recall_score(
            y_true, y_pred, labels=labels, average='weighted', zero_division=0
        )
        recall_micro = recall_score(
            y_true, y_pred, labels=labels, average='micro', zero_division=0
        )

        f1_macro = f1_score(
            y_true, y_pred, labels=labels, average='macro', zero_division=0
        )
        f1_weighted = f1_score(
            y_true, y_pred, labels=labels, average='weighted', zero_division=0
        )
        f1_micro = f1_score(
            y_true, y_pred, labels=labels, average='micro', zero_division=0
        )

        per_class = self._per_class_metrics(y_true, y_pred, labels)
        loss_curves = self._loss_curves()

        print('Accuracy:', accuracy)
        print('Precision (macro):', precision_macro)
        print('Precision (weighted):', precision_weighted)
        print('Precision (micro):', precision_micro)
        print('Recall (macro):', recall_macro)
        print('Recall (weighted):', recall_weighted)
        print('Recall (micro):', recall_micro)
        print('F1-score (macro):', f1_macro)
        print('F1-score (weighted):', f1_weighted)
        print('F1-score (micro):', f1_micro)
        print('Confusion Matrix:')
        print(confusion)

        return {
            'accuracy': accuracy,
            'acc': accuracy,
            'precision': precision_macro,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'precision_micro': precision_micro,
            'recall': recall_macro,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'recall_micro': recall_micro,
            'f1_score': f1_macro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_micro': f1_micro,
            'per_class': per_class,
            'confusion_matrix': confusion,
            'labels': labels,
            'loss_curves': loss_curves
        }

    def _to_numpy(self, values):
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy()
        if hasattr(values, 'detach'):
            values = values.detach()
        if hasattr(values, 'cpu'):
            values = values.cpu()
        if hasattr(values, 'numpy'):
            values = values.numpy()
        return np.asarray(values)

    def _labels(self, y_true, y_pred):
        if 'labels' in self.data and self.data['labels'] is not None:
            return list(self.data['labels'])
        return sorted(np.unique(np.concatenate((y_true, y_pred))).tolist())

    def _per_class_metrics(self, y_true, y_pred, labels):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0
        )
        metrics = {}

        for idx, label in enumerate(labels):
            metrics[label] = {
                'precision': float(precision[idx]),
                'recall': float(recall[idx]),
                'f1_score': float(f1[idx]),
                'support': int(support[idx])
            }

        return metrics

    def _loss_curves(self):
        train_losses = self.data.get('train_losses')
        test_losses = self.data.get('test_losses')

        if train_losses is None and test_losses is None:
            return None

        train_losses = self._to_list(train_losses)
        test_losses = self._to_list(test_losses)

        epoch_numbers = self.data.get('epoch_numbers')
        if epoch_numbers is None:
            curve_length = max(len(train_losses), len(test_losses))
            epoch_numbers = list(range(1, curve_length + 1))
        else:
            epoch_numbers = self._to_list(epoch_numbers)

        curves = {
            'epoch_numbers': epoch_numbers,
            'train_losses': train_losses,
            'test_losses': test_losses
        }

        self._plot_loss_curves(curves)
        return curves

    def _to_list(self, values):
        if values is None:
            return []
        values = self._to_numpy(values)
        if values.ndim == 0:
            return [float(values)]
        return values.tolist()

    def _plot_loss_curves(self, curves):
        if not curves['train_losses'] and not curves['test_losses']:
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        plt.figure(figsize=(8, 5))

        if curves['train_losses']:
            plt.plot(
                curves['epoch_numbers'][:len(curves['train_losses'])],
                curves['train_losses'],
                label='Training Loss'
            )

        if curves['test_losses']:
            plt.plot(
                curves['epoch_numbers'][:len(curves['test_losses'])],
                curves['test_losses'],
                label='Testing Loss'
            )

        plt.title('GNN Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
