'''
Concrete Evaluate class for text-generation metrics
'''

from local_code.base_class.evaluate import evaluate
import numpy as np
import torch
import torch.nn.functional as F


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating generation performance...')

        cross_entropy_loss = self._get_cross_entropy_loss()
        perplexity = float(np.exp(cross_entropy_loss))

        print('Cross entropy loss:', cross_entropy_loss)
        print('Perplexity:', perplexity)

        learning_curves = self._plot_learning_curves()

        return {
            'cross_entropy_loss': cross_entropy_loss,
            'perplexity': perplexity,
            'learning_curves': learning_curves
        }

    def _get_cross_entropy_loss(self):
        if 'cross_entropy_loss' in self.data:
            return float(self.data['cross_entropy_loss'])

        if 'loss' in self.data:
            return float(self.data['loss'])

        y_true = self.data['true_y']
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.long()
        else:
            y_true = torch.LongTensor(np.array(y_true))

        if 'logits' in self.data:
            logits = self.data['logits']
            if not isinstance(logits, torch.Tensor):
                logits = torch.FloatTensor(np.array(logits))
            return float(F.cross_entropy(logits, y_true).item())

        if 'y_score' in self.data:
            probabilities = self.data['y_score']
            if not isinstance(probabilities, torch.Tensor):
                probabilities = torch.FloatTensor(np.array(probabilities))

            probabilities = torch.clamp(probabilities, min=1e-12)
            log_probabilities = torch.log(probabilities)
            return float(F.nll_loss(log_probabilities, y_true).item())

        raise ValueError(
            "Generation evaluation requires 'logits', 'y_score', "
            "'cross_entropy_loss', or 'loss' in self.data."
        )

    def _plot_learning_curves(self):
        train_losses = self.data.get('train_losses')
        validation_losses = self.data.get('validation_losses')
        if validation_losses is None:
            validation_losses = self.data.get('val_losses')
        if validation_losses is None:
            validation_losses = self.data.get('test_losses')

        if train_losses is None or validation_losses is None:
            return None

        epoch_numbers = self.data.get('epoch_numbers')
        if epoch_numbers is None:
            epoch_numbers = list(range(1, len(train_losses) + 1))

        train_losses = self._to_list(train_losses)
        validation_losses = self._to_list(validation_losses)
        epoch_numbers = self._to_list(epoch_numbers)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(epoch_numbers, train_losses, label='Training Loss')
        plt.plot(epoch_numbers, validation_losses, label='Validation Loss')
        plt.title('Training and Validation Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

        return {
            'epoch_numbers': epoch_numbers,
            'train_losses': train_losses,
            'validation_losses': validation_losses
        }

    def _to_list(self, values):
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        if hasattr(values, 'tolist'):
            values = values.tolist()
        return list(values)
