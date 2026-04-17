'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')

        y_true = self.data['true_y'] #labels from test set
        y_pred = self.data['pred_y'] #labels predicted by the model

        #used macro to average the metrics equally
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        #prints the metrics 
        print("Accuracy:",acc)
        print("Precision (macro):",precision)
        print("Recall (macro):",recall)
        print("F1 (macro):", f1)

        #returns all metrics
        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        