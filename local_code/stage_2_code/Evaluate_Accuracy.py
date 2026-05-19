'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
import numpy as np

class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')

        y_true = self.data['true_y'] #labels from test set
        y_pred = self.data['pred_y'] #labels predicted by the model

        #used macro to average the metrics equally
        acc = accuracy_score(y_true, y_pred)
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_micro = precision_score(y_true, y_pred, average='micro')

        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_micro = recall_score(y_true, y_pred, average='micro')


        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')

        auc_macro = None
        if 'y_score' in self.data and self.data['y_score'] is not None:
            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)
            y_score = self.data['y_score']
            if hasattr(y_score, 'numpy'):
                y_score = y_score.numpy()
            auc_macro = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')
            print("AUC (macro, OvR):", auc_macro)

        #prints the metrics 
        print("Accuracy:",acc)
        print("Precision (weighted):",precision_weighted)
        print("Precision (macro):",precision_macro)
        print("Precision (micro):",precision_micro)
        print("Recall (weighted):",recall_weighted)
        print("Recall (macro):",recall_macro)
        print("Recall (micro):",recall_micro)
        print("F1 (weighted):", f1_weighted)
        print("F1 (macro):", f1_macro)
        print("F1 (micro):", f1_micro)


        #returns all metrics
        return {
            'acc':                acc,
            'f1_weighted':        f1_weighted,
            'f1_macro':           f1_macro,
            'f1_micro':           f1_micro,
            'recall_weighted':    recall_weighted,
            'recall_macro':       recall_macro,
            'recall_micro':       recall_micro,
            'precision_weighted': precision_weighted,
            'precision_macro':    precision_macro,
            'precision_micro':    precision_micro,
        }