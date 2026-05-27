from local_code.stage_5_code.Dataset_Loader import Dataset_Loader
from local_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_5_code.Method_GNN import Method_GNN
from local_code.stage_5_code.Result_Saver import Result_Saver
from local_code.stage_5_code.Setting_Train_Test_Split import Setting_Train_Test_Split

import numpy as np
import torch

#To run only one dataset, change it to, for example:
#DATASETS = ['cora']

DATASETS = ['cora', 'citeseer', 'pubmed']


DATASET_SETTINGS = {
    'cora': {
        'hidden_size': 64,
        'dropout': 0.5,
        'learning_rate': 1e-2,
        'weight_decay': 5e-4,
        'max_epoch': 200
    },
    'citeseer': {
        'hidden_size': 64,
        'dropout': 0.5,
        'learning_rate': 1e-2,
        'weight_decay': 5e-4,
        'max_epoch': 200
    },
    'pubmed': {
        'hidden_size': 64,
        'dropout': 0.5,
        'learning_rate': 1e-2,
        'weight_decay': 5e-4,
        'max_epoch': 200
    }
}


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    for dataset_name in DATASETS:
        params = DATASET_SETTINGS[dataset_name]

        data_obj = Dataset_Loader(
            dataset_name,
            f'{dataset_name} citation graph node-classification dataset',
            folder_path='./data/stage_5_data',
            file_name=dataset_name,
            random_state=2,
            add_self_loops=True,
            normalize_adjacency=True
        )

        method_obj = Method_GNN(
            'gnn',
            'two-layer GCN for citation graph node classification',
            hidden_size=params['hidden_size'],
            dropout=params['dropout'],
            normalize_edges=True
        )
        method_obj.max_epoch = params['max_epoch']
        method_obj.learning_rate = params['learning_rate']
        method_obj.weight_decay = params['weight_decay']

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = './result/stage_5_result'
        result_obj.result_destination_file_name = 'gnn_prediction_result'
        result_obj.result_dataset_name = dataset_name
        result_obj.result_model_name = 'gnn'
        result_obj.result_task_name = 'node_classification'

        evaluate_obj = Evaluate_Accuracy('evaluator', '')

        setting_obj = Setting_Train_Test_Split(
            'readme train-test split',
            'class-balanced train/test split specified by the stage 5 README'
        )
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

        print('************ Start ************')
        print('Dataset:', dataset_name)
        print('Model: 2-layer GCN (one hidden layer)')
        print('Hidden size:', params['hidden_size'])
        print('Dropout:', params['dropout'])
        print('Learning rate:', params['learning_rate'])
        print('Weight decay:', params['weight_decay'])
        print('Max epoch:', params['max_epoch'])

        result, _ = setting_obj.load_run_save_evaluate()

        print('************ Overall Performance ************')
        print(result)
        print('************ Finish ************')
