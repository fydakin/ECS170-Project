from local_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_5_code.Result_Loader import Result_Loader


DATASETS = ['cora', 'citeseer', 'pubmed']


if __name__ == '__main__':
    for dataset_name in DATASETS:
        result_obj = Result_Loader('loader', '')
        result_obj.result_destination_folder_path = './result/stage_5_result'
        result_obj.result_destination_file_name = 'gnn_prediction_result'
        result_obj.result_dataset_name = dataset_name
        result_obj.result_model_name = 'gnn'
        result_obj.result_task_name = 'node_classification'

        try:
            result = result_obj.load()
        except FileNotFoundError:
            print(f'No saved stage 5 GNN result found for {dataset_name}.')
            continue

        print('************ Loaded Result ************')
        print('Dataset:', dataset_name)

        evaluate_obj = Evaluate_Accuracy('evaluator', '')
        evaluate_obj.data = result
        metrics = evaluate_obj.evaluate()

        print('************ Metrics ************')
        print(metrics)
