from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy

import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2) #Neural networks usually start with random weights, these fix the randomness
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- object initialization section -------------------
    data_obj = Dataset_Loader(
        'digit dataset',
        'MNIST-style csv dataset',
        folder_path='./data/stage_2_data', #changed data folder pathway
        file_name='train.csv'
    )

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_2_result/MLP_' #changed result folder pathway so it is inside of project folder
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')
    setting_obj.test_dataset = Dataset_Loader(
        'digit test dataset',
        'MNIST-style csv test set',
        folder_path='./data/stage_2_data', #changed data folder pathway to be accurate
        file_name='test.csv'
    )

    evaluate_obj = Evaluate_Accuracy('multi-metric evaluator', '') #changed accuracy to multi-metric
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    result = setting_obj.load_run_save_evaluate() #changed to result since no longer doing k-fold
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(result)) #changed to only print result because there is no std
    print('************ Finish ************')
    # ------------------------------------------------------
    