import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Result_Saver import Result_Saver

import numpy as np
import torch


#Change this to 'generation' to train on the joke-generation dataset.
TASK = 'classification'


if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    
    #EDIT HERE, do not not touch epoch and learning rate values in Method_RNN.py
    max_epoch = 10 ##will overfit if too large because our datatset is not big
    learning_rate = 1e-3
    embedding_dim = 128
    hidden_size = 128
    batch_size = 64
    # ------------------------------------------------------

    # ---- object initialization section -------------------
    if TASK == 'classification':
        from local_code.stage_4_code.Evaluate_Accuracy_Classification import Evaluate_Accuracy

        data_obj = Dataset_Loader(
            'IMDb sentiment dataset',
            'binary movie review sentiment classification dataset',
            folder_path='./data/stage_4_data',
            file_name='text_classification',
            max_sequence_length=500,
            min_word_frequency=2,
            debug_mode=False,
            debug_size=200,
        )
        result_task_name = 'classification'

    elif TASK == 'generation':
        from local_code.stage_4_code.Evaluate_Accuracy_Generation import Evaluate_Accuracy

        data_obj = Dataset_Loader(
            'joke generation dataset',
            'short joke next-word generation dataset',
            folder_path='./data/stage_4_data',
            file_name='text_generation',
            min_word_frequency=1,
            generation_context_size=3
        )
        result_task_name = 'generation'

    else:
        raise ValueError("TASK must be either 'classification' or 'generation'.")

    method_obj = Method_RNN(
        'rnn',
        '',
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        batch_size=batch_size,
        task=TASK
    )
    method_obj.max_epoch = max_epoch
    method_obj.learning_rate = learning_rate

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_4_result'
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.result_task_name = result_task_name

    evaluate_obj = Evaluate_Accuracy('evaluator', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    print('Task:', TASK)

    data = data_obj.load()
    method_obj.data = data
    learned_result = method_obj.run()

    result_obj.data = learned_result
    result_obj.save()

    if TASK == 'generation':
        evaluate_obj.data = learned_result
    else:
        evaluate_obj.data = learned_result

    result = evaluate_obj.evaluate()

    print('************ Overall Performance ************')
    print(result)

    if TASK == 'generation':
        print('Generated example:')
        print(method_obj.generate_joke('what did the', max_new_words=30)) 
        #can change max_new_words to whatever you wish

    print('************ Finish ************')
    # ------------------------------------------------------
