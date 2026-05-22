from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Result_Saver import Result_Saver

import numpy as np
import torch


#Change this to 'generation' to train on the joke-generation dataset.
TASK = 'generation'


if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    
    #EDIT HERE, do not not touch epoch and learning rate values in Method_RNN.py
    max_epoch = 10 ##will overfit if too large because our datatset is not big
    learning_rate = 5e-4
    embedding_dim = 256
    hidden_size = 256
    batch_size = 64
    num_layers = 1
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
            min_word_frequency=2
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
        num_layers=num_layers,
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
        print('Generated example 1:')
        print(method_obj.generate_joke('Why did the', max_new_words=20))
        print('Generated example 2:')
        print(method_obj.generate_joke('What do you', max_new_words=20)) 
        print('Generated example 3:')
        print(method_obj.generate_joke('How many does', max_new_words=20)) 
        print('Generated example 4:')
        print(method_obj.generate_joke('A man walks', max_new_words=20)) 
        print('Generated example 5:')
        print(method_obj.generate_joke('Knock knock, whos', max_new_words=20)) 
        print('Generated example 6:')
        print(method_obj.generate_joke('What did the', max_new_words=20)) 
        print('Generated example 7:')
        print(method_obj.generate_joke('Why does the', max_new_words=20)) 
        print('Generated example 8:')
        print(method_obj.generate_joke('Have you heard', max_new_words=20))
        #can change max_new_words to whatever you wish

    print('************ Finish ************')
    # ------------------------------------------------------
