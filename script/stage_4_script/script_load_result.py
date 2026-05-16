from local_code.stage_4_code.Result_Loader import Result_Loader


if 1:
    result_obj = Result_Loader('loader', '')
    result_obj.result_destination_folder_path = './result/stage_4_result'
    result_obj.result_destination_file_name = 'prediction_result'

    for task_name in ['classification', 'generation']:
        result_obj.result_task_name = task_name

        try:
            result = result_obj.load()
        except FileNotFoundError:
            print(f'No saved result found for {task_name}.')
            continue

        print('Task:', task_name)
        print('Result:', result)
