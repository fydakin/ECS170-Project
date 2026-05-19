'''
Concrete ResultModule class for stage 4 experiment output
'''

from local_code.base_class.result import result
import os
import pickle


class Result_Loader(result):
    data = None
    fold_count = None
    result_destination_folder_path = './result/stage_4_result'
    result_destination_file_name = 'prediction_result'
    result_task_name = None

    def load(self):
        print('loading results...')
        file_path = self._get_result_file_path()

        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

        print(f'results loaded from {file_path}')
        return self.data

    def _get_result_file_path(self):
        folder_path = self.result_destination_folder_path
        file_name = self.result_destination_file_name

        if self.result_task_name:
            file_name += '_' + str(self.result_task_name)

        if self.fold_count is not None:
            file_name += '_' + str(self.fold_count)

        if not file_name.endswith('.pkl'):
            file_name += '.pkl'

        return os.path.join(folder_path, file_name)
