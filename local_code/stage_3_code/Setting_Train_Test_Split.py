'''
Concrete SettingModule class for a specific experimental SettingModule
'''

from local_code.base_class.setting import setting

#Changed to fit MNIST (already contains both splits)
class Setting_Train_Test_Split(setting):
    test_dataset = None

    def load_run_save_evaluate(self):

        loaded_data = self.dataset.load()

        self.method.data = {
        'train': loaded_data['train'],
        'test': loaded_data['test']
        }

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.fold_count = 1
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate()

        