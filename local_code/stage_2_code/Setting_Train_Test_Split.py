'''
Concrete SettingModule class for a specific experimental SettingModule
'''

from local_code.base_class.setting import setting


class Setting_Train_Test_Split(setting):
    test_dataset = None

    def load_run_save_evaluate(self):
        #took out code that split the dataset as its already been split for us
        #load pre-split training dataset
        train_data = self.dataset.load()

        #load pre-split testing dataset
        test_data = self.test_dataset.load()

        #run MethodModule
        self.method.data = {
            'train': {
                'X': train_data['X'],
                'y': train_data['y']
            },
            'test': {
                'X': test_data['X'],
                'y': test_data['y']
            }
        }

        learned_result = self.method.run()

        #save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None

        