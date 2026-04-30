from local_code.base_class.dataset import dataset
import numpy as np
import os
import pickle


class Dataset_Loader(dataset):
    def __init__(self, dName=None, dDescription=None, folder_path='', file_name=''):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = folder_path
        self.dataset_source_file_name = file_name

    def load(self):
        print(f'loading data from {self.dataset_source_file_name}...')
        X = []
        y = []

        file_path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)

        with open(file_path, 'rb') as f:
          data = pickle.load(f)

        X_train, y_train = [], []
        for instance in data['train']:
            img   = np.array(instance['image'], dtype=np.float32)  # (112, 92, 3)
            label = instance['label'] - 1                          # shift 1-40 → 0-39
            X_train.append(img)
            y_train.append(label)

        X_test, y_test = [], []
        for instance in data['test']:
            img   = np.array(instance['image'], dtype=np.float32)
            label = instance['label'] - 1
            X_test.append(img)
            y_test.append(label)

        # Stack into arrays
        X_train = np.stack(X_train) / 255.0   # (360, 112, 92, 3)
        X_test  = np.stack(X_test)  / 255.0   # (40,  112, 92, 3)

        # Transpose to PyTorch format: (N, C, H, W)
        X_train = X_train.transpose(0, 3, 1, 2)  # (360, 3, 112, 92)
        X_test  = X_test.transpose(0, 3, 1, 2)   # (40,  3, 112, 92)

        return {
            'train': {
                'X': X_train,
                'y': np.array(y_train, dtype=np.int64)
            },
            'test': {
                'X': X_test,
                'y': np.array(y_test, dtype=np.int64)
            }
        }