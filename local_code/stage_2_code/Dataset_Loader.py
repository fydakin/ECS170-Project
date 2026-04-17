from local_code.base_class.dataset import dataset
import numpy as np
import os


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

        with open(file_path, 'r') as f:
           for line in f:
                elements = [int(i) for i in line.strip().split(',')] #CSV is comma seperated
                y.append(elements[0]) #label is the first column
                X.append(elements[1:]) #remaining values are features

        X = np.array(X, dtype=np.float32) / 255.0   #normalized pixels to [0,1]
        y = np.array(y, dtype=np.int64)

        return {'X': X, 'y': y}