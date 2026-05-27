'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np


class Setting_Train_Test_Split(setting):
    SPLIT_CONFIGS = {
        'cora': {
            'train_per_class': 20,
            'test_per_class': 150
        },
        'citeseer': {
            'train_per_class': 20,
            'test_per_class': 200
        },
        'pubmed': {
            'train_per_class': 20,
            'test_per_class': 200
        }
    }

    random_state = 2

    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()

        if self._has_prepared_split(loaded_data):
            train_data = loaded_data['train']
            test_data = loaded_data['test']
        else:
            train_idx, test_idx = self._sample_readme_split(loaded_data)
            train_data = {
                'X': loaded_data['X'][train_idx],
                'y': loaded_data['y'][train_idx],
                'idx': train_idx,
                'mask': self._indices_to_mask(train_idx, len(loaded_data['y']))
            }
            test_data = {
                'X': loaded_data['X'][test_idx],
                'y': loaded_data['y'][test_idx],
                'idx': test_idx,
                'mask': self._indices_to_mask(test_idx, len(loaded_data['y']))
            }
            loaded_data['train'] = train_data
            loaded_data['test'] = test_data

        self.method.data = loaded_data
        self.method.data['train'] = train_data
        self.method.data['test'] = test_data

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None

    def _has_prepared_split(self, loaded_data):
        return (
            'train' in loaded_data
            and 'test' in loaded_data
            and 'X' in loaded_data['train']
            and 'y' in loaded_data['train']
            and 'X' in loaded_data['test']
            and 'y' in loaded_data['test']
        )

    def _sample_readme_split(self, loaded_data):
        dataset_name = self._dataset_name(loaded_data)
        if dataset_name not in self.SPLIT_CONFIGS:
            raise ValueError(
                f"Unknown stage 5 dataset '{dataset_name}'. "
                "Use one of: cora, citeseer, pubmed."
            )

        labels = loaded_data['y']
        config = self.SPLIT_CONFIGS[dataset_name]
        rng = np.random.default_rng(self.random_state)
        train_indices = []
        test_indices = []

        for label in sorted(np.unique(labels)):
            class_indices = np.flatnonzero(labels == label)
            required = config['train_per_class'] + config['test_per_class']

            if class_indices.size < required:
                raise ValueError(
                    f'Class {label} has {class_indices.size} nodes; '
                    f'{required} are needed for the configured split.'
                )

            sampled = rng.permutation(class_indices)
            train_end = config['train_per_class']
            test_end = train_end + config['test_per_class']
            train_indices.extend(sampled[:train_end])
            test_indices.extend(sampled[train_end:test_end])

        return (
            np.array(train_indices, dtype=np.int64),
            np.array(test_indices, dtype=np.int64)
        )

    def _dataset_name(self, loaded_data):
        if 'dataset_name' in loaded_data:
            return loaded_data['dataset_name'].lower()

        dataset_file_name = getattr(
            self.dataset,
            'dataset_source_file_name',
            ''
        )
        return dataset_file_name.lower()

    def _indices_to_mask(self, indices, size):
        mask = np.zeros(size, dtype=bool)
        mask[indices] = True
        return mask
