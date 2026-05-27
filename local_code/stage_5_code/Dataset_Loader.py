from local_code.base_class.dataset import dataset
import os

import numpy as np


class Dataset_Loader(dataset):
    DATASET_CONFIGS = {
        'cora': {
            'num_features': 1433,
            'train_per_class': 20,
            'test_per_class': 150,
            'labels': [
                'Case_Based',
                'Genetic_Algorithms',
                'Neural_Networks',
                'Probabilistic_Methods',
                'Reinforcement_Learning',
                'Rule_Learning',
                'Theory'
            ]
        },
        'citeseer': {
            'num_features': 3703,
            'train_per_class': 20,
            'test_per_class': 200,
            'labels': [
                'AI',
                'Agents',
                'DB',
                'HCI',
                'IR',
                'ML'
            ]
        },
        'pubmed': {
            'num_features': 500,
            'train_per_class': 20,
            'test_per_class': 200,
            'labels': [
                '0',
                '1',
                '2'
            ]
        }
    }

    def __init__(
            self,
            dName=None,
            dDescription=None,
            folder_path='./data/stage_5_data',
            file_name='cora',
            random_state=2,
            add_self_loops=True,
            normalize_adjacency=True,
            dense_adjacency_max_nodes=10000):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = folder_path
        self.dataset_source_file_name = file_name
        self.random_state = random_state
        self.add_self_loops = add_self_loops
        self.normalize_adjacency = normalize_adjacency
        self.dense_adjacency_max_nodes = dense_adjacency_max_nodes

    def load(self):
        dataset_name = self.dataset_source_file_name.lower()
        dataset_path = self._resolve_dataset_path(dataset_name)
        print(f'loading data from {dataset_path}...')

        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown stage 5 dataset '{self.dataset_source_file_name}'. "
                "Use one of: cora, citeseer, pubmed."
            )

        node_path = os.path.join(dataset_path, 'node')
        link_path = os.path.join(dataset_path, 'link')
        if not os.path.isfile(node_path) or not os.path.isfile(link_path):
            raise FileNotFoundError(
                f"Expected 'node' and 'link' files under {dataset_path}."
            )

        config = self.DATASET_CONFIGS[dataset_name]
        features, labels, node_ids, label_map = self._load_nodes(
            node_path,
            config
        )
        edges = self._load_edges(link_path, node_ids)
        graph_edges, edge_weights = self._prepare_graph_edges(
            len(node_ids),
            edges
        )
        adjacency = self._build_dense_adjacency(
            len(node_ids),
            graph_edges,
            edge_weights
        )

        train_idx, test_idx = self._sample_train_test_indices(labels, config)
        train_mask = self._indices_to_mask(train_idx, len(node_ids))
        test_mask = self._indices_to_mask(test_idx, len(node_ids))

        return {
            'dataset_name': dataset_name,
            'X': features,
            'y': labels,
            'node_ids': node_ids,
            'node_id_to_index': {
                node_id: idx for idx, node_id in enumerate(node_ids)
            },
            'edge_index': graph_edges.T,
            'edge_weight': edge_weights,
            'edges': edges,
            'adjacency': adjacency,
            'train': {
                'X': features[train_idx],
                'y': labels[train_idx],
                'idx': train_idx,
                'mask': train_mask
            },
            'test': {
                'X': features[test_idx],
                'y': labels[test_idx],
                'idx': test_idx,
                'mask': test_mask
            },
            'label_map': label_map,
            'idx_to_label': {
                label_idx: label_name
                for label_name, label_idx in label_map.items()
            },
            'num_features': config['num_features'],
            'num_classes': len(config['labels'])
        }

    def _resolve_dataset_path(self, dataset_name):
        if os.path.isabs(self.dataset_source_file_name):
            return self.dataset_source_file_name

        direct_path = os.path.join(
            self.dataset_source_folder_path,
            self.dataset_source_file_name
        )
        if os.path.isdir(direct_path):
            return direct_path

        return os.path.join(self.dataset_source_folder_path, dataset_name)

    def _load_nodes(self, node_path, config):
        expected_fields = config['num_features'] + 2
        label_map = {
            label_name: idx
            for idx, label_name in enumerate(config['labels'])
        }

        node_ids = []
        features = []
        labels = []

        with open(node_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                parts = line.rstrip('\n').split('\t')
                if len(parts) != expected_fields:
                    raise ValueError(
                        f'{node_path}:{line_number} has {len(parts)} fields; '
                        f'expected {expected_fields}.'
                    )

                label_name = parts[-1]
                if label_name not in label_map:
                    raise ValueError(
                        f"{node_path}:{line_number} has unknown label "
                        f"'{label_name}'."
                    )

                node_ids.append(parts[0])
                features.append([float(value) for value in parts[1:-1]])
                labels.append(label_map[label_name])

        return (
            np.array(features, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            node_ids,
            label_map
        )

    def _load_edges(self, link_path, node_ids):
        node_id_to_index = {
            node_id: idx for idx, node_id in enumerate(node_ids)
        }
        edges = []

        with open(link_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                parts = line.rstrip('\n').split('\t')
                if len(parts) != 2:
                    raise ValueError(
                        f'{link_path}:{line_number} has {len(parts)} fields; '
                        'expected 2.'
                    )

                target_id, source_id = parts
                if source_id not in node_id_to_index:
                    raise ValueError(
                        f"{link_path}:{line_number} source node "
                        f"'{source_id}' is missing from node file."
                    )
                if target_id not in node_id_to_index:
                    raise ValueError(
                        f"{link_path}:{line_number} target node "
                        f"'{target_id}' is missing from node file."
                    )

                # README notation is "A B", meaning B points to A.
                edges.append([
                    node_id_to_index[source_id],
                    node_id_to_index[target_id]
                ])

        return np.array(edges, dtype=np.int64)

    def _prepare_graph_edges(self, num_nodes, edges):
        graph_edges = edges

        if self.add_self_loops:
            self_loops = np.column_stack((
                np.arange(num_nodes, dtype=np.int64),
                np.arange(num_nodes, dtype=np.int64)
            ))
            graph_edges = np.vstack((graph_edges, self_loops))

        if not self.normalize_adjacency:
            return graph_edges, np.ones(graph_edges.shape[0], dtype=np.float32)

        out_degrees = np.bincount(
            graph_edges[:, 0],
            minlength=num_nodes
        ).astype(np.float32)
        out_degrees[out_degrees == 0] = 1.0
        edge_weights = 1.0 / out_degrees[graph_edges[:, 0]]

        return graph_edges, edge_weights.astype(np.float32)

    def _build_dense_adjacency(self, num_nodes, graph_edges, edge_weights):
        if (
                self.dense_adjacency_max_nodes is not None
                and num_nodes > self.dense_adjacency_max_nodes):
            return None

        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        if graph_edges.size:
            adjacency[graph_edges[:, 0], graph_edges[:, 1]] = edge_weights

        return adjacency

    def _sample_train_test_indices(self, labels, config):
        rng = np.random.default_rng(self.random_state)
        train_indices = []
        test_indices = []

        for label_idx in range(len(config['labels'])):
            class_indices = np.flatnonzero(labels == label_idx)
            required = config['train_per_class'] + config['test_per_class']

            if class_indices.size < required:
                raise ValueError(
                    f'Class {label_idx} has {class_indices.size} nodes; '
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

    def _indices_to_mask(self, indices, size):
        mask = np.zeros(size, dtype=bool)
        mask[indices] = True
        return mask
