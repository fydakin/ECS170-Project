from local_code.base_class.dataset import dataset
import csv
import html
import os
import re
from collections import Counter

import numpy as np


class Dataset_Loader(dataset):
    PAD_TOKEN = '<PAD>' #pad the shorter sentence with a special token
    UNK_TOKEN = '<UNK>' #means UNKNOWN WORD, to map unseen words

    def __init__(
            self,
            dName=None,
            dDescription=None,
            folder_path='./data/stage_4_data',
            file_name='text_classification',
            debug_mode=False,
            debug_size=200,
            max_sequence_length=500,
            min_word_frequency=1,
            generation_context_size=3):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = folder_path
        self.dataset_source_file_name = file_name
        self.max_sequence_length = max_sequence_length
        self.min_word_frequency = min_word_frequency
        self.generation_context_size = generation_context_size
        self.debug_mode = debug_mode
        self.debug_size = debug_size

    def load(self):
        dataset_path = self._resolve_dataset_path()
        print(f'loading data from {dataset_path}...')

        if self._is_text_classification_dataset(dataset_path):
            return self._load_text_classification(dataset_path)

        if self._is_text_generation_dataset(dataset_path):
            return self._load_text_generation(dataset_path)

        raise ValueError(
            'Could not identify the stage 4 dataset. Use file_name='
            "'text_classification' or 'text_generation', or point directly "
            'to one of those folders.'
        )

    def _resolve_dataset_path(self):
        if os.path.isabs(self.dataset_source_file_name):
            return self.dataset_source_file_name

        direct_path = os.path.join(
            self.dataset_source_folder_path,
            self.dataset_source_file_name
        )
        if os.path.exists(direct_path):
            return direct_path

        if os.path.exists(self.dataset_source_folder_path):
            return self.dataset_source_folder_path

        return direct_path

    def _is_text_classification_dataset(self, dataset_path):
        return all(
            os.path.isdir(os.path.join(dataset_path, split, label))
            for split in ('train', 'test')
            for label in ('neg', 'pos')
        )

    def _is_text_generation_dataset(self, dataset_path):
        return (
            os.path.isdir(dataset_path)
            and os.path.isfile(os.path.join(dataset_path, 'data'))
        ) or os.path.isfile(dataset_path)

    def _load_text_classification(self, dataset_path):
        train_texts, train_labels = self._read_classification_split(
            dataset_path,
            'train'
        )
        test_texts, test_labels = self._read_classification_split(
            dataset_path,
            'test'
        )

        train_tokens = [self._clean_and_tokenize(text) for text in train_texts]
        test_tokens = [self._clean_and_tokenize(text) for text in test_texts]
        vocab = self._build_vocab(train_tokens)

        X_train = self._pad_sequences(
            [self._tokens_to_ids(tokens, vocab) for tokens in train_tokens],
            self.max_sequence_length
        )
        X_test = self._pad_sequences(
            [self._tokens_to_ids(tokens, vocab) for tokens in test_tokens],
            self.max_sequence_length
        )

        if self.debug_mode:
            X_train = X_train[:self.debug_size]
            train_labels = train_labels[:self.debug_size]
            train_texts = train_texts[:self.debug_size]

            X_test = X_test[:self.debug_size]
            test_labels = test_labels[:self.debug_size]
            test_texts = test_texts[:self.debug_size]

        return {
            'train': {
                'X': X_train,
                'y': np.array(train_labels, dtype=np.int64),
                'raw_text': train_texts
            },
            'test': {
                'X': X_test,
                'y': np.array(test_labels, dtype=np.int64),
                'raw_text': test_texts
            },
            'vocab': vocab,
            'idx_to_word': self._invert_vocab(vocab),
            'label_map': {'neg': 0, 'pos': 1},
            'max_sequence_length': self.max_sequence_length,
            'pad_index': vocab[self.PAD_TOKEN],
            'unk_index': vocab[self.UNK_TOKEN]
        }

    def _read_classification_split(self, dataset_path, split):
        texts = []
        labels = []
        label_map = {'neg': 0, 'pos': 1}

        for label_name in ('neg', 'pos'):
            label_dir = os.path.join(dataset_path, split, label_name)
            for file_name in sorted(os.listdir(label_dir)):
                if not file_name.endswith('.txt'):
                    continue

                file_path = os.path.join(label_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(label_map[label_name])

        return texts, labels

    def _load_text_generation(self, dataset_path):
        data_file = dataset_path
        if os.path.isdir(dataset_path):
            data_file = os.path.join(dataset_path, 'data')

        jokes = self._read_jokes(data_file)
        joke_tokens = [self._clean_and_tokenize(joke) for joke in jokes]
        vocab = self._build_vocab(joke_tokens)

        X = []
        y = []
        for tokens in joke_tokens:
            token_ids = self._tokens_to_ids(tokens, vocab)
            if len(token_ids) <= self.generation_context_size:
                continue

            for idx in range(self.generation_context_size, len(token_ids)):
                X.append(token_ids[idx - self.generation_context_size:idx])
                y.append(token_ids[idx])

        return {
            'train': {
                'X': np.array(X, dtype=np.int64),
                'y': np.array(y, dtype=np.int64)
            },
            'jokes': jokes,
            'encoded_jokes': [
                np.array(self._tokens_to_ids(tokens, vocab), dtype=np.int64)
                for tokens in joke_tokens
            ],
            'vocab': vocab,
            'idx_to_word': self._invert_vocab(vocab),
            'context_size': self.generation_context_size,
            'pad_index': vocab[self.PAD_TOKEN],
            'unk_index': vocab[self.UNK_TOKEN]
        }

    def _read_jokes(self, data_file):
        jokes = []
        with open(data_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                joke = row.get('Joke')
                if joke:
                    jokes.append(joke)
        return jokes

    def _clean_and_tokenize(self, text):
        text = html.unescape(text)
        text = text.replace('<br />', ' ')
        text = text.lower()
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)

    def _build_vocab(self, tokenized_texts):
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        vocab = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1
        }

        for token, count in counter.most_common():
            if count >= self.min_word_frequency and token not in vocab:
                vocab[token] = len(vocab)

        return vocab

    def _tokens_to_ids(self, tokens, vocab):
        unk_index = vocab[self.UNK_TOKEN]
        return [vocab.get(token, unk_index) for token in tokens]

    def _pad_sequences(self, sequences, max_length):
        if max_length is None:
            max_length = max(len(sequence) for sequence in sequences)

        padded = np.zeros((len(sequences), max_length), dtype=np.int64)
        for idx, sequence in enumerate(sequences):
            truncated = sequence[:max_length]
            padded[idx, :len(truncated)] = truncated

        return padded

    def _invert_vocab(self, vocab):
        return {idx: token for token, idx in vocab.items()}

