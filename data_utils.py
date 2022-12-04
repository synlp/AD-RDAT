import os
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, segment_ids, label_ids, valid_ids=None, text=None, label=None,
                 word_bound=None, word_list=None):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.valid_ids = valid_ids
        self.word_bound = word_bound
        self.text = text
        self.label = label
        self.word_list = word_list

class DiaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, data_dir, dataset):
        self.data_dir = data_dir
        self.dataset = dataset

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "{}_train.tsv".format(self.dataset))), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "{}_dev.tsv".format(self.dataset))), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "{}_test.tsv".format(self.dataset))), "test")

    def get_farasa_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "{}_train_farasa.tsv".format(self.dataset))), "test")

    def get_labels(self):
        return ['#', 'a', 'i', 'o', 'u', 'K', 'F', 'N', '~','~a','~i', '~u', '~K', '~F', '~N', '[CLS]', '[SEP]', '-']

    def get_label_map(self):
        label_map = {label: i for i, label in enumerate(self.get_labels(), 1)}
        label_map['[unk]'] = 0
        return label_map

    def get_id2label_map(self):
        return {i: label for label, i in self.get_label_map().items()}

    def get_tag_size(self):
        return len(self.get_labels()) + 1

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text=sentence, label=label))
        return examples

    @classmethod
    def _read_tsv(cls, input_file):
        '''
        read file
        return format :
        '''
        if os.path.exists(input_file) is False:
            return []
        data = []
        sentence = []
        label = []
        with open(input_file, "r", encoding = "utf-8") as f:
            for line in f:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                    if len(sentence) > 0:
                        data.append((sentence, label))
                        sentence = []
                        label = []
                    continue
                splits = line.strip().split('\t')
                sentence.append(splits[0])
                label.append(splits[-1])

            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
        return data

    def convert_to_feature(self, tokenizer, examples, max_seq_length=512):
        label_map = self.get_label_map()
        features = []
        for ex_index, example in enumerate(examples):
            word_sequence = [""]
            for i, word in enumerate(example.text):
                if word == "[Sep]":
                    word_sequence.append("")
                    continue
                word_sequence[-1] += word

            word_sequence_offset = 0
            w_tokens_offset_w = 0
            w_tokens_offset_c = 0

            c_tokens = []
            w_tokens = []
            c2w_map = []
            labels = []
            valid_ids = []
            word_bound = []
            word_list = []

            if len(w_tokens) == 0:
                w_tokens_offset_w = len(w_tokens)
                w_tokens_offset_c = 0
                w_tokens.extend(tokenizer.tokenize(word_sequence[word_sequence_offset]))
                word_sequence_offset += 1
            for word,label in zip(example.text, example.label):
                if word == "[Sep]":
                    if len(word_bound) > 0:
                        word_bound[-1] = 1
                    w_tokens_offset_w = len(w_tokens)
                    w_tokens_offset_c = 0
                    w_tokens.extend(tokenizer.tokenize(word_sequence[word_sequence_offset]))
                    word_sequence_offset += 1
                    if len(w_tokens) > max_seq_length - 2:
                        w_tokens = w_tokens[:max_seq_length - 2]
                    continue

                if w_tokens_offset_c + len(word) > len(w_tokens[w_tokens_offset_w]) and w_tokens_offset_w+1 < len(w_tokens):
                    w_tokens_offset_w += 1
                    w_tokens_offset_c = 0
                if w_tokens_offset_c == 0 and w_tokens[w_tokens_offset_w][:2] == "##":
                    w_tokens_offset_c = 2
                w_tokens_offset_c += len(word)

                token = tokenizer.tokenize(word)
                if len(c_tokens) + len(token) > max_seq_length - 2:
                    break
                c_tokens.extend(token)
                labels.append(label)
                word_list.append(word)
                word_bound.append(0)
                for m in range(len(token)):
                    if m == 0:
                        valid_ids.append(1)
                    else:
                        valid_ids.append(0)
                    c2w_map.append(w_tokens_offset_w+1)

            word_bound[-1] = 1

            c_tokens = ["[CLS]"] + c_tokens + ["[SEP]"]
            w_tokens = ["[CLS]"] + w_tokens + ["[SEP]"]
            word_list = ["[CLS]"] + word_list + ["[SEP]"]
            labels = ["[CLS]"] + labels + ["[SEP]"]
            c2w_map = [0] + c2w_map + [len(w_tokens)-1]
            word_bound = [1] + word_bound + [1]
            valid_ids = [1] + valid_ids + [1]
            input_ids = tokenizer.convert_tokens_to_ids(c_tokens)
            wordpiece_ids = tokenizer.convert_tokens_to_ids(w_tokens)
            label_ids = [label_map[label] for label in labels]
            segment_ids = [0] * max_seq_length

            if len(input_ids) < max_seq_length:
                input_ids += [0] * (max_seq_length - len(input_ids))
                valid_ids += [0] * (max_seq_length - len(valid_ids))
                c2w_map += [0] * (max_seq_length - len(c2w_map))

            if len(wordpiece_ids) < max_seq_length:
                wordpiece_ids += [0] * (max_seq_length - len(wordpiece_ids))

            if len(label_ids) <  max_seq_length:
                label_ids += [0] * (max_seq_length - len(label_ids))
                word_bound += [0] * (max_seq_length - len(word_bound))

            assert len(input_ids) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid_ids) == max_seq_length
            assert len(wordpiece_ids) == max_seq_length
            assert len(c2w_map) == max_seq_length

            features.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "wordpiece_ids": torch.tensor(wordpiece_ids, dtype=torch.long),
                "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
                "label_ids": torch.tensor(label_ids, dtype=torch.long),
                "valid_ids": torch.tensor(valid_ids, dtype=torch.long),
                "word_bound": torch.tensor(word_bound, dtype=torch.long),
                "c2w_map": torch.tensor(c2w_map, dtype=torch.long),
                "text": example.text,
                "label": example.label,
                "word_list": "\t".join(word_list),
            })

        return features

    def get_dataloader(self, features, batch_size, mode='train', rank=0,  world_size=1):
        if mode == "train" and world_size > 1:
            features = features[rank::world_size]

        data_set = DiaDataset(features)
        sampler = RandomSampler(data_set)
        return DataLoader(data_set, sampler=sampler, batch_size=batch_size)

    def get_all_dataloader(self, tokenizer, args):
        #train
        train_examples = self.get_train_examples()
        train_features = self.convert_to_feature(tokenizer, train_examples, args.max_seq_len)
        train_data_loader = self.get_dataloader(train_features, mode="train", rank=args.rank,
                                                    world_size=args.world_size, batch_size=args.batch_size)

        # train farasa
        farasa_examples = self.get_farasa_examples()
        if len(farasa_examples) > len(train_examples):
            farasa_examples = farasa_examples[:len(train_examples)]
        elif len(farasa_examples) < len(train_examples):
            farasa_examples = farasa_examples + farasa_examples[:len(train_examples)-len(farasa_examples)]
        farasa_features = self.convert_to_feature(tokenizer, farasa_examples, args.max_seq_len)
        farasa_data_loader = self.get_dataloader(farasa_features, mode="train", rank=args.rank,
                                                world_size=args.world_size, batch_size=args.batch_size)

        #dev
        dev_examples = self.get_dev_examples()
        dev_features = self.convert_to_feature(tokenizer, dev_examples, args.max_seq_len)
        dev_dataloader = self.get_dataloader(dev_features, mode="dev", batch_size=args.batch_size)

        return train_data_loader, farasa_data_loader, dev_dataloader