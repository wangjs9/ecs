# coding: UTF-8
import torch
from tqdm import tqdm
import time
import pickle as pkl
import os
from datetime import timedelta

UNK, PAD, CLS, END = '<UNK>', '<PAD>', '<CLS>', '<END>'  # padding符号, bert中综合信息符号

MAX_VOCAB_SIZE = 10000  # 词表长度限制

def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

def build_dataset(config, do_train, do_test, use_word=False, do_pred=False):
    if use_word:
        tokenizer = lambda x: x.split(' ')  # word-level
    else:
        tokenizer = lambda x: [y for y in x if y!=' ']  # char-level

    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=5)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                if len(lin.split()) == 1:
                    continue
                content, label = lin.split('\t')
                token_ids = []
                token = tokenizer(content)
                token = [CLS] + token +[END]
                seq_len = len(token)
                mask = []
                # word to id
                for word in token:
                    token_ids.append(vocab.get(word, vocab.get(UNK)))
                if pad_size:
                    if len(token_ids) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token = token[:pad_size]
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                
                contents.append((token_ids, int(label), seq_len, mask))
        return contents  # [([...], 0), ([...], 1), ...]
    train = None
    dev = None
    test = None
    if do_train:
        train = load_dataset(config.train_path, config.pad_size)
        dev = load_dataset(config.dev_path, config.pad_size)
    if do_test:
        test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def get_time_dif(start_time):
    """get time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))