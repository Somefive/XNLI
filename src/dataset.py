import torch.utils.data
from dataloader import *
import numpy as np


class XLMDataset(torch.utils.data.Dataset):
    
    def __init__(self, dico, filenames, para=False, dataset_size=1000000, max_seq_len=64, alpha=0.5, vocab_size=20000, labeled=False):
        self.max_seq_len, self.vocab_size, self.para, self.labeled = max_seq_len, vocab_size, para, labeled
        self.corpuses, self.corpus_size, self.corpus_reset_pos, self.labels = [], [], [], []
        for filename in filenames:
            if not para:
                corpus, size = load_monolingual_data(filename, dico, max_seq_len=max_seq_len, maxlines=dataset_size)
            else:
                if not labeled:
                    corpus, size, reset_pos = load_parallel_data(filename[0], filename[1], dico, max_seq_len=max_seq_len, maxlines=dataset_size)
                else:
                    corpus, size, reset_pos, label = load_xnli_data(filename[0], filename[1], filename[2], dico, max_seq_len=max_seq_len, maxlines=dataset_size)
                    self.labels.append(label)
                self.corpus_reset_pos.append(reset_pos)
            self.corpuses.append(corpus)
            self.corpus_size.append(size)
        self.sample_prob = np.array(self.corpus_size) ** alpha
        self.sample_prob /= self.sample_prob.sum()
        self.size = sum(self.corpus_size)
        self.C = len(self.corpuses)
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        c = np.random.choice(self.C, p=self.sample_prob)
        idx = index % self.corpus_size[c]
        x = np.array(self.corpuses[c][idx], dtype=np.int)
        if not self.para:
            l = self.max_seq_len
            pos = np.arange(l)
        else:
            l, reset_pos_x = len(self.corpuses[c][idx]), self.corpus_reset_pos[c][idx]
            pos = np.concatenate([np.arange(reset_pos_x), np.arange(l - reset_pos_x)])
        if not self.labeled:
            p = np.random.rand(l)
            r = np.random.randint(BGN_IDX, self.vocab_size, (l,))
            y = (p < .12) * MASK_IDX + (p > .12) * (p < .135) * r + (p > .15) * x
            return y, x, (p < .15) * 1, l, pos
        else:
            return x, l, pos, self.labels[c][idx]

    def get_generator(self, params={}):
        if self.para:
            params['collate_fn'] = collate_fn if not self.labeled else collate_fn_labeled
        return torch.utils.data.DataLoader(self, **params)


def collate_fn(data):
    xs, ys, masks, ls, poss = zip(*data)
    xs = pad_sequence([torch.LongTensor(x) for x in xs], batch_first=True, padding_value=PAD_IDX)
    ys = pad_sequence([torch.LongTensor(y) for y in ys], batch_first=True, padding_value=PAD_IDX)
    masks = pad_sequence([torch.LongTensor(mask * 1) for mask in masks], batch_first=True, padding_value=0)
    poss = pad_sequence([torch.LongTensor(pos) for pos in poss], batch_first=True, padding_value=0)
    return xs, ys, masks, torch.LongTensor(ls), poss 

def collate_fn_labeled(data):
    xs, ls, poss, labels = zip(*data)
    xs = pad_sequence([torch.LongTensor(x) for x in xs], batch_first=True, padding_value=PAD_IDX)
    poss = pad_sequence([torch.LongTensor(pos) for pos in poss], batch_first=True, padding_value=0)
    return xs, torch.LongTensor(ls), poss, torch.LongTensor(labels)