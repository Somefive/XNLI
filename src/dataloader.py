from collections import Counter
import torch
import torch.utils.data
import numpy as np
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = 0
UNK_IDX = 1
MASK_IDX = 2
POS_IDX = 3
BGN_IDX = 4

def load_codes(filenames, size=20000):
    counter = Counter()
    for filename in filenames:
        for line in open(filename):
            token, cnt = line.strip().split(' ')
            counter[token] += int(cnt)
    print('%d words loaded.' % len(counter))
    dico = dict()
    dico['<PAD>'] = PAD_IDX
    dico['<UNK>'] = UNK_IDX
    dico['<MASK>'] = MASK_IDX
    dico['<POS>'] = POS_IDX
    for token, cnt in counter.most_common(size-len(dico)):
        dico[token] = len(dico)
    return dico, counter


def load_monolingual_data(filename, dico, max_seq_len=64, maxlines=100000):
    data, n = [POS_IDX], 0
    for _, line in zip(range(maxlines), open(filename)):
        data.extend([dico[token] if token in dico else UNK_IDX for token in line.strip().split(' ')])
        data.append(POS_IDX)
        n += 1
    size = len(data) // max_seq_len * max_seq_len
    return np.array(data[:size]).reshape(-1, max_seq_len), n

# # max_seq_len=256, maxlines=5000000
# def load_corpuses(filenames, dico, max_seq_len=64, maxlines=100000):
#     corpuses = []
#     for filename in filenames:
#         data, length = [], []
#         for _, line in zip(range(maxlines), open(filename)):
#             entry = [POS_IDX]
#             entry.extend([dico[token] if token in dico else UNK_IDX for token in line.strip().split(' ')])
#             entry.append(POS_IDX)
#             entry = entry[:max_seq_len]
#             l = len(entry)
#             data.append(entry)
#             length.append(l)
#         corpuses.append((data, length))
#     return corpuses

class MTMDataset(torch.utils.data.Dataset):

    def __init__(self, vocab_filenames, text_filenames, dataset_size=100000, vocab_size=20000, max_seq_len=64, alpha=0.5):
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len
        self.dico, self.counter = load_codes(vocab_filenames, vocab_size)
        self.corpuses, self.corpus_size = [], []
        for filename in text_filenames:
            corpus, size = load_monolingual_data(filename, self.dico, max_seq_len=max_seq_len, maxlines=dataset_size)
            self.corpuses.append(corpus)
            self.corpus_size.append(size)
        self.sample_prob = np.array(self.corpus_size) ** alpha
        self.sample_prob /= self.sample_prob.sum()
        self.size = dataset_size
        self.C = len(self.corpuses)
        self.counter = np.zeros(self.C, dtype=np.int)
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        c = np.random.choice(self.C)
        idx = self.counter[c] % self.corpus_size[c]
        self.counter[c] += 1
        x = np.array(self.corpuses[c][idx], dtype=np.int)
        p = np.random.rand(self.max_seq_len)
        r = np.random.randint(BGN_IDX, self.vocab_size, (self.max_seq_len, ))
        mask = p < .12
        y = mask * MASK_IDX + (p > .12) * (p < .2) * r + (p > .2) * x
        pos = np.arange(len(x))
        return x, y, mask * 1, len(x), pos


# def collate_fn(data):
#     xs, ys, masks, ls, poss = zip(*data)
#     # xs = pad_sequence([torch.LongTensor(x) for x in xs], batch_first=True, padding_value=PAD_IDX)
#     # ys = pad_sequence([torch.LongTensor(y) for y in ys], batch_first=True, padding_value=PAD_IDX)
#     # masks = pad_sequence([torch.LongTensor(mask * 1) for mask in masks], batch_first=True, padding_value=0)
#     return torch.LongTensor(xs), torch.LongTensor(ys), torch.LongTensor(masks), torch.LongTensor(ls), torch.LongTensor(poss)
