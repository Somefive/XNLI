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

# max_seq_len=256, maxlines=5000000
def load_data(filenames, dico, max_seq_len=64, maxlines=100000):
    data, length = [], []
    for filename in filenames:
        for _, line in zip(range(maxlines), open(filename)):
            entry = [POS_IDX]
            entry.extend([dico[token] if token in dico else UNK_IDX for token in line.strip().split(' ')])
            entry.append(POS_IDX)
            entry = entry[:max_seq_len]
            l = len(entry)
            data.append(entry)
            length.append(l)
    return data, length


class MTMDataset(torch.utils.data.Dataset):

    def __init__(self, vocab_filenames, text_filenames, vocab_size=20000, max_seq_len=64, maxlines_per_file=100000):
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len
        self.dico, self.counter = load_codes(vocab_filenames, vocab_size)
        self.data, self.length = load_data(text_filenames, self.dico, max_seq_len=max_seq_len, maxlines=maxlines_per_file)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, l = np.array(self.data[index], dtype=np.int), self.length[index]
        p = np.random.rand(l)
        r = np.random.randint(BGN_IDX, self.vocab_size, (l, ))
        mask = p < .12
        y = mask * MASK_IDX + (p > .12) * (p < .2) * r + (p > .2) * x
        return x, y, mask, l


def collate_fn(data):
    xs, ys, masks, ls = zip(*data)
    xs = pad_sequence([torch.LongTensor(x) for x in xs], batch_first=True, padding_value=PAD_IDX)
    ys = pad_sequence([torch.LongTensor(y) for y in ys], batch_first=True, padding_value=PAD_IDX)
    masks = pad_sequence([torch.LongTensor(mask * 1) for mask in masks], batch_first=True, padding_value=0)
    return xs, ys, masks, torch.LongTensor(ls)
