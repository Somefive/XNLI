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

def composed_dataloader(dataloader1, dataloader2):
    while True:
        for data1, data2 in zip(dataloader1, dataloader2):
            yield data1
            yield data2



class MaskedDataset(torch.utils.data.Dataset):

    def __init__(self, dico, filenames, maxlines=1e8, max_seq_len=256):
        self.data = []
        for filename in filenames:
            self.data += load_LM_data(filename, dico, maxlines)
        self.size, self.vocab_size = len(self.data), len(dico)
        self.max_seq_len = max_seq_len
        unk_rate = sum([seq.count(UNK_IDX) / len(seq) for seqs, _ in self.data for seq in seqs]) / self.size * 100
        avg_seq_len = sum([len(seq) for seqs, _ in self.data for seq in seqs]) / self.size
        print('[MaskedDataset] load %s data. %.2f%% <UNK>. Avg Length: %.2f.' % (self.size, unk_rate, avg_seq_len))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        seqs, langs = self.data[index]
        if len(seqs) == 2:
            if np.random.rand() < 0.5:
                seq1, seq2 = seqs
                lang1, lang2 = langs
            else:
                seq2, seq1 = seqs
                lang2, lang1 = langs
            seq = seq1 + seq2
            langs = [lang1] * len(seq1) + [lang2] * len(seq2)
            pos = list(range(len(seq1))) + list(range(len(seq2)))
        else:
            seq = seqs[0]
            langs = [langs[0]] * len(seq)
            pos = list(range(len(seq)))
        label, langs, pos = seq[:self.max_seq_len], langs[:self.max_seq_len], pos[:self.max_seq_len]
        p = np.random.rand(len(label))
        r = np.random.randint(BGN_IDX, self.vocab_size, (len(label),))
        data = (p < .12) * MASK_IDX + (p > .12) * (p < .135) * r + (p > .15) * label
        return data, label, len(seq), pos, langs, (p < .15)

    def get_generator(self, params={}):
        params['collate_fn'] = collate_fn_masked
        return torch.utils.data.DataLoader(self, **params)

def collate_fn_masked(data):
    xs, ys, ls, poss, langs, masks = zip(*data)
    xs = pad_sequence([torch.LongTensor(x) for x in xs], batch_first=True, padding_value=PAD_IDX)
    ys = pad_sequence([torch.LongTensor(y) for y in ys], batch_first=True, padding_value=PAD_IDX)
    ls = torch.LongTensor(ls)
    poss = pad_sequence([torch.LongTensor(pos) for pos in poss], batch_first=True, padding_value=0)
    langs = pad_sequence([torch.LongTensor(lang) for lang in langs], batch_first=True, padding_value=0)
    masks = pad_sequence([torch.LongTensor(mask * 1) for mask in masks], batch_first=True, padding_value=0)
    return xs, ys, ls, poss, langs, masks


class XNLIDataset(torch.utils.data.Dataset):

    def __init__(self, dico, lang, type_, maxlines=1e8, max_seq_len=256):
        s1_path, s2_path, label_path = ['data/xnli/%s.%s.%s' % (type_, t, lang) for t in ['s1', 's2', 'label']]
        self.data = load_LM_data([s1_path, s2_path], dico, maxlines, True)
        self.labels = [CLASS2ID[line.strip()] for line in open(label_path)]
        self.size, self.vocab_size = len(self.data), len(dico)
        self.max_seq_len = max_seq_len
        unk_rate = sum([seq.count(UNK_IDX) / len(seq) for seqs, _ in self.data for seq in seqs]) / self.size * 100
        avg_seq_len = sum([len(seq) for seqs, _ in self.data for seq in seqs]) / self.size
        print('[MaskedDataset] load %s data. %.2f%% <UNK>. Avg Length: %.2f.' % (self.size, unk_rate, avg_seq_len))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        (seqs, langs), label = self.data[index], self.labels[index]
        if np.random.rand() < 0.5:
            seq1, seq2 = seqs
            lang1, lang2 = langs
        else:
            seq2, seq1 = seqs
            lang2, lang1 = langs
        seq = seq1 + seq2
        langs = [lang1] * len(seq1) + [lang2] * len(seq2)
        pos = list(range(len(seq1))) + list(range(len(seq2)))
        data, langs, pos = seq[:self.max_seq_len], langs[:self.max_seq_len], pos[:self.max_seq_len]
        return data, label, len(seq), pos, langs

    def get_generator(self, params={}):
        params['collate_fn'] = collate_fn_xnli
        return torch.utils.data.DataLoader(self, **params)


def collate_fn_xnli(data):
    xs, ys, ls, poss, langs = zip(*data)
    xs = pad_sequence([torch.LongTensor(x) for x in xs], batch_first=True, padding_value=PAD_IDX)
    ys = torch.LongTensor(ys)
    ls = torch.LongTensor(ls)
    poss = pad_sequence([torch.LongTensor(pos) for pos in poss], batch_first=True, padding_value=0)
    langs = pad_sequence([torch.LongTensor(lang) for lang in langs], batch_first=True, padding_value=0)
    return xs, ys, ls, poss, langs
