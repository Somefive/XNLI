import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
import os
import torch.utils.data
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

def load_fasttext(filename, dico, dim=300):
    weight = np.zeros((len(dico), dim))
    loaded = set()
    fp = open(filename, encoding='utf-8')
    fp.readline()
    for line in tqdm(fp):
        data = line.strip().split()
        if data[0] in dico:
            weight[dico[data[0]]] = np.asarray([float(d) for d in data[1:]])
            loaded.add(data[0])
    fp.close()
    print('load %d pretrained weight' % len(loaded))
    return weight

def load_data(filename):
    return [line.lower().split(' ') for line in open(filename)]

LABEL_DICT = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
def load_label(filename):
    return [LABEL_DICT[line.strip()] for line in open(filename)]

def convert(lines, dico):
    return [[dico[word] if word in dico else UNK_IDX for word in line] for line in lines]

def load_dataset(filename, dico):
    s1, s2 = load_data(filename % 's1'), load_data(filename % 's2')
    return convert(s1, dico), convert(s2, dico), load_label(filename % 'label')

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

def get_vocab(filename, size=40000):
    counter = Counter()
    for line in tqdm(open(filename, encoding='utf-8')):
        counter.update(line.lower().strip().split())
    dico = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3}
    for token, _ in counter.most_common(size-len(dico)):
        dico[token] = len(dico)
    print('%d unique word found. Generate vocab with %d words.' % (len(counter), len(dico)))
    return dico


class ClassifierModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=300, lstm_hidden_dim=512, lstm_n_layer=1, fc_hidden_dim=128):
        super(ClassifierModel, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, 
                             hidden_size=lstm_hidden_dim, 
                             num_layers=lstm_n_layer,
                             bias=True,
                             batch_first=True,
                             dropout=0.1,
                             bidirectional=True)
        self.fc1 = nn.Linear(in_features=8*lstm_hidden_dim, out_features=fc_hidden_dim, bias=True)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(in_features=fc_hidden_dim, out_features=3, bias=True)

    def forward(self, x, y):
        x, _ = self.lstm(self.embed(x)) # batch_size, seq_len, 2*lstm_hidden_size
        y, _ = self.lstm(self.embed(y))
        (x, _), (y, _) = torch.max(x, dim=1), torch.max(y, dim=1)
        z = torch.cat([x, y, torch.abs(x-y), x*y])
        return self.fc2(self.dropout(self.fc1(z)))


class MimicEncoderModel(nn.Module):

    def __init__(self, vocab_size, par_vocab_size, embed_dim=300, lstm_hidden_dim=512, lstm_n_layer=1, fc_hidden_dim=128, lbda=0.25):
        super(MimicEncoderModel, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.embed_par = nn.Embedding(num_embeddings=par_vocab_size, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, 
                             hidden_size=lstm_hidden_dim, 
                             num_layers=lstm_n_layer,
                             bias=True,
                             batch_first=True,
                             dropout=0.1,
                             bidirectional=True)
        self.lstm_par = nn.LSTM(input_size=embed_dim, 
                             hidden_size=lstm_hidden_dim, 
                             num_layers=lstm_n_layer,
                             bias=True,
                             batch_first=True,
                             dropout=0.1,
                             bidirectional=True)
        self.lbda = lbda
        self.embed.require_grad = False
        self.lstm.require_grad = False
        self.l2 = nn.MSELoss()

    def forward(self, x, y, xc, yc):
        x, _ = self.lstm(self.embed(x)) # batch_size, seq_len, 2*lstm_hidden_size
        y, _ = self.lstm_par(self.embed_par(y))
        xc, _ = self.lstm(self.embed(xc))
        yc, _ = self.lstm_par(self.embed_par(yc))
        return self.l2(x, y) - self.lbda * (self.l2(xc, y) + self.l2(x, yc))


if __name__ == '__main__':
    pass