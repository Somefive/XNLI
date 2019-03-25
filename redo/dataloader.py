from collections import Counter
from collections import defaultdict
import numpy as np
import csv
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch


EMBEDDING_PATH='redo/data/emb' #?
XNLI_PATH='redo/data/xnli/XNLI-1.0'

LABEL_DICT = {'neutral': 0, 'entailment': 1, 'contradiction': 2}

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
UNK = w2i["<unk>"]


def load_lg_xnli_train(lg): # en-train:redo/data/xnli/XNLI-1.0/en.train ?
    train_lg_f = XNLI_PATH + r"/" + lg + ".train"
    f_reader = csv.reader(open(train_lg_f), delimiter="\t")

    labels = []
    sent_pairs = []
    for line in f_reader:
        if len(line) != 3:
            print("Line data problem! Doesn't have 3 entries \n")
        labels.append(LABEL_DICT[line[-1]])
        s1 = line[0]
        s2 = line[1]
        vec1 = [w2i[x] for x in s1.lower().split(" ")]
        vec2 = [w2i[x] for x in s2.lower().split(" ")]
        sent_pairs.append([vec1, vec2])
    return sent_pairs, labels


VOCAB_SIZE = len(w2i)


def load_embedding_lg(lg):
    matName = "fastText_mat_" + lg + ".npy"
    lg_emb = EMBEDDING_PATH + "cc." + lg + ".300.vec"

    if os.path.exists(matName):
        embedding_matrix = np.load(matName)
    else:
        embedding_matrix = np.random.randn(len(w2i), 300)
        with open(lg_emb) as f:
            for line in tqdm(f, total=2000001):
                line = line.strip().split()
                word = line[0] # TODO: run this to see if there's formatting error
                emb_vec = line[-300:]
                if word in w2i:
                    embedding_matrix[w2i[word],] = np.array(emb_vec)
    np.save(matName, embedding_matrix)
    #return ()


train_pairs, train_labels = load_lg_xnli_train(lg)


class FastTextDataset(Dataset):
    """ Pretrained FastText word embedding """
    def __init__(self, lg):
        pairs, labels = load_lg_xnli_train(lg)
        premise = [l[0] for l in pairs]
        hypothesis = [l[1] for l in pairs]
        tags = [labels[l] for l in labels]
        d = {"premise":premise, "hypothesis":hypothesis, "tag":tags}
        self.data = pd.DataFrame(data = d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise = self.data.iloc[idx,0]
        hypothesis = self.data.iloc[idx, 1]
        tag = self.data.iloc[idx, 2]
        sample = {'premise': torch.LongTensor(premise),
                  'hypothesis': torch.LongTensor(hypothesis)
                  'tag': torch.LongTensor([int(tag)])} # TODO:torch.LongTensor(vec) may throw error
        return sample

def collate_fn(data):
    xs, ys, labels = zip(*data)
    xs = pad_sequence([torch.LongTensor(x) for x in xs], batch_first=True, padding_value=PAD_IDX)
    ys = pad_sequence([torch.LongTensor(y) for y in ys], batch_first=True, padding_value=PAD_IDX)
    return xs, ys, torch.LongTensor(labels)



train_dataset = FastTextDataset("en")





