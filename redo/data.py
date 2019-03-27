import os
import numpy as np
import torch
import csv
import tqdm
import pickle
from tqdm import tqdm
import sys
from collections import defaultdict

EMBEDDING_PATH='redo/data/emb' #?
#XNLI_PATH='redo/data/xnli/XNLI-1.0'
LABEL_DICT = {'neutral': 0, 'entailment': 1, 'contradiction': 2}


def get_batch(batch, word_vec, emb_dim=300):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), emb_dim))
    ccount = 0
    count = 0

    for i in range(len(batch)):
        #print(batch[i])
        ccount += 1
        for j in range(len(batch[i])):
            count += 1
            if len(word_vec[batch[i][j]]) != 300:
                print (batch[i][j])
                print ("\n")
                print (word_vec[batch[i][j]])
                print ("\n")
                print (ccount, count)
                bij = np.random.rand(300)
                #continue
            #print (type(word_vec[batch[i][j]]), len(word_vec[batch[i][j]]))
            else:
                bij = [float(n) for n in word_vec[batch[i][j]]]
            embed[j, i, :] = bij #word_vec[batch[i][j]]
    #print (type(embed))
    return torch.from_numpy(embed).float(), lengths

def get_word_dict(pairs, lg="en"):
    word_dict = {}
    for pair in pairs:
        for sent in pair:
            for word in sent:#.split():
                if word not in word_dict:
                    word_dict[word] = ''
    return word_dict

def get_fastTEXT(word_dict, emb_path, lg="en"):
    matName = emb_path + "embed_dict_" + lg + ".pkl"
    lg_emb = emb_path + "cc." + lg + ".300.vec"

    if os.path.exists(matName):
        embedding_matrix = np.load(matName)
    else:
        embedding_matrix = word_dict
    word_count = 0
    with open(lg_emb) as f:
        for line in tqdm(f, total=2000001):
            line = line.strip().split()
            word = line[0]  # TODO: run this to see if there's formatting error
            emb_vec = line[-300:]
            if word in word_dict:
                word_count += 1
                embedding_matrix [word] = np.array(emb_vec)
                #print('Found {0}(/{1}) words with glove vectors'.format(
                #    word_count, len(word_dict)))

        for k in embedding_matrix:
            if embedding_matrix[k] is None:
                embedding_matrix[k] = np.random.rand(300)
    pickle.dump(embedding_matrix, open(matName, "wb"))
    return (embedding_matrix)


def build_vocab(data, emb_path, lg="en"):
    sents = []
    for f in data:
        sents.extend(f["pairs"])
    word_dict = get_word_dict(sents)
    word_vec = get_fastTEXT(word_dict, emb_path, lg)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec

def get_xnli(lg, XNLI_PATH): # train, test or valid
    train, test, valid = {}, {}, {}

    for mode in ["train", "test", "valid"]:
        train_lg_f = XNLI_PATH + r"/" + lg + "." + mode

        with open(train_lg_f) as f:
            content = f.readlines()
        labels = []
        sent_pairs = []
        content = [x.strip().split("\t") for x in content]
        for line in content:
            if line[-1] == "label":
                continue
            s1 = line[0]
            s2 = line[1]

            s1lst = [x for x in s1.lower().split(" ")]
            s2lst = [x for x in s2.lower().split(" ")]
            sent_pairs.append([s1lst, s2lst])
            labels.append(LABEL_DICT[line[-1]])

        eval(mode)["pairs"] = sent_pairs
        eval(mode)["labels"] = labels
        print (sent_pairs[0], labels[0])

    print (len(train["pairs"]), len(train["labels"]))
    return (train, test, valid)

