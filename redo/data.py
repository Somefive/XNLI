import os
import numpy as np
import torch
import csv
import tqdm
import pickle

EMBEDDING_PATH='redo/data/emb' #?
#XNLI_PATH='redo/data/xnli/XNLI-1.0'
LABEL_DICT = {'neutral': 0, 'entailment': 1, 'contradiction': 2}


def get_batch(batch, word_vec, emb_dim=300):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), emb_dim))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths




def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def load_embedding(emb_path, lg, word_dict):
    matName = "embed_dict_" + lg + ".pkl" # a dictionary
    #matName = "fastText_mat_" + lg + ".npy"
    lg_emb = emb_path + "cc." + lg + ".300.vec"

    if os.path.exists(matName):
        embedding_matrix = np.load(matName)
    else:
        embedding_matrix = word_dict
        with open(lg_emb) as f:
            for line in tqdm(f, total=2000001):
                line = line.strip().split()
                word = line[0] # TODO: run this to see if there's formatting error
                emb_vec = line[-300:]
                if word in embedding_matrix:
                    embedding_matrix[word_dict[word]] = np.array(emb_vec)
        for k in embedding_matrix:
            if embedding_matrix[k] is None:
                embedding_matrix[k] = np.random.rand(300)
    pickle.dump(embedding_matrix, open(matName, "wb"))
    return embedding_matrix


def build_embed(files, emb_path, lg):
    emb_mat = load_embedding(emb_path, lg)
    word2vec = {}
    word_dict = []
    for f in files:
        for pair in f:
            word_dict.extend(pair[0])
            word_dict.extend(pair[1])
    word_dict.fromkeys(word_dict)
    #word_dict['<s>'] = None
    #word_dict['</s>'] = None
    #word_dict['<p>'] = None
    return load_embedding(emb_path, lg, word_dict)

def get_xnli(lg, XNLI_PATH): # train, test or valid
    train, test, valid = {}, {}, {}
    w2i = defaultdict(lambda: len(w2i))
    UNK = w2i["<unk>"]
    #t2i = defaultdict(lambda: len(t2i))

    for mode in ["train", "test", "valid"]:
        train_lg_f = XNLI_PATH + r"/" + lg + "." + mode
        f_reader = csv.reader(open(train_lg_f), delimiter="\t")

        labels = []
        sent_pairs = []
        for line in f_reader:
            if len(line) != 3:
                print("Line data problem! Doesn't have 3 entries \n")
            labels.append(LABEL_DICT[line[-1]])
            s1 = line[0]
            s2 = line[1]
            s1lst = [x for x in s1.lower().split(" ")]
            s2lst = [x for x in s2.lower().split(" ")]
            sent_pairs.append([s1lst, s2lst])
        eval(mode)["pairs"] = sent_pairs
        eval(mode)["labels"] = labels
    return train, test, valid
