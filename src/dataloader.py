from collections import Counter
import torch
import torch.utils.data
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from utils import extract_lang

PAD_IDX = 0
UNK_IDX = 1
MASK_IDX = 2
POS_IDX = 3
BGN_IDX = 4

LANG_DICT = {lang: idx for idx, lang in enumerate('ar bg de el en es fr hi ru sw th tr ur vi zh'.split(' '))}
CLASS2ID = {'neutral': 0, 'entailment': 1, 'contradiction': 2}

def load_vocab(filenames, size=20000):
    counter = Counter()
    for filename in filenames:
        for line in open(filename):
            if len(line.strip().split(' ')) == 1:
                continue
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
    print('Vocab size is %d' % len(dico))
    return dico, counter

def line2id(line, dico):
    return [dico[token] if token in dico else UNK_IDX for token in line.lower().strip().split(' ')]

def load_monolingual_data(filename, dico, max_seq_len=64, maxlines=100000):
    data, n = [POS_IDX], 0
    for _, line in tqdm(zip(range(maxlines), open(filename)), leave=False):
        data += line2id(line, dico)
        data += [POS_IDX]
        n += 1
    size = len(data) // max_seq_len * max_seq_len
    data = np.array(data[:size]).reshape(-1, max_seq_len)
    print('load monolingual data from %s' % filename)
    return data, data.shape[0]

def load_parallel_data(filename1, filename2, dico, max_seq_len=64, maxlines=100000):
    data, n, reset_pos = [], 0, []
    for _, line1, line2 in tqdm(zip(range(maxlines), open(filename1), open(filename2)), leave=False):
        entry = [POS_IDX]
        entry += line2id(line1, dico)
        entry += [POS_IDX, POS_IDX]
        rp = len(entry) - 1
        entry += line2id(line2, dico)
        entry += [POS_IDX]
        data.append(entry[:max_seq_len])
        reset_pos.append(min(rp, max_seq_len))
        n += 1
    print('load parallel data from %s & %s' % (filename1, filename2))
    return data, n, reset_pos

def load_xnli_data(filename1, filename2, filename3, dico, max_seq_len=64, maxlines=100000):
    data, n, reset_pos = load_parallel_data(filename1, filename2, dico, max_seq_len=max_seq_len, maxlines=maxlines)
    label2id = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
    labels = [label2id[line.strip()] for _, line in zip(range(maxlines), open(filename3))]
    n = min([n, len(labels)])
    print('load xnli data label from %s' % filename3)
    return data[:n], n, reset_pos[:n], labels[:n]





def _extract(line, dico):
    return [POS_IDX] + line2id(line, dico) + [POS_IDX]

def load_LM_data(filename, dico, maxlines=100000, lang_end=False):
    data = []
    if len(filename) == 2:
        filename1, filename2 = filename
        lang1, lang2 = LANG_DICT[extract_lang(filename1, lang_end)], LANG_DICT[extract_lang(filename2, lang_end)]
        for _, line1, line2 in tqdm(zip(range(maxlines), open(filename1), open(filename2)), leave=False):
            seq1, seq2 = _extract(line1, dico), _extract(line2, dico)
            data.append(((seq1, seq2), (lang1, lang2)))
        print('load %d parallel data from %s & %s' % (len(data), filename1, filename2))
    else:
        lang = LANG_DICT[extract_lang(filename)]
        for _, line in tqdm(zip(range(maxlines), open(filename)), leave=False):
            seq = _extract(line, dico)
            data.append(((seq, ), (lang, )))
        print('load %d single data from %s' % (len(data), filename))
    return data


