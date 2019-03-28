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
import pickle
import string
import argparse

def parse_args():
    def str2ints(x):
        return [int(i) for i in x.split(',')]
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_workers', default=12, type=int, help='batch size')
    parser.add_argument('--mode', choices=['nli','par','eval'], default='eval', help='running mode: [nli, par, eval]')
    parser.add_argument('--vocab_size', default=100000, type=int, help='batch size')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--gpus', default=[0], type=str2ints, help="gpu ids")
    parser.add_argument('--dataset_size', default=10000, type=int, help='dataset size')
    parser.add_argument('--max_seq_len', default=256, type=int, help='max sequence length')
    parser.add_argument('--max_epoch', default=100, type=int, help='max epoch')
    parser.add_argument('--epoch_size', default=2000, type=int, help='epoch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nli_model', default='model/en-nli', type=str, help='NLI model path')
    parser.add_argument('--par_model', default='model/par', type=str, help='PAR model path')
    args = parser.parse_args()
    return args

args = parse_args()
generator_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers}

def load_dicos(filenames):
    return [pickle.load(open(filename, 'rb')) for filename in filenames]

def load_fasttext(filename, dico, dim=300):
    weight = np.zeros((len(dico), dim))
    loaded = set()
    fp = open(filename, encoding='utf-8')
    fp.readline()
    for line in tqdm(fp):
        data = line.strip().split()
        token = ' '.join(data[:-dim])
        if token in dico:
            weight[dico[token]] = np.asarray([float(d) for d in data[-dim:]])
            loaded.add(token)
    fp.close()
    print('load %d pretrained weight' % len(loaded))
    return weight

LABEL_DICT = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
TABLE = str.maketrans(string.punctuation+string.digits, ' '*len(string.punctuation+string.digits))

def convert(line, dico):
    return [BOS_IDX] + [dico[word] if word in dico else UNK_IDX for word in line.strip().lower().translate(TABLE).split()] + [EOS_IDX]

def load_dataset(filename, dico):
    fp = open(filename, encoding='utf-8')
    fp.readline()
    data = []
    unk_rate = 0
    for line in tqdm(fp):
        s1, s2, label = line.strip().split('\t')
        s1, s2, label = convert(s1, dico), convert(s2, dico), LABEL_DICT[label]
        unk_rate += s1.count(UNK_IDX) / len(s1) + s2.count(UNK_IDX) / len(s2)
        data.append((s1, s2, label))
    fp.close()
    print('load %d data from %s. <unk> rate is %.3f.' % (len(data), filename, unk_rate / len(data) / 2))
    return data

def load_parallel_dataset(filename1, dico1, filename2, dico2):
    data = []
    unk_rate = 0
    for line1, line2 in tqdm(zip(open(filename1), open(filename2))):
        line1, line2 = convert(line1, dico1), convert(line2, dico2)
        unk_rate += line1.count(UNK_IDX) / len(line1) + line2.count(UNK_IDX) / len(line2)
        data.append((line1[:args.max_seq_len], line2[:args.max_seq_len]))
        if len(data) >= args.dataset_size:
            break
    print('load %d data from %s,%s. <unk> rate is %.3f.' % (len(data), filename1, filename2, unk_rate / len(data) / 2))
    return data

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

def get_vocab_and_weight(filename, weight_filename, size=80000):
    with open(weight_filename) as f:
        weights = {}
        n, d = map(int, f.readline().rstrip().split())
        for line in tqdm(f):
            tokens = line.rstrip().split()
            w = np.array(list(map(float, tokens[-d:])))
            token = ' '.join(tokens[:-d])
            weights[token] = w
        print('%d weights, dim=%d.' % (n, d))
    with open(filename) as f:
        counter = Counter()
        for line in tqdm(f):
            counter.update(line.lower().translate(TABLE).rstrip().split())
        dico = {'<pad>': PAD_IDX, '<s>': BOS_IDX, '</s>': EOS_IDX, '<unk>': UNK_IDX}
        unks = []
        for word, cnt in counter.most_common():
            if word in weights:
                dico[word] = len(dico)
            else:
                unks.append((word, cnt))
        print('%d word loaded. %d unks.' % (len(dico), len(unks)))
    return weights, dico, unks

def resolve_dico_and_weights(weights, dico, size=200000, dim=300):
    w = np.zeros((size, dim))
    _dico = {}
    for token, idx in dico.items():
        if token in weights and idx < size:
            w[idx] = weights[token]
        if idx < size:
            _dico[token] = idx
    return w, _dico

def preprocess(size=200000):
    weights, dico, unks = get_vocab_and_weight('data/para/en-fr.en.all', 'data/embed/en')
    w, _dico = resolve_dico_and_weights(weights, dico, size)
    np.save('data/weight/en', w)
    pickle.dump(_dico, open('data/dico/en', 'wb'))
    weights, dico, unks = get_vocab_and_weight('data/para/en-fr.fr.all', 'data/embed/fr')
    w, _dico = resolve_dico_and_weights(weights, dico, size)
    np.save('data/weight/fr', w)
    pickle.dump(_dico, open('data/dico/fr', 'wb'))

def extract_weight(weight, name):
    w = {}
    for key in weight:
        if key.startswith(name+'.'):
            w[key[len(name)+1:]] = weight[key]
        elif key.startswith('module.'+name+'.'):
            w[key[len('module.')+len(name)+1:]] = weight[key]
    return w

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

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
        self.relu = nn.ReLU()
        freeze_layer(self.embed)

    def forward(self, x, y):
        x, _ = self.lstm(self.embed(x)) # batch_size, seq_len, 2*lstm_hidden_size
        y, _ = self.lstm(self.embed(y))
        (x, _), (y, _) = torch.max(x, dim=1), torch.max(y, dim=1)
        z = torch.cat([x, y, torch.abs(x-y), x*y], dim=1)
        return self.fc2(self.dropout(self.relu(self.fc1(z))))


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
        freeze_layer(self.embed)
        freeze_layer(self.lstm)
        freeze_layer(self.embed_par)
        self.l2 = nn.MSELoss()

    def forward(self, x, y, xc, yc):
        x, _ = self.lstm(self.embed(x)) # batch_size, seq_len, 2*lstm_hidden_size
        y, _ = self.lstm_par(self.embed_par(y))
        xc, _ = self.lstm(self.embed(xc))
        yc, _ = self.lstm_par(self.embed_par(yc))
        x, _ = torch.max(x, dim=1)
        y, _ = torch.max(y, dim=1)
        xc, _ = torch.max(xc, dim=1)
        yc, _ = torch.max(yc, dim=1)
        return self.l2(x, y) - self.lbda * (self.l2(xc, y) + self.l2(x, yc))


class NLIDataset(torch.utils.data.Dataset):
    
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def get_generator(self, params={}):
        params['collate_fn'] = collate_fn
        return torch.utils.data.DataLoader(self, **params)

def collate_fn(data):
    xs, ys, labels = zip(*data)
    xs = pad_sequence([torch.LongTensor(x) for x in xs], batch_first=True, padding_value=PAD_IDX)
    ys = pad_sequence([torch.LongTensor(y) for y in ys], batch_first=True, padding_value=PAD_IDX)
    return xs, ys, torch.LongTensor(labels)

def collate_fn_par(data):
    xs, ys, xcs, ycs = zip(*data)
    xs = pad_sequence([torch.LongTensor(x) for x in xs], batch_first=True, padding_value=PAD_IDX)
    ys = pad_sequence([torch.LongTensor(y) for y in ys], batch_first=True, padding_value=PAD_IDX)
    xcs = pad_sequence([torch.LongTensor(x) for x in xcs], batch_first=True, padding_value=PAD_IDX)
    ycs = pad_sequence([torch.LongTensor(y) for y in ycs], batch_first=True, padding_value=PAD_IDX)
    return xs, ys, xcs, ycs


class ParallelDataset(torch.utils.data.Dataset):
    
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        i, j = index, index
        while i == index:
            i = np.random.randint(len(self.data))
        while j == index:
            j = np.random.randint(len(self.data))
        xc, yc = self.data[i][0], self.data[j][1]
        return x, y, xc, yc
    
    def get_generator(self, params={}):
        params['collate_fn'] = collate_fn_par
        return torch.utils.data.DataLoader(self, **params)

def collate_fn_par(data):
    xs, ys, xcs, ycs = zip(*data)
    xs = pad_sequence([torch.LongTensor(x) for x in xs], batch_first=True, padding_value=PAD_IDX)
    ys = pad_sequence([torch.LongTensor(y) for y in ys], batch_first=True, padding_value=PAD_IDX)
    xcs = pad_sequence([torch.LongTensor(x) for x in xcs], batch_first=True, padding_value=PAD_IDX)
    ycs = pad_sequence([torch.LongTensor(y) for y in ycs], batch_first=True, padding_value=PAD_IDX)
    return xs, ys, xcs, ycs

def go_nli(train, model, generator):
    model.train(train)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(enumerate(generator), ncols=0)
    total_loss, total_acc, cnt = 0, 0, 0
    for idx, (batch_X, batch_Y, batch_labels) in pbar:
        batch_X, batch_Y, batch_labels = batch_X.to(args.device), batch_Y.to(args.device), batch_labels.to(args.device)
        optimizer.zero_grad()
        pred_labels = model(batch_X, batch_Y)
        loss = criterion(pred_labels, batch_labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch_labels.size(0)
        total_acc += (pred_labels.argmax(dim=1) == batch_labels).float().sum().item()
        cnt += batch_labels.size(0)
        if idx % 10 == 9:
            pbar.set_description('[%s] loss: %.4f acc: %.4f' % ('Train' if train else 'EVAL', total_loss / cnt, total_acc / cnt))
    if train:
        torch.save(model.state_dict(), args.nli_model)
        print('model save to %s' % args.nli_model)

def train_nli():
    en_dico = pickle.load(open('data/dico/en', 'rb'))
    model = ClassifierModel(vocab_size=args.vocab_size).float().to(args.device)
    if not os.path.exists(args.nli_model):
        model.embed.load_state_dict({'weight': torch.as_tensor(np.load('data/weight/en.npy'))})
        print('load pretrained weight')
    if args.device != 'cpu':
        model = torch.nn.DataParallel(model, device_ids=args.gpus)
    if os.path.exists(args.nli_model):
        model.load_state_dict(torch.load(args.nli_model, map_location=args.device))
        print('continuous training model')
    train_data = load_dataset('data/xnli/en.train', en_dico)
    valid_data = load_dataset('data/xnli/en.valid', en_dico)
    test_data = load_dataset('data/xnli/en.test', en_dico)
    train_generator = NLIDataset(train_data).get_generator(generator_params)
    valid_generator = NLIDataset(valid_data).get_generator(generator_params)
    test_generator = NLIDataset(test_data).get_generator(generator_params)
    for epoch in range(args.max_epoch):
        print('Epoch: %d' % epoch)
        go_nli(True, model, train_generator)
        go_nli(False, model, valid_generator)
        go_nli(False, model, test_generator)

def eval_nli():
    print('EVAL NLI')
    fr_dico = pickle.load(open('data/dico/fr', 'rb'))
    model = ClassifierModel(vocab_size=args.vocab_size).float().to(args.device)
    nli_weight = torch.load(args.nli_model, map_location=args.device)
    par_weight = torch.load(args.par_model, map_location=args.device)
    lstm_weight, embed_weight = extract_weight(par_weight, 'lstm_par'), extract_weight(par_weight, 'par_embed')
    model.embed.load_state_dict(embed_weight)
    model.lstm.load_state_dict(lstm_weight)
    fc1_weight, fc2_weight = extract_weight(nli_weight, 'fc1'), extract_weight(nli_weight, 'fc2')
    model.fc1.load_state_dict(fc1_weight)
    model.fc2.load_state_dict(fc2_weight)
    valid_data = load_dataset('data/xnli/fr.valid', fr_dico)
    test_data = load_dataset('data/xnli/fr.test', fr_dico)
    valid_generator = NLIDataset(valid_data).get_generator(generator_params)
    test_generator = NLIDataset(test_data).get_generator(generator_params)
    go_nli(False, model, valid_generator)
    go_nli(False, model, test_generator)

def go_par(train, model, generator):
    model.train(train)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    pbar = tqdm(enumerate(generator), ncols=0)
    total_loss, cnt = 0, 0
    for idx, (batch_X, batch_Y, batch_Xc, batch_Yc) in pbar:
        batch_X, batch_Y, batch_Xc, batch_Yc = batch_X.to(args.device), batch_Y.to(args.device), batch_Xc.to(args.device), batch_Yc.to(args.device)
        optimizer.zero_grad()
        loss = model(batch_X, batch_Y, batch_Xc, batch_Yc).sum()
        if train:
           loss.backward()
           optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
        cnt += batch_X.size(0)
        if idx % 10 == 9:
            pbar.set_description('[%s] loss: %.4e' % ('Train' if train else 'EVAL', total_loss / cnt))
        if idx >= args.epoch_size:
            pbar.close()
            break
    if train:
        torch.save(model.state_dict(), args.par_model)
        print('model save to %s' % args.par_model)

def train_par():
    en_dico = pickle.load(open('data/dico/en', 'rb'))
    fr_dico = pickle.load(open('data/dico/fr', 'rb'))
    model = MimicEncoderModel(vocab_size=args.vocab_size, par_vocab_size=args.vocab_size).float().to(args.device)
    if not os.path.exists(args.par_model):
        weight = torch.load(args.nli_model, map_location=args.device)
        lstm_weight, embed_weight = extract_weight(weight, 'lstm'), extract_weight(weight, 'embed')
        model.embed.load_state_dict(embed_weight)
        model.lstm.load_state_dict(lstm_weight)
        model.embed_par.load_state_dict({'weight': torch.as_tensor(np.load('data/weight/fr.npy'))})
        print('load pretrained weight')        
    if args.device != 'cpu':
        model = torch.nn.DataParallel(model, device_ids=args.gpus)
    if os.path.exists(args.par_model):
        model.load_state_dict(torch.load(args.par_model, map_location=args.device))
        print('continuous training model')
    train_data = load_parallel_dataset('data/para/en-fr.en.train', en_dico, 'data/para/en-fr.fr.train', fr_dico)
    valid_data = load_parallel_dataset('data/para/en-fr.en.valid', en_dico, 'data/para/en-fr.fr.valid', fr_dico)
    test_data = load_parallel_dataset('data/para/en-fr.en.test', en_dico, 'data/para/en-fr.fr.test', fr_dico)
    train_generator = ParallelDataset(train_data).get_generator(generator_params)
    valid_generator = ParallelDataset(valid_data).get_generator(generator_params)
    test_generator = ParallelDataset(test_data).get_generator(generator_params)
    
    for epoch in range(args.max_epoch):
        print('Epoch: %d' % epoch)
        go_par(True, model, train_generator)
        go_par(False, model, valid_generator)
        go_par(False, model, test_generator)

if __name__ == '__main__':
    print('model: %s' % args.mode)
    if args.mode == 'nli':
        train_nli()
    elif args.mode == 'par':
        train_par()
    elif args.mode == 'eval':
        eval_nli()
