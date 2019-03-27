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

MAX_SEQ_LEN=256

def load_parallel_dataset(filename1, dico1, filename2, dico2):
    data = []
    unk_rate = 0
    for line1, line2 in tqdm(zip(open(filename1), open(filename2))):
        line1, line2 = convert(line1, dico1), convert(line2, dico2)
        unk_rate += line1.count(UNK_IDX) / len(line1) + line2.count(UNK_IDX) / len(line2)
        data.append((line1[:MAX_SEQ_LEN], line2[:MAX_SEQ_LEN]))
        if len(data) >= 100000:
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
        #xc, _ = self.lstm(self.embed(xc))
        #yc, _ = self.lstm_par(self.embed_par(yc))
        x, _ = torch.max(x, dim=1)
        y, _ = torch.max(y, dim=1)
        #xc, _ = torch.max(xc, dim=1)
        #yc, _ = torch.max(yc, dim=1)
        return self.l2(x, y) #- self.lbda * (self.l2(xc, y) + self.l2(x, yc))


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
        i, j = np.random.randint(len(self.data)), np.random.randint(len(self.data))
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

DEVICE='cuda:0'
MAX_EPOCH=100

def go_xnli(train, model, optimizer, criterion, generator, model_path=None):
    model.train(train)
    optimizer.zero_grad()
    pbar = tqdm(enumerate(generator), ncols=0)
    total_loss, total_acc = 0, 0
    for idx, (batch_X, batch_Y, batch_labels) in pbar:
        batch_X, batch_Y, batch_labels = batch_X.to(DEVICE), batch_Y.to(DEVICE), batch_labels.to(DEVICE)
        optimizer.zero_grad()
        pred_labels = model(batch_X, batch_Y)
        loss = criterion(pred_labels, batch_labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        total_acc += (pred_labels.argmax(dim=1) == batch_labels).float().mean().item()
        if idx % 10 == 9:
            pbar.set_description('[%s] loss: %.4f acc: %.4f' % ('Train' if train else 'EVAL', total_loss / idx, total_acc / idx))
    if train and model_path is not None:
        torch.save(model.state_dict(), model_path)
        print('model save to %s' % model_path)

generator_params = {'batch_size': 256, 'shuffle': True, 'num_workers': 12}

def train_nli():
    print('NLI')
    en_dico = pickle.load(open('data/dico/en', 'rb'))
    model = ClassifierModel(vocab_size=100000).float().to(DEVICE)
    if not os.path.exists('model/en-nli'):
        model.embed.load_state_dict({'weight': torch.as_tensor(np.load('data/weight/en.npy'))})
        print('load pretrained weight')
    if DEVICE != 'cpu':
        model = torch.nn.DataParallel(model, device_ids=[0])
    if os.path.exists('model/en-nli'):
        model.load_state_dict(torch.load('model/en-nli'))
        print('continuous training model')
    train_data = load_dataset('data/xnli/en.train', en_dico)
    valid_data = load_dataset('data/xnli/en.valid', en_dico)
    test_data = load_dataset('data/xnli/en.test', en_dico)
    train_generator = NLIDataset(train_data).get_generator(generator_params)
    valid_generator = NLIDataset(valid_data).get_generator(generator_params)
    test_generator = NLIDataset(test_data).get_generator(generator_params)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(MAX_EPOCH):
        print('Epoch: %d' % epoch)
        go_xnli(True, model, optimizer, criterion, train_generator, 'model/en-nli')
        go_xnli(False, model, optimizer, criterion, valid_generator, None)
        go_xnli(False, model, optimizer, criterion, test_generator, None)

def eval_nli():
    print('EVAL NLI')
    fr_dico = pickle.load(open('data/dico/fr', 'rb'))
    model = ClassifierModel(vocab_size=100000).float().to(DEVICE)
    nli_weight = torch.load('model/en-nli')
    par_weight = torch.load('model/par')
    lstm_weight, embed_weight = extract_weight(par_weight, 'lstm_par'), extract_weight(par_weight, 'embed_par')
    model.embed.load_state_dict(embed_weight)
    model.lstm.load_state_dict(lstm_weight)
    fc1_weight, fc2_weight = extract_weight(nli_weight, 'fc1'), extract_weight(nli_weight, 'fc2')
    model.fc1.load_state_dict(fc1_weight)
    model.fc2.load_state_dict(fc2_weight)
    valid_data = load_dataset('data/xnli/fr.valid', fr_dico)
    test_data = load_dataset('data/xnli/fr.test', fr_dico)
    valid_generator = NLIDataset(valid_data).get_generator(generator_params)
    test_generator = NLIDataset(test_data).get_generator(generator_params)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    go_xnli(False, model, optimizer, criterion, valid_generator, None)
    go_xnli(False, model, optimizer, criterion, test_generator, None)

def go_par(train, model, optimizer, generator, model_path=None, epoch_size=2000):
    model.train(train)
    optimizer.zero_grad()
    pbar = tqdm(enumerate(generator), ncols=0)
    total_loss = 0
    for idx, (batch_X, batch_Y, batch_Xc, batch_Yc) in pbar:
        batch_X, batch_Y, batch_Xc, batch_Yc = batch_X.to(DEVICE), batch_Y.to(DEVICE), batch_Xc.to(DEVICE), batch_Yc.to(DEVICE)
        optimizer.zero_grad()
        loss = model(batch_X, batch_Y, batch_Xc, batch_Yc).sum()
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        if idx % 10 == 9:
            pbar.set_description('[%s] loss: %.4f' % ('Train' if train else 'EVAL', total_loss / idx))
        if idx >= epoch_size:
            pbar.close()
            break
    if train and model_path is not None:
        torch.save(model.state_dict(), model_path)
        print('model save to %s' % model_path)

def extract_weight(weight, name):
    w = {}
    for key in weight:
        if key.startswith(name):
            w[key[len(name)+1:]] = weight[key]
        elif key.startswith('module.'+name):
            w[key[len('module.')+len(name)+1:]] = weight[key]
    return w

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

def train_par():
    print('PAR')
    en_dico = pickle.load(open('data/dico/en', 'rb'))
    fr_dico = pickle.load(open('data/dico/fr', 'rb'))
    model = MimicEncoderModel(vocab_size=100000, par_vocab_size=100000).float().to(DEVICE)
    if not os.path.exists('model/par'):
        weight = torch.load('model/en-nli')
        lstm_weight, embed_weight = extract_weight(weight, 'lstm'), extract_weight(weight, 'embed')
        model.embed.load_state_dict(embed_weight)
        model.lstm.load_state_dict(lstm_weight)
        model.embed_par.load_state_dict({'weight': torch.as_tensor(np.load('data/weight/fr.npy'))})
        print('load pretrained weight')        
    if DEVICE != 'cpu':
        model = torch.nn.DataParallel(model, device_ids=[0])
    if os.path.exists('model/par'):
        model.load_state_dict(torch.load('model/par'))
        print('continuous training model')
    train_data = load_parallel_dataset('data/para/en-fr.en.train', en_dico, 'data/para/en-fr.fr.train', fr_dico)
    valid_data = load_parallel_dataset('data/para/en-fr.en.valid', en_dico, 'data/para/en-fr.fr.valid', fr_dico)
    test_data = load_parallel_dataset('data/para/en-fr.en.test', en_dico, 'data/para/en-fr.fr.test', fr_dico)
    train_generator = ParallelDataset(train_data).get_generator(generator_params)
    valid_generator = ParallelDataset(valid_data).get_generator(generator_params)
    test_generator = ParallelDataset(test_data).get_generator(generator_params)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(MAX_EPOCH):
        print('Epoch: %d' % epoch)
        go_par(True, model, optimizer, train_generator, 'model/par')
        go_par(False, model, optimizer, valid_generator, None)
        go_par(False, model, optimizer, test_generator, None)

if __name__ == '__main__':
    train_nli()
    #train_par()
    #eval_nli()
