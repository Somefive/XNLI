import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bpemb import BPEmb
from tqdm import tqdm
from collections import Counter
import os
import torch.utils.data
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

VOCAB_SIZE=25000
EMBED_DIM=200
HIDDEN_DIM=256
MODEL_PATH='model/bilstm'

DEVICE='cpu'
MAX_EPOCH = 50

PAD_IDX=0
UNK_IDX=1
POS_IDX=2
BGN_IDX=3

# bpemb = BPEmb(lang='en', vs=VOCAB_SIZE, dim=EMBED_DIM)

class BiLSTMModel(nn.Module):

    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=256):
        super(BiLSTMModel, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, 
                             hidden_size=hidden_dim, 
                             num_layers=2,
                             bias=True,
                             batch_first=True,
                             dropout=0.1,
                             bidirectional=True)
        self.fc = nn.Linear(in_features=4*hidden_dim, out_features=3, bias=True)

    def forward(self, x, y):
        x, _ = self.lstm(self.embed(x)) # batch_size, seq_len, 2*hidden_size 
        y, _ = self.lstm(self.embed(y))
        x_attn = F.softmax(torch.matmul(x, x.transpose(1,2)), dim=-1)
        y_attn = F.softmax(torch.matmul(y, y.transpose(1,2)), dim=-1)
        x, y = torch.matmul(x_attn, x), torch.matmul(y_attn, y) # batch_size, seq_len, 2*hidden_size
        z, _ = torch.max(torch.cat([x, y], dim=-1), dim=1) # batch_size, 4*hidden_size
        return self.fc(z)

train_file = 'data/xnli/train.%s.en'
valid_file = 'data/xnli/valid.%s.en'
test_file = 'data/xnli/test.%s.en'

def load_data(filename):
    return [line.lower().split(' ') for line in open(filename)]

LABEL_DICT = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
def load_label(filename):
    return [LABEL_DICT[line.strip()] for line in open(filename)]

def convert(lines, dico):
    return [[dico[word] if word in dico else UNK_IDX for word in line] for line in lines]

def extract_vocab():
    s1, s2 = load_data(train_file % 's1'), load_data(train_file % 's2')
    counter = Counter()
    for line in s1:
        counter.update(line)
    for line in s2:
        counter.update(line)
    dico = {'<unk>': UNK_IDX, '<pad>': PAD_IDX, '<s>': POS_IDX}
    print('Total BPE: %d' % len(counter))
    for word, cnt in counter.most_common(VOCAB_SIZE-BGN_IDX):
        dico[word] = len(dico)
    s1, s2, labels = convert(s1, dico), convert(s2, dico), load_label(train_file % 'label')
    return dico, s1, s2, labels
    
def load_dataset(filename, dico):
    s1, s2 = load_data(filename % 's1'), load_data(filename % 's2')
    return convert(s1, dico), convert(s2, dico), load_label(filename % 'label')

# def load_pretrained(dico):
#     weight = np.zeros((VOCAB_SIZE, EMBED_DIM))
#     miss = 0
#     for token, idx in dico.items():
#         if token[-2:] == '@@':
#             token = '▁' + token[:-2]
#         if token in bpemb.emb:
#             weight[idx] = bpemb.emb[token]
#         else:
#             print(token)
#             weight[idx] = bpemb.emb['<unk>']
#             miss += 1
#     print('pretrain miss rate: %.2f' % (miss / VOCAB_SIZE))
#     return weight


class XNLIDataset(torch.utils.data.Dataset):
    def __init__(self, s1, s2, labels):
        self.data = list(zip(s1, s2, labels))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y, label = self.data[index]
        l1, l2 = len(x), len(y)
        l = max([l1, l2])
        x += [PAD_IDX] * (l-l1)
        y += [PAD_IDX] * (l-l2)
        return x, y, label
    
    def get_generator(self, params={}):
        params['collate_fn'] = collate_fn
        return torch.utils.data.DataLoader(self, **params)

def collate_fn(data):
    xs, ys, labels = zip(*data)
    xs = pad_sequence([torch.LongTensor(x) for x in xs], batch_first=True, padding_value=PAD_IDX)
    ys = pad_sequence([torch.LongTensor(y) for y in ys], batch_first=True, padding_value=PAD_IDX)
    return xs, ys, torch.LongTensor(labels)

if __name__ == '__main__':
    dico, train_s1, train_s2, train_labels = extract_vocab()
    valid_s1, valid_s2, valid_labels = load_dataset(valid_file, dico)
    # test_s1, test_s2, test_labels = load_dataset(test_file, dico)
    model = BiLSTMModel(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    model.float().to(DEVICE)
    # weight = load_pretrained(dico)
    # model.embed.from_pretrained(torch.as_tensor(weight), freeze=False)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    print('prepared')
    
    generator_params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 6
    }

    train_dataset = XNLIDataset(train_s1, train_s2, train_labels).get_generator(generator_params)
    valid_dataset = XNLIDataset(valid_s1, valid_s2, valid_labels).get_generator(generator_params)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(MAX_EPOCH):
        model.train()
        pbar = tqdm(enumerate(train_dataset), ncols=0)
        total_loss, total_acc = 0, 0
        for idx, (batch_X, batch_Y, batch_labels) in pbar:
            batch_X, batch_Y, batch_labels = batch_X.to(DEVICE), batch_Y.to(DEVICE), batch_labels.to(DEVICE)
            optimizer.zero_grad()
            pred_labels = model(batch_X, batch_Y)
            loss = criterion(pred_labels, batch_labels)
            loss.backward()
            total_loss += loss.item()
            total_acc += (pred_labels.argmax(dim=1) == batch_labels).float().mean().item()
            if idx % 10 == 9:
                pbar.set_description('loss: %.4f acc: %.4f' % (total_loss / idx, total_acc / idx))
        model.eval()
        pbar = tqdm(enumerate(valid_dataset), ncols=0)
        total_loss, total_acc = 0, 0
        for idx, (batch_X, batch_Y, batch_labels) in pbar:
            batch_X, batch_Y, batch_labels = batch_X.to(DEVICE), batch_Y.to(DEVICE), batch_labels.to(DEVICE)
            optimizer.zero_grad()
            pred_labels = model(batch_X, batch_Y)
            loss = criterion(pred_labels, batch_labels)
            total_loss += loss.item()
            total_acc += (pred_labels.argmax(dim=1) == batch_labels).sum().item()
            if idx % 10 == 9:
                pbar.set_description('loss: %.4f acc: %.4f' % (total_loss / idx, total_acc / idx))
        torch.save(model.state_dict(), MODEL_PATH)


        