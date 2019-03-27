import numpy as np
import torch
import torch.nn as nn

"""
BLSTM_encoder. Trained on English, being approximated by other lg's encoder
"""
class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.batch_size = config['bsize'] # default: 64
        self.word_emb_dim = config['word_emb_dim'] # 300
        self.enc_lstm_dim = config['enc_lstm_dim'] # 2048 # todo: tune it at 256 or 512 firs?
        self.pool_type = config['pool_type'] # max
        self.dpout_model = config['dpout_model'] # 0.1 # todo: what is dpout_fc and what value should it take in train.py?

        self.enc_lstm = nn.LSTM(input_size=self.word_emb_dim,
                                hidden_size=self.enc_lstm_dim,
                                num_layers=1, # TODO: try 2?
                                bidirectional=True,
                                dropout=self.dpout_model)
        #self.embed todo: necessary?

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: (seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        batch_size = sent.size(1)

        # Sort by length (keep idx)BiLSTM
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        #sent = sent.index_select(1, torch.LongTensor(idx_sort))
        sent = sent.index_select(1, torch.cuda.LongTensor(idx_sort))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed)[0]
        # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        #sent_output = sent_output.index_select(1, torch.LongTensor(idx_unsort))
        sent_output = sent_output.index_select(1, torch.cuda.LongTensor(idx_unsort))

        # Pooling
        if self.pool_type == "mean":
            #sent_len = torch.FloatTensor(sent_len).unsqueeze(1)
            sent_len = torch.FloatTensor(sent_len).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max": # todo delete the pool_type
            emb = torch.max(sent_output, 0)[0].squeeze(0)

        return emb


"""
Main module for Classification(Testing?) 
"""
class ClassifierNet(nn.Module):
    def __init__(self, config):
        super(ClassifierNet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        self.encoder = eval(self.encoder_type)(config)
        #self.inputdim = 2*self.enc_lstm_dim
        self.inputdim = 4 * 2 * self.enc_lstm_dim
        #self.inputdim = 4*self.inputdim if self.encoder_type == "ConvNetEncoder" else self.inputdim
        #self.inputdim = self.enc_lstm_dim if self.encoder_type =="LSTMEncoder" else self.inputdim
        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
            )
            """
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, 512),
                nn.Linear(512, self.n_classes),
            )
            """

    def forward(self, s1, s2):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        v = self.encoder(s2)

        features = torch.cat((u, v, torch.abs(u - v), u * v), 1)
        output = self.classifier(features)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb

