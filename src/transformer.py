#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class MultiHeadAttention(nn.Module):
    
        def __init__(self, dim=1024, heads=8, dropout=0.1):
            super(MultiHeadAttention, self).__init__()
            self.dim = dim
            self.heads = heads
            self.d_k = dim // heads
            self.fc_Q = nn.Linear(dim, dim)
            self.fc_K = nn.Linear(dim, dim)
            self.fc_V = nn.Linear(dim, dim)
            self.scale = self.d_k ** -0.5
            self.fc = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(dropout)
            self.softmax = nn.Softmax(dim=-1)
            self.res_dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(dim)
            torch.nn.init.xavier_uniform_(self.fc_Q.weight)
            torch.nn.init.xavier_uniform_(self.fc_K.weight)
            torch.nn.init.xavier_uniform_(self.fc_V.weight)
            torch.nn.init.xavier_uniform_(self.fc.weight)

        def forward(self, q, k, v, mask):
            batch_size, qlen, dim = q.size()
            batch_size, klen, dim = k.size()
            d_k, heads = self.d_k, self.heads
            res = q
            q = self.fc_Q(q).view(batch_size, -1, heads, d_k).transpose(1,2)
            k = self.fc_K(k).view(batch_size, -1, heads, d_k).transpose(1,2)
            v = self.fc_V(v).view(batch_size, -1, heads, d_k).transpose(1,2)
            attention = torch.matmul(q, k.transpose(2, 3)) * self.scale
            # print(attention.size())
            if mask.dim() == 2:
                mask = (mask == 0).view(batch_size, 1, 1, klen).expand_as(attention)
            else:
                mask = (mask == 0).view(batch_size, 1, klen, klen).expand_as(attention)
            attention.masked_fill_(mask, -np.inf)
            attention = self.dropout(self.softmax(attention))
            context = torch.matmul(attention, v).transpose(1,2).contiguous().view(batch_size, -1, dim)
            context = self.res_dropout(self.fc(context))
            return self.layer_norm(context + res), attention


class PositionalEncoding(nn.Module):
    
    def __init__(self, dim, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dim, self.max_seq_len = dim, max_seq_len
        pe = np.array([
            [pos / (10000 ** (2.0 * (j // 2) / dim)) for j in range(dim)]
            for pos in range(max_seq_len)
        ])
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        self.pe_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=dim)
        self.pe_embedding.from_pretrained(torch.as_tensor(pe), freeze=True)

    def forward(self, pos): # pos: (batch_size, seq_len)
        return self.pe_embedding(pos)


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, dim, d_ff=2048, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, d_ff)
        self.fc2 = nn.Linear(d_ff, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        res = x
        x = gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return self.layer_norm(x + res)


class TransformerLayer(nn.Module):

    def __init__(self, decoder=False, dim=1024, d_ff=2048, dropout=0.1, heads=8):
        super(TransformerLayer, self).__init__()
        self.decoder = decoder
        self.ffn = PositionalWiseFeedForward(dim=dim, d_ff=d_ff, dropout=dropout)
        self.self_attention = MultiHeadAttention(dim=dim, heads=heads, dropout=dropout)
        if decoder:
            self.enc_attention = MultiHeadAttention(dim=dim, heads=heads, dropout=dropout)
    
    def forward(self, x, self_mask, enc_output=None, enc_mask=None):
        batch_size, seq_len, dim = x.size()
        output, attention = self.self_attention(x, x, x, self_mask)
        if self.decoder:
            output, attention = self.enc_attention(output, enc_output, enc_output, enc_mask)
        output = self.ffn(output)
        return output


class Transformer(nn.Module):

    def __init__(self, max_seq_len, vocab_size, 
                 n_layers=6, dim=1024, d_ff=2048, dropout=0.1, heads=8, encoder_only=True, n_langs=15):
        super(Transformer, self).__init__()
        self.n_layers, self.max_seq_len, self.dim = n_layers, max_seq_len, dim
        self.encoder_only = encoder_only
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.pe = PositionalEncoding(dim=dim, max_seq_len=max_seq_len)
        self.lang_embed = nn.Embedding(n_langs, dim)
        self.encoders = nn.ModuleList([
            TransformerLayer(decoder=False, dim=dim, d_ff=d_ff, dropout=dropout, heads=heads)
            for _ in range(n_layers)
        ])
        if not encoder_only:
            self.decoders = nn.ModuleList([
                TransformerLayer(decoder=True, dim=dim, d_ff=d_ff, dropout=dropout, heads=heads)
                for _ in range(n_layers)
            ])
        self.pred = nn.AdaptiveLogSoftmaxWithLoss(in_features=dim, n_classes=vocab_size, cutoffs=[8000, 20000], head_bias=True)
        self.xnli_fc = nn.Linear(dim, 3)
        torch.nn.init.xavier_uniform_(self.embed.weight)
        torch.nn.init.xavier_uniform_(self.lang_embed.weight)
        torch.nn.init.xavier_uniform_(self.xnli_fc.weight)

    def encode(self, x, length, pos, langs):
        batch_size, seq_len = x.size()
        rng = torch.arange(seq_len, dtype=torch.long, device=length.device)
        self_mask = rng < length[:, None]
        x = self.embed(x)
        x = x + self.pe(pos) + self.lang_embed(langs)
        enc_output = x
        for encoder in self.encoders:
            enc_output = encoder(enc_output, self_mask)
        if self.encoder_only:
            return enc_output
        else:
            enc_mask = (rng[None, :] <= rng[:, None]).repeat(batch_size, 1, 1)
            dec_output = x
            for decoder in self.decoders:
                dec_output = decoder(dec_output, self_mask, enc_output=enc_output, enc_mask=enc_mask)
            return dec_output

    def forward(self, x, length, pos, langs, mask=None, y=None, with_prob=False):
        if mask is None:
            return self.xnli_fc(self.encode(x, length, pos, langs)[:,0,:].squeeze())
        mask = mask.byte() 
        hidden_state = self.encode(x, length, pos, langs)
        hidden_state_masked = torch.masked_select(hidden_state, mask[:,:,None]).view(-1, self.dim)
        y_masked = torch.masked_select(y, mask)
        _, loss = self.pred(hidden_state_masked, y_masked)
        if not with_prob:
            return loss
        else:
            y_pred_masked = self.pred.log_prob(hidden_state_masked)
            return y_masked, y_pred_masked, loss

if __name__ == '__main__':
    model = Transformer(100, 100, 100, n_layers=2, dim=256, d_ff=256, dropout=0.1, heads=4)
