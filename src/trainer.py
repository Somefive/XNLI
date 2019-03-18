
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from dataloader import MTMDataset, collate_fn
from nltk.translate.bleu_score import corpus_bleu

def get_bleu(preds, labels, length):
    _preds, _labels, _length = preds.detach().numpy().argmax(axis=-1), labels.detach().numpy(), length.detach().numpy()
    refs, hypos = [], []
    for _p, _l, l in zip(_preds, _labels, _length):
        hypos.append(_p[:l])
        refs.append([_l[:l]])
    return corpus_bleu(refs, hypos)


def train(model, train_data_generator, max_epoch=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(max_epoch):
        model.train()
        total_loss, total_bleu = 0, 0
        for idx, (local_batch, local_labels, batch_mask, batch_length) in enumerate(train_data_generator):
            optimizer.zero_grad()
            pred_labels = model(local_batch, batch_length)
            batch_size, seq_len = local_labels.size()
            axis = torch.arange(seq_len)[None,:] < batch_length[:,None]
            pl, ll = torch.masked_select(pred_labels, axis[:,:,None]), torch.masked_select(local_labels, axis)
            loss = criterion(pl.view(-1, model.vocab_size), ll.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_bleu += get_bleu(pred_labels, local_labels, batch_length)
            if idx % 2 == 0:
                print('idx: %d, total loss: %.4f, total bleu: %.3f' % (idx, total_loss, total_bleu))
                total_loss, total_bleu = 0, 0


if __name__ == '__main__':
    
    dataset = MTMDataset(vocab_filenames=['preprocessing/vocab.en.20000', 'preprocessing/vocab.fr.20000'], 
                        text_filenames=['preprocessing/train.fr.20000', 'preprocessing/train.en.20000'], 
                        maxlines_per_file=5000)
    generator = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=16, shuffle=True, num_workers=6)
    from transformer import Transformer
    model = Transformer(max_seq_len=64, vocab_size=20000, n_layers=2, dim=256, d_ff=256, dropout=0.1, heads=4)
    model = model.double()
    train(model, generator, max_epoch=1)
    