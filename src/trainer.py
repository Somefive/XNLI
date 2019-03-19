
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import load_codes 
from dataset import XLMDataset
import os
from transformer import Transformer
from tqdm import tqdm
from utils import get_bleu


class Trainer:

    def __init__(self, model, evaluate_interval=100, print_interval=2, verbose=True):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.total_loss, self.total_bleu, self.total_acc, self.n_step = 0, 0, 0, 0
        self.evaluate_interval, self.print_interval = evaluate_interval, print_interval
        self.verbose = verbose

    def forward(self, batch, tune=False):
        self.optimizer.zero_grad()
        if not tune:
            local_batch, local_labels, batch_mask, batch_length, batch_pos = batch
            pred_labels = self.model(local_batch, batch_length, batch_pos)
            batch_size, seq_len = local_labels.size()
            batch_mask = batch_mask.byte()
            pl, ll = torch.masked_select(pred_labels, batch_mask[:,:,None]), torch.masked_select(local_labels, batch_mask)
            loss = self.criterion(pl.view(-1, model.vocab_size), ll.view(-1))
        return pred_labels, loss

    def run_batch(self, batch, tune=False):
        pred_labels, loss = self.forward(batch, tune=tune)
        if self.model.training:
            loss.backward()
            self.optimizer.step()
        self.total_loss += loss.item()
        if not tune:
            local_batch, local_labels, batch_mask, batch_length, batch_pos = batch
            self.total_acc += (local_labels == pred_labels.argmax(dim=2)).float().mean().item()
            self.total_bleu += get_bleu(pred_labels, local_labels, batch_length)
        self.step()

    def step(self):
        self.n_step += 1
        
    def reset_summary(self):
        self.total_loss, self.total_bleu, self.total_acc, self.n_step = 0, 0, 0, 0

    def evaluate(self, eval_data_generator, tune=False):
        self.reset_summary()
        self.model.eval()
        for batch in eval_data_generator:
            self.run_batch(batch, tune=tune)
        if not tune:
            print('\nValidation Loss: %.4f, Acc: %.4f, BLEU: %.4f' % (self.total_loss / self.n_step, self.total_acc / self.n_step, self.total_bleu / self.n_step))
        self.reset_summary()
        self.model.train()

    def train(self, train_data_generator, tune=False, eval_data_generator=None, save_path=None):
        self.reset_summary()
        self.model.train()
        if self.verbose:
            pbar = tqdm(ncols=0)
        for batch in train_data_generator:
            self.run_batch(batch, tune=tune)
            if self.verbose and self.n_step % self.print_interval == self.print_interval - 1:
                pbar.update(self.print_interval)
                if not tune:
                    pbar.set_description('Training Loss: %.4f, Acc: %.4f, BLEU: %.4f' % (self.total_loss / self.n_step, self.total_acc / self.n_step, self.total_bleu / self.n_step))
            if self.n_step % self.evaluate_interval == self.evaluate_interval - 1:
                if eval_data_generator:
                    self.evaluate(eval_data_generator, tune=tune)
                if save_path:
                    torch.save(model.state_dict(), save_path)
        if pbar:
            pbar.close()


if __name__ == '__main__':
    VOCAB_SIZE = 20000
    MAX_SEQ_LEN = 64
    MODEL_PATH = 'model/mlm'

    generator_params = {'batch_size': 16, 'shuffle': True, 'num_workers': 6}

    dico, token_counter = load_codes(['preprocessing/vocab.en.20000', 'preprocessing/vocab.fr.20000'])
    # train_dataset = XLMDataset(dico=dico, dataset_size=1000000, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, alpha=0.5, para=False,
    #                     filenames=['preprocessing/train.fr.20000', 'preprocessing/train.en.20000'])
    # train_data_generator = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
    valid_dataset = XLMDataset(dico=dico, dataset_size=1000, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, alpha=0.5, para=False,
                        filenames=['preprocessing/en-fr.en.valid', 'preprocessing/en-fr.fr.valid'])
    train_dataset = XLMDataset(dico=dico, dataset_size=100000, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN, alpha=0.5, para=True,
                        filenames=[['preprocessing/train.fr.20000', 'preprocessing/train.en.20000']])
    valid_data_generator = valid_dataset.get_generator(params=generator_params)
    train_data_generator = train_dataset.get_generator(params=generator_params)
    
    model = Transformer(max_seq_len=MAX_SEQ_LEN, vocab_size=VOCAB_SIZE, n_layers=2, dim=256, d_ff=256, dropout=0.1, heads=4)
    model = model.double()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))

    trainer = Trainer(model, evaluate_interval=100, print_interval=2, verbose=True)
    for epoch in range(5):
        print('Epoch %d' % epoch)
        trainer.train(train_data_generator, tune=False, eval_data_generator=valid_data_generator, save_path=MODEL_PATH)
    