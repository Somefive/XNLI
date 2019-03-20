import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import get_bleu
import datetime

class Trainer:

    def __init__(self, model, epoch_size=10000000, print_interval=2, verbose=True, device='cpu'):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.total_loss, self.total_bleu, self.total_acc, self.n_step = 0, 0, 0, 0
        self.epoch_size, self.print_interval = epoch_size, print_interval
        self.verbose = verbose

    def forward(self, batch, tune=False):
        self.optimizer.zero_grad()
        if not tune:
            local_batch, local_labels, batch_mask, batch_length, batch_pos = batch
            if self.model.training:
                loss = self.model.forward(local_batch, batch_length, batch_pos, batch_mask, local_labels, with_prob=False)
            else:
                local_labels, pred_labels, loss = self.model.forward(local_batch, batch_length, batch_pos, batch_mask, local_labels, with_prob=True)
        else:
            local_batch, batch_length, batch_pos, local_labels = batch
            pred_labels = self.model.classify(local_batch, batch_length, batch_pos)
            loss = self.criterion(pred_labels, local_labels)
        if self.model.training:
            return loss
        else:
            return local_labels, pred_labels, loss

    def run_batch(self, batch, tune=False):
        if self.model.training:
            loss = self.forward(batch, tune=tune)
        else:
            local_labels, pred_labels, loss = self.forward(batch, tune=tune)
            self.total_acc += (local_labels == pred_labels.argmax(dim=1)).float().mean().item()
        loss = loss.mean()
        if self.model.training:
            loss.backward()
            self.optimizer.step()
        self.total_loss += loss.item()
        self.step()

    def step(self):
        self.n_step += 1
        
    def reset_summary(self):
        self.total_loss, self.total_bleu, self.total_acc, self.n_step = 0, 0, 0, 0

    def evaluate(self, eval_data_generator, tune=False, name=''):
        self.reset_summary()
        self.model.eval()
        for batch in eval_data_generator:
            batch = [b.to(self.device) for b in batch]
            self.run_batch(batch, tune=tune)
        print('%sValidation Loss: %.4f, Acc: %.4f' % (name, self.total_loss / self.n_step, self.total_acc / self.n_step))

    def train(self, train_data_generator, tune=False, save_path=None, epoch_size=None, name=''):
        if epoch_size is None:
            epoch_size = self.epoch_size
        self.reset_summary()
        self.model.train()
        if self.verbose:
            pbar = tqdm(zip(range(epoch_size), train_data_generator), ncols=0)
        for _, batch in pbar:
            batch = [b.to(self.device) for b in batch]
            self.run_batch(batch, tune=tune)
            if self.verbose and self.n_step % self.print_interval == self.print_interval - 1:
                pbar.set_description('%sTraining Loss: %.4f' % (name, self.total_loss / self.n_step))
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            print('[%s] model saved to %s' % (datetime.datetime.now(), save_path))

    def to(self, device):
        self.device = device
        return self
    
