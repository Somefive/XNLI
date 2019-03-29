import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_xnli, get_batch, build_vocab
from mutils import get_optimizer
from model import BiLSTM, ClassifierNet

from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Encoder Training?')
# paths
lg = "en"
parser.add_argument("--language", type=str, default="en", help="2-letter lang abbv")
parser.add_argument("--xnlipath", type=str, default='./data/xnli/XNLI-1.0', help="XNLI data path ")
parser.add_argument("--outputdir", type=str, default='./output/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--word_emb_path", type=str, default="data/emb/", help="word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0.1, help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0.1, help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="adam,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='BiLSTM', help="see list of encoders")
parser.add_argument("--encoder", type=str, default='BiLSTM', help="only BiLSTM")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=2, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)
device = torch.device("cuda:0")


# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
DATA
"""
lg = "en"
train, valid, test = get_xnli(lg, params.xnlipath)
word_embed = build_vocab([train, valid, test], params.word_emb_path, lg)

"""
MODEL
"""
# model config
config_model = {
    'vocav_size':len(word_embed),
    'word_emb_dim':params.word_emb_dim,
    'enc_lstm_dim':params.enc_lstm_dim,
    'n_enc_layers':params.n_enc_layers,
    'dpout_model':params.dpout_model,
    'dpout_fc':params.dpout_fc,
    'fc_dim':params.fc_dim,
    'bsize':params.batch_size,
    'n_classes' :params.n_classes,
    'pool_type':params.pool_type,
    'nonlinear_fc':params.nonlinear_fc,
    'encoder_type':params.encoder_type,
    'use_cuda':True,

}

# model
classifier = ClassifierNet(config_model).to(device)
print(classifier)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight).to(device) # todo: alignment loss?
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(classifier.parameters(), **optim_params)

# cuda 
#classifier.cuda()
#classifier = classifer.to(device)
#loss_fn.cuda()
#loss_fn = loss_fn.to(device)

"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    device = torch.device("cuda:0")
    print('\nTRAINING : Epoch ' + str(epoch))
    classifier.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    permutation = np.random.permutation(len(train['pairs']))

    pairs = train['pairs']
    s1 = [p[0] for p in pairs]
    s2 = [p[1] for p in pairs]
    target = train['labels']


    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size): # todo: change s1, s2 according to my pair, label structure
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_embed, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_embed, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size]))
        k = s1_batch.size(1)  # actual batch size

        # forward
        output = classifier((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])
        # loss
        loss = loss_fn(output, tgt_batch)
        try: 
            all_costs.append(loss.data[0])
        except IndexError:
            all_costs.append(loss.data.item())
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in classifier.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*correct.item()/(stidx+k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct.item()/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    return train_acc


def evaluate(epoch, eval_type='valid', final_eval=False):
    classifier.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    for i in range(0, len(s1), params.batch_size):
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_embed, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_embed, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size]))

        output = classifier((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * correct.item() / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(classifier.state_dict(), os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc


"""
Train model on Natural Language Inference task
"""
epoch = 1

pbar = tqdm(total = 4)
while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1
    pbar.update(1)
pbar.close()


# Run best model on test set.
classifier.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, 'valid', True)
evaluate(0, 'test', True)

# Save encoder instead of full model
print ("******************** save model!**********************")
torch.save(classifier.encoder.state_dict(), os.path.join(params.outputdir, params.outputmodelname + '.encoder.pkl'))





