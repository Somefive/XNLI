import os
from transformer import Transformer
from dataloader import load_vocab 
from dataset import XLMDataset, composed_dataloader, MaskedDataset, XNLIDataset
from trainer import Trainer
import torch
import argparse
from utils import extract_path

def parse_args():

    def str2bool(x):
        return x != 'False'

    def str2ints(x):
        return [int(i) for i in x.split(',')]

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", default='model/default', type=str, help="load model from this location")
    parser.add_argument("--save_path", default='model/default', type=str, help="store model in this location")
    parser.add_argument("--save_path_inc", default=0, type=int, help="store model with increment number every <save_path_inc> epoch. default is 0 means not increment.")
    parser.add_argument("--vocab_size", default=20000, type=int, help="vocabulary size")
    parser.add_argument("--max_seq_len", default=64, type=int, help="max sequence size")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--shuffle", default=True, type=bool, help="dataset shuffle")
    parser.add_argument("--num_workers", default=6, type=int, help="dataset fetch worker number")
    
    parser.add_argument("--model_n_layers", default=2, type=int, help="model layers")
    parser.add_argument("--model_dim", default=256, type=int, help="model hidden dimensions")
    parser.add_argument("--model_d_ff", default=512, type=int, help="model feed forward dimensions")
    parser.add_argument("--model_dropout", default=0.1, type=float, help="model dropout")
    parser.add_argument("--model_heads", default=4, type=int, help="model heads")
    parser.add_argument("--model_encoder_only", default=True, type=str2bool, help="model only use encoder")

    parser.add_argument("--max_epoch", default=10, type=int, help="trainer max epoch")
    parser.add_argument("--epoch_size", default=10, type=int, help="trainer epoch size")
    parser.add_argument("--print_interval", default=2, type=int, help="trainer print interval")
    parser.add_argument("--verbose", default=True, type=bool, help="trainer verbose")

    parser.add_argument("--mlm_train_set_size", default=1000000, type=int, help="train dataset size for mlm")
    parser.add_argument("--mlm_valid_set_size", default=1000, type=int, help="valid dataset size for mlm")
    parser.add_argument("--tlm_train_set_size", default=1000000, type=int, help="train dataset size for tlm")
    parser.add_argument("--tlm_valid_set_size", default=1000, type=int, help="valid dataset size for tlm")
    parser.add_argument("--xnli_train_set_size", default=1000000, type=int, help="train dataset size for xnli")
    parser.add_argument("--xnli_valid_set_size", default=1000, type=int, help="valid dataset size for xnli")
    parser.add_argument("--xnli_test_set_size", default=1000, type=int, help="test dataset size for xnli")

    parser.add_argument("--mlm", default=False, type=str2bool, help="Enable MLM")
    parser.add_argument("--tlm", default=False, type=str2bool, help="Enable TLM")
    parser.add_argument("--xlm", default=False, type=str2bool, help="Enable MLM+TLM")
    parser.add_argument("--xnli", default=False, type=str2bool, help="Enable XNLI Fine Tune")
    parser.add_argument("--xnli_test", default=False, type=str2bool, help="Enable XNLI Test")

    parser.add_argument("--vocab_paths", nargs='+', help="vocabulary load paths")
    parser.add_argument("--mlm_train_paths", nargs='*', help="MLM train dataset paths")
    parser.add_argument("--mlm_valid_paths", nargs='*', help="MLM valid dataset paths")
    parser.add_argument("--tlm_train_paths", nargs='*', help="TLM train dataset paths")
    parser.add_argument("--tlm_valid_paths", nargs='*', help="TLM valid dataset paths")
    parser.add_argument("--xnli_langs", nargs='*', help="XNLI langs")

    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate of optimizer")

    parser.add_argument("--device", default='cpu', type=str, help="device to use")
    parser.add_argument("--fp16", default=False, type=str2bool, help="using fp16")
    parser.add_argument("--gpus", default=[0], type=str2ints, help="gpu ids")

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    dico, token_counter = load_vocab(extract_path(args.vocab_paths, prefix='data/vocab/'), size=args.vocab_size)

    generator_params = {
        'batch_size': args.batch_size, 
        'shuffle': args.shuffle, 
        'num_workers': args.num_workers
    }
    dataset_params = {
        'dico': dico,
        'max_seq_len': args.max_seq_len 
    }
    trainer_params = {
        'epoch_size': args.epoch_size,
        'print_interval': args.print_interval,
        'verbose': args.verbose,
        'lr': args.lr,
        'fp16': args.fp16,
        'gpus': args.gpus,
        'device': args.device
    }
    model_params = {
        'max_seq_len': args.max_seq_len,
        'vocab_size': args.vocab_size,
        'n_layers': args.model_n_layers,
        'dim': args.model_dim,
        'd_ff': args.model_d_ff,
        'dropout': args.model_dropout,
        'heads': args.model_heads,
        'encoder_only': args.model_encoder_only
    }

    model = Transformer(**model_params).float()
    trainer = Trainer(model, **trainer_params)
    trainer.load_model(args.load_path)


    if args.mlm or args.xlm:
        train_mlm_data_generator = MaskedDataset(filenames=extract_path(args.mlm_train_paths, prefix='data/mlm/', suffix='.train'), 
                                                 maxlines=args.mlm_train_set_size, **dataset_params).get_generator(
                                                 params=generator_params) if args.mlm_train_paths is not None else None
        valid_mlm_data_generator = MaskedDataset(filenames=extract_path(args.mlm_valid_paths, prefix='data/mlm/', suffix='.valid'), 
                                                 maxlines=args.mlm_valid_set_size, **dataset_params).get_generator(
                                                 params=generator_params) if args.mlm_valid_paths is not None else None
    if args.tlm or args.xlm:
        train_tlm_data_generator = MaskedDataset(filenames=extract_path(args.tlm_train_paths, prefix='data/tlm/', suffix='.train', groupby=2),
                                                 maxlines=args.tlm_train_set_size, **dataset_params).get_generator(
                                                 params=generator_params) if args.tlm_train_paths is not None else None
        valid_tlm_data_generator = MaskedDataset(filenames=extract_path(args.tlm_valid_paths, prefix='data/tlm/', suffix='.valid', groupby=2), 
                                                 maxlines=args.tlm_valid_set_size, **dataset_params).get_generator(
                                                 params=generator_params) if args.tlm_valid_paths is not None else None
    if args.xnli:
        train_xnli_data_generator = XNLIDataset(lang='en', type_='train',
                                                maxlines=args.xnli_train_set_size, **dataset_params).get_generator(
                                                params=generator_params)
        valid_xnli_data_generators = dict()
        if args.xnli_langs is not None:
            for lang in args.xnli_langs:
                valid_xnli_data_generators[lang] = XNLIDataset(lang=lang, type_='valid',
                                                               maxlines=args.xnli_valid_set_size, **dataset_params).get_generator(
                                                               params=generator_params)
        test_xnli_data_generators = dict()
        if args.xnli_langs is not None:
            for lang in args.xnli_langs:
                test_xnli_data_generators[lang] = XNLIDataset(lang=lang, type_='test',
                                                               maxlines=args.xnli_test_set_size, **dataset_params).get_generator(
                                                               params=generator_params)
    print('data prepared')

    for epoch in range(args.max_epoch):
        print('Epoch %d' % epoch)
        save_path = args.save_path if args.save_path_inc == 0 else '%s-%d' % (args.save_path, epoch / args.save_path_inc)
        if args.xlm:
            if train_mlm_data_generator and train_tlm_data_generator:
                train_xlm_data_generator = composed_dataloader(train_mlm_data_generator, train_tlm_data_generator)
                trainer.train(train_xlm_data_generator, tune=False, save_path=save_path, name='[XLM] ')
            if valid_mlm_data_generator:
                trainer.evaluate(valid_mlm_data_generator, tune=False, name='[MLM] ')
            if valid_tlm_data_generator:
                trainer.evaluate(valid_tlm_data_generator, tune=False, name='[TLM] ')
        elif args.mlm:
            if train_mlm_data_generator:
                trainer.train(train_mlm_data_generator, tune=False, save_path=save_path, name='[MLM] ')
            if valid_mlm_data_generator:
                trainer.evaluate(valid_mlm_data_generator, tune=False, name='[MLM] ')
        elif args.tlm:
            if train_tlm_data_generator:
                trainer.train(train_tlm_data_generator, tune=False, save_path=save_path, name='[TLM] ')
            if valid_tlm_data_generator:
                trainer.evaluate(valid_tlm_data_generator, tune=False, name='[TLM] ')
        elif args.xnli:
            if args.xnli_test:
                for lang, test_xnli_data_generator in test_xnli_data_generators.items():
                    trainer.evaluate(test_xnli_data_generator, tune=True, name='[XNLI-%s-Test] ' % lang)
            else:
                if train_xnli_data_generator:
                    trainer.train(train_xnli_data_generator, tune=True, save_path=save_path, name='[XNLI] ')
                for lang, valid_xnli_data_generator in valid_xnli_data_generators.items():
                    trainer.evaluate(valid_xnli_data_generator, tune=True, name='[XNLI-%s-Valid] ' % lang)
            
