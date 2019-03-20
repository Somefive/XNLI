#!/bin/bash
python src/train.py \
--load_path model/en-fr \
--save_path model/en-fr \
--vocab_size 60000 \
--max_seq_len 256 \
--batch_size 8 \
--shuffle True \
--num_workers 1 \
\
--model_n_layers 6 \
--model_dim 128 \
--model_d_ff 512 \
--model_dropout 0.1 \
--model_heads 4 \
\
--max_epoch 100 \
--epoch_size 1000 \
--print_interval 10 \
--verbose True \
\
--mlm_train_set_size 500000 \
--tlm_train_set_size 1000000 \
--tlm_valid_set_size 5000 \
--xnli_train_set_size 400000 \
--xnli_valid_set_size 2500 \
\
--xlm True \
--xnli True \
\
--vocab_path en fr \
--mlm_train_paths en fr \
--tlm_train_paths en-fr.en en-fr.fr \
--tlm_valid_paths en-fr.en en-fr.fr \
--xnli_train_paths s1.en s2.en label.en \
--xnli_valid_paths s1.en s2.en label.en s1.fr s2.fr label.fr \
\
--device cuda:1
