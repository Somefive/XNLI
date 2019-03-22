#!/bin/bash
python src/train.py \
--load_path model/en-fr-mini \
--save_path model/en-fr-mini \
--save_path_inc 5 \
--vocab_size 20000 \
--max_seq_len 128 \
--batch_size 64 \
--shuffle True \
--num_workers 12 \
\
--model_n_layers 4 \
--model_dim 256 \
--model_d_ff 512 \
--model_dropout 0.1 \
--model_heads 8 \
--model_encoder_only True \
\
--max_epoch 100 \
--epoch_size 2000 \
--print_interval 1 \
--verbose True \
\
--mlm_train_set_size 5000000 \
--tlm_train_set_size 10000000 \
--tlm_valid_set_size 5000 \
--xnli_train_set_size 400000 \
--xnli_valid_set_size 2500 \
\
--tlm False \
--xnli Tune \
\
--vocab_path en fr \
--mlm_train_paths en fr \
--tlm_train_paths en-fr.en en-fr.fr \
--tlm_valid_paths en-fr.en en-fr.fr \
--xnli_train_paths s1.en s2.en label.en \
--xnli_valid_paths s1.en s2.en label.en s1.fr s2.fr label.fr \
\
--device cuda:0 \
--multiple_gpu True \
--fp16 False \
--lr 2e-4
