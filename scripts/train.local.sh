#!/bin/bash
python src/train.py \
--load_path model/en-fr \
--save_path model/en-fr \
--vocab_size 30000 \
--max_seq_len 128 \
--batch_size 32 \
--shuffle True \
--num_workers 12 \
\
--model_n_layers 4 \
--model_dim 256 \
--model_d_ff 1024 \
--model_dropout 0.1 \
--model_heads 16 \
--model_encoder_only True \
\
--max_epoch 100 \
--epoch_size 10 \
--print_interval 1 \
--verbose True \
\
--mlm_train_set_size 50000 \
--tlm_train_set_size 100000 \
--tlm_valid_set_size 5000 \
--xnli_train_set_size 400000 \
--xnli_valid_set_size 2500 \
\
--xlm True \
--xnli False \
\
--vocab_path en fr \
--mlm_train_paths en fr \
--tlm_train_paths en-fr.en en-fr.fr \
--tlm_valid_paths en-fr.en en-fr.fr \
--xnli_train_paths s1.en s2.en label.en \
--xnli_valid_paths s1.en s2.en label.en s1.fr s2.fr label.fr \
\
--device cpu
