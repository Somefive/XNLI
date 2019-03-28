#!/biN/BASH
python src/train.py \
--load_path model/en-fr-ex \
--save_path model/en-fr-ex \
--save_path_inc 0 \
--vocab_size 20400 \
--max_seq_len 256 \
--batch_size 128 \
--shuffle True \
--num_workers 12 \
\
--model_n_layers 6 \
--model_dim 512 \
--model_d_ff 1024 \
--model_dropout 0.1 \
--model_heads 8 \
--model_encoder_only True \
\
--max_epoch 100 \
--epoch_size 2000 \
--print_interval 1 \
--verbose True \
\
--mlm_train_set_size 1000000 \
--mlm_valid_set_size 5000 \
--tlm_train_set_size 10000000 \
--tlm_valid_set_size 5000 \
--xnli_train_set_size 400000 \
--xnli_valid_set_size 2500 \
\
--tlm True \
\
--vocab_path en fr \
--mlm_train_paths en \
--mlm_valid_paths en \
--tlm_train_paths en-fr.en en-fr.fr \
--tlm_valid_paths en-fr.en en-fr.fr \
--xnli_langs en fr \
--xnli_test False \
\
--device cuda:0 \
--gpus 0,1,2 \
--fp16 False \
--lr 2e-4
