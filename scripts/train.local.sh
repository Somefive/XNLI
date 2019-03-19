#!/bin/bash
python src/train.py \
--model_d_ff 256 \
--xlm True --xnli True \
--vocab_path en.20000 fr.20000 \
--mlm_train_paths en fr \
--tlm_train_paths en-fr.en en-fr.fr \
--tlm_valid_paths en-fr.en en-fr.fr \
--xnli_train_paths s1.en s2.en label.en \
--xnli_valid_paths s1.en s2.en label.en s1.fr s2.fr label.fr \
--device cpu