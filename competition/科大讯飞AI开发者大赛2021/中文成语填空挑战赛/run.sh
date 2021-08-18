#!/bin/bash

python -u baseline.py \
  --model_name_or_path 'hfl/chinese-xlnet-base' \
  --do_train \
  --do_eval \
  --do_predict \
  --logging_steps=100 \
  --max_seq_length 200 \
  --train_file data/new_train.csv \
  --validation_file data/new_valid.csv \
  --test_file data/new_test.csv \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --output_dir 'models/xlnet' \
  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 16 \
  --per_device_train_batch_size 16 \
  --overwrite_output
