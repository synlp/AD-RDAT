#!/bin/bash

python train_main.py --dataset demo --data_dir ./data/demo --learning_rate 1e-5 --multi_criteria --adversary --batch_size 2  --num_epoch 2 --warmup_proportion 0.06 --fp16 --encoder bilstm  --bert_model bert-model
