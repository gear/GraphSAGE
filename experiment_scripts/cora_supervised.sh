#!/bin/bash

python -m graphsage.supervised_train \
          --train_prefix ~/data/cora_gcn/cora \
          --model graphsage_seq --sigmoid \
          --epochs 100
