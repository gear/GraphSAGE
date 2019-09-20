#!/bin/bash

python -m graphsage.supervised_train \
          --train_prefix high/freq \
          --high_freq 0.5 \
          --model graphsage_seq --sigmoid \
          --epochs 200
