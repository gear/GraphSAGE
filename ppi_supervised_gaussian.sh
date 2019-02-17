python -m graphsage.supervised_train --train_prefix ~/data/ppi/ppi --feats_suffix "gaussian0_$1" --model graphsage_maxpool --model_size small --sigmoid --identity_dim $2
