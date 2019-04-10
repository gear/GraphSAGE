python -m graphsage.supervised_train \
       --train_prefix ~/data/ppi/ppi \
       --model graphsage_maxpool \
       --model_size small \
       --sigmoid \
       --gaussian_noise \
       --gaussian_mean 0 \
       --gaussian_std 1 \
       --epochs 10
