for i in $(seq 0.05 0.05 0.95)
do
    i=$(echo $i | sed 's/0*$//') 
    echo ==== Training at noise rate ==== $i
    python -m graphsage.supervised_train \
              --train_prefix ~/data/ppi/ppi \
              --feats_suffix "swap_train_$i" \
              --model graphsage_maxpool \
              --model_size small \
              --sigmoid
    echo ==== DONE ====
done
