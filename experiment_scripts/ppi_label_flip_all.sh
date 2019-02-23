# Hoang NT
# Randomly flip the binary labels for PPI dataset

cd ..
source activate graph_sage
for i in $(seq 0.05 0.05 0.95)
do
    i=$(echo $i | sed 's/0*$//') 
    echo ==== Noise rate $i ====
    python -m graphsage.supervised_train \
              --train_prefix ~/data/ppi/ppi \
              --model graphsage_maxpool \
              --model_size small \
              --label_flip $i \
              --base_log_dir ./label_flip \
              --sigmoid >log_$i
    cat label_flip/sup-ppi/graphsage_maxpool_small_0.0100/test_stats.txt
    echo ==== DONE ====
done
