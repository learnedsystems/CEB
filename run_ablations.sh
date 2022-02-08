ALG=$1
SKIP7a=0
EPOCHS=20
QDIR=queries/imdb

## everything off
time python3 main.py --algs $ALG --query_template all -n -1 \
--query_dir $QDIR --regen_featstats 0 \
--max_epochs $EPOCHS --table_features 0 --join_features 0 \
--max_discrete_featurizing_buckets 1 --mb_size 4096 \
--load_padded_mscn_feats 0 --skip7a $SKIP7a \
--eval_epoch 20 --set_column_feature none

## only bins = 1
time python3 main.py --algs $ALG --query_template all -n -1 \
--query_dir $QDIR --regen_featstats 0 \
--max_epochs $EPOCHS --table_features 1 --join_features 1 \
--max_discrete_featurizing_buckets 1 --mb_size 4096 \
--load_padded_mscn_feats 0 --skip7a $SKIP7a \
--eval_epoch 20 --set_column_feature onehot

## only bins = 10, rest off
time python3 main.py --algs $ALG --query_template all -n -1 \
--query_dir $QDIR --regen_featstats 0 \
--max_epochs $EPOCHS --table_features 0 --join_features 0 \
--max_discrete_featurizing_buckets 10 --mb_size 4096 \
--load_padded_mscn_feats 0 --skip7a $SKIP7a \
--eval_epoch 20 --set_column_feature none

## default run
time python3 main.py --algs $ALG --query_template all -n -1 \
--query_dir $QDIR --regen_featstats 0 \
--max_epochs $EPOCHS --table_features 1 --join_features 1 \
--max_discrete_featurizing_buckets 10 --mb_size 4096 \
--load_padded_mscn_feats 0 --skip7a $SKIP7a \
--eval_epoch 20 --set_column_feature onehot
