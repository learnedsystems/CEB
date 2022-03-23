#DROPOUT_KIND=$1
BINS=1
BITMAP=1
EPOCHS=-1
QDIR=queries/imdb-unique-plans

TEST_TMPS=all
DECAY=0.0
FEAT_ALIAS_SEP=0
REPS=(1 1 1)
NOREGEX=1
TRAIN_TMPS=(1a 2a 2b 2c 3a 5a 6a 7a 8a)

#TRAIN_TMPS=(2b 2c)
#TRAIN_TMPS=(1a 2a 2b 2c 3a 4a 5a 6a 7a 8a 9a 10a 11a)

EVAL_FNS=qerr,ppc,ppc2

for i in "${!REPS[@]}";
do
  for ti in "${!TRAIN_TMPS[@]}";
  do
    CMD="time python3 main.py --algs mscn \
    --query_dir $QDIR \
    --early_stopping 1 \
    --onehot_dropout 0 \
    --loss_func_name mse \
    --lr 0.0001 \
    --embedding_fn none \
    --weight_decay $DECAY \
    --embedding_pooling sum \
    --max_discrete_featurizing_buckets $BINS \
    --sample_bitmap $BITMAP \
    -n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
    --use_wandb 1 --load_padded_mscn_feats 1 \
    --train_tmps ${TRAIN_TMPS[$ti]} \
    --test_tmps $TEST_TMPS --no_regex $NOREGEX \
    --eval_fns $EVAL_FNS \
    --eval_epoch 100 --feat_onlyseen_preds 1 \
    --feat_separate_alias $FEAT_ALIAS_SEP \
    --table_features 1 --join_features onehot \
    --set_column_feature onehot --feat_mcvs 0 \
    --implied_pred_features 0"
    echo $CMD
    eval $CMD
  done
done

