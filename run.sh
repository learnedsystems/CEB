#bash run_all_diff.sh 1
#bash run_all_diff.sh 0
#bash run_all_diff.sh 0 embeddings1.pkl

SWA=swa
BINS=10
FLNORM=1
EPOCHS=10
QDIR=queries/imdb
#TRAIN_TMPS=1a
#TEST_TMPS=2a,2b
TRAIN_TMPS=2b
TEST_TMPS=2a
#EMBS=(none embeddings/sampled_data_new30.pkl)
EMBS=(none)
DECAY=0.0
FEAT_ALIAS_SEP=0
#REPS=(1)
REPS=(1 1 1 1 1)

for i in "${!REPS[@]}";
do
  for ei in "${!EMBS[@]}";
  do
  CMD="time python3 main.py --algs mscn \
  --training_opt $SWA \
  --query_dir $QDIR \
  --loss_func_name mse \
  --lr 0.00001 \
  --normalize_flow_loss $FLNORM \
  --embedding_fn ${EMBS[$ei]} \
  --weight_decay $DECAY \
  --embedding_pooling sum \
  --max_discrete_featurizing_buckets $BINS \
  -n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  --use_wandb 1 --load_padded_mscn_feats 1 \
  --train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  --eval_epoch 100 --feat_onlyseen_preds 1 \
  --feat_separate_alias $FEAT_ALIAS_SEP \
  --table_features 1 --join_features onehot \
  --set_column_feature onehot --feat_mcvs 0 \
  --implied_pred_features 0"
  echo $CMD
  eval $CMD
  done
done


#for i in "${!REPS[@]}";
#do
  #for ei in "${!EMBS[@]}";
  #do
  #CMD="time python3 main.py --algs mscn \
  #--query_dir $QDIR \
  #--loss_func_name flowloss \
  #--lr 0.00001 \
  #--normalize_flow_loss $FLNORM \
  #--embedding_fn ${EMBS[$ei]} \
  #--weight_decay $DECAY \
  #--embedding_pooling sum \
  #--max_discrete_featurizing_buckets $BINS \
  #-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  #--use_wandb 1 --load_padded_mscn_feats 1 \
  #--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  #--eval_epoch 100 --feat_onlyseen_preds 1 \
  #--feat_separate_alias $FEAT_ALIAS_SEP \
  #--table_features 0 --join_features stats \
  #--set_column_feature stats --feat_mcvs 0 \
  #--implied_pred_features 0"
  #echo $CMD
  #eval $CMD
  #done
#done

#for i in "${!REPS[@]}";
#do
  #for ei in "${!EMBS[@]}";
  #do
  #CMD="time python3 main.py --algs mscn \
  #--query_dir $QDIR \
  #--embedding_fn ${EMBS[$ei]} \
  #--weight_decay $DECAY \
  #--embedding_pooling sum \
  #--max_discrete_featurizing_buckets $BINS \
  #-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  #--use_wandb 1 --load_padded_mscn_feats 1 \
  #--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  #--eval_epoch 100 --feat_onlyseen_preds 1 \
  #--feat_separate_alias $FEAT_ALIAS_SEP \
  #--table_features 0 --join_features 0 \
  #--set_column_feature 0 --feat_mcvs 0 \
  #--implied_pred_features 0"
  #echo $CMD
  #eval $CMD
  #done
#done

#for i in "${!REPS[@]}";
#do
  #for ei in "${!EMBS[@]}";
  #do
  #CMD="time python3 main.py --algs mscn \
  #--query_dir $QDIR \
  #--embedding_fn ${EMBS[$ei]} \
  #--weight_decay $DECAY \
  #--embedding_pooling sum \
  #--max_discrete_featurizing_buckets $BINS \
  #-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  #--use_wandb 1 --load_padded_mscn_feats 1 \
  #--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  #--eval_epoch 100 --feat_onlyseen_preds 1 \
  #--feat_separate_alias $FEAT_ALIAS_SEP \
  #--table_features 0 --join_features stats \
  #--set_column_feature stats --feat_mcvs 0 \
  #--implied_pred_features 0"
  #echo $CMD
  #eval $CMD
  #done
#done

#for i in "${!REPS[@]}";
#do
  #for ei in "${!EMBS[@]}";
  #do
  #CMD="time python3 main.py --algs mscn \
  #--query_dir $QDIR \
  #--embedding_fn ${EMBS[$ei]} \
  #--weight_decay $DECAY \
  #--embedding_pooling sum \
  #--max_discrete_featurizing_buckets $BINS \
  #-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  #--use_wandb 1 --load_padded_mscn_feats 1 \
  #--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS \
  #--no_regex 1 \
  #--eval_epoch 100 --feat_onlyseen_preds 1 \
  #--feat_separate_alias $FEAT_ALIAS_SEP \
  #--set_column_feature onehot \
  #--table_features 1 --join_features onehot"
  #echo $CMD
  #eval $CMD
  #done
#done

#for i in "${!REPS[@]}";
#do
  #for ei in "${!EMBS[@]}";
  #do
  #CMD="time python3 main.py --algs mscn \
  #--query_dir $QDIR \
  #--embedding_fn ${EMBS[$ei]} \
  #--weight_decay $DECAY \
  #--embedding_pooling sum \
  #--max_discrete_featurizing_buckets $BINS \
  #-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  #--use_wandb 1 --load_padded_mscn_feats 1 \
  #--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  #--eval_epoch 100 --feat_onlyseen_preds 1 \
  #--feat_separate_alias $FEAT_ALIAS_SEP \
  #--table_features 0 --join_features onehot"
  #echo $CMD
  #eval $CMD
  #done
#done

#for i in "${!REPS[@]}";
#do
  #for ei in "${!EMBS[@]}";
  #do
  #CMD="time python3 main.py --algs mscn \
  #--query_dir $QDIR \
  #--embedding_fn ${EMBS[$ei]} \
  #--weight_decay $DECAY \
  #--embedding_pooling sum \
  #--max_discrete_featurizing_buckets $BINS \
  #-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  #--use_wandb 1 --load_padded_mscn_feats 1 \
  #--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  #--eval_epoch 100 --feat_onlyseen_preds 1 \
  #--feat_separate_alias $FEAT_ALIAS_SEP \
  #--table_features 0 --join_features onehot \
  #--set_column_feature onehot --feat_mcvs 1 --implied_pred_features 1"
  #echo $CMD
  #eval $CMD
  #done
#done

#for i in "${!REPS[@]}";
#do
  #for ei in "${!EMBS[@]}";
  #do
  #CMD="time python3 main.py --algs mscn \
  #--query_dir $QDIR \
  #--embedding_fn ${EMBS[$ei]} \
  #--weight_decay $DECAY \
  #--embedding_pooling sum \
  #--max_discrete_featurizing_buckets $BINS \
  #-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  #--use_wandb 1 --load_padded_mscn_feats 1 \
  #--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  #--eval_epoch 100 --feat_onlyseen_preds 1 \
  #--feat_separate_alias $FEAT_ALIAS_SEP \
  #--table_features 0 --join_features stats \
  #--set_column_feature stats --feat_mcvs 0"
  #echo $CMD
  #eval $CMD
  #done
#done

#for i in "${!REPS[@]}";
#do
  #for ei in "${!EMBS[@]}";
  #do
  #CMD="time python3 main.py --algs mscn \
  #--query_dir $QDIR \
  #--embedding_fn ${EMBS[$ei]} \
  #--weight_decay $DECAY \
  #--embedding_pooling sum \
  #--max_discrete_featurizing_buckets $BINS \
  #-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  #--use_wandb 1 --load_padded_mscn_feats 1 \
  #--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  #--eval_epoch 100 --feat_onlyseen_preds 1 \
  #--feat_separate_alias $FEAT_ALIAS_SEP \
  #--table_features 1 --join_features stats"
  #echo $CMD
  #eval $CMD
  #done
#done

#for i in "${!REPS[@]}";
#do
  #for ei in "${!EMBS[@]}";
  #do
  #CMD="time python3 main.py --algs mscn \
  #--query_dir $QDIR \
  #--embedding_fn ${EMBS[$ei]} \
  #--weight_decay $DECAY \
  #--embedding_pooling sum \
  #--max_discrete_featurizing_buckets $BINS \
  #-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  #--use_wandb 1 --load_padded_mscn_feats 1 \
  #--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  #--eval_epoch 100 --feat_onlyseen_preds 1 \
  #--feat_separate_alias $FEAT_ALIAS_SEP \
  #--table_features 0 --join_features stats \
  #--set_column_feature stats"
  #echo $CMD
  #eval $CMD
  #done
#done

