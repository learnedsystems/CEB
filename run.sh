
#LOSS_FUNC=$1
#FLOW_FEATS=$2
#ONEHOT_DROPOUT=$3
#ESTOP=$4
#BITMAP=$5
#JOINBITMAP=$6

#MAX_EPOCHS="${7:-20}"
#SKIP7a="${8:-0}"
#BUCKETS="${9:-1}"
#HLS="${10:-128}"
#NH="${11:-2}"
#EVAL_QDIR="${13:-queries/job-joinkeys-tebow-temps/}"

#HLS=512
## default joinbitmap + dropout run

#bash run_default.sh mse 1 2 0 0 1 10 0 1 $HLS 2 5433 queries/job-joinkeys-tebow-temps 1
#bash run_default.sh mse 1 2 0 0 1 10 0 1 $HLS 2 5433 queries/job-joinkeys-tebow-temps 0
#bash run_default.sh mse 1 2 0 0 1 5 0 1 $HLS 2 5433 queries/job-joinkeys-tebow-temps 0

#bash run_default.sh mse 1 2 0 0 1 20 0 1 $HLS 2 5433 queries/job-joinkeys-tebow-temps

#bash run_default.sh mse 1 2 0 0 1 10 0 1 $HLS 2 5433 queries/job-joinkeys-tebow-temps
#bash run_default.sh mse 1 2 0 0 1 10 0 1 $HLS 2 5433 queries/job-joinkeys-tebow-temps

## default, samplebitmap approach
#bash run_default.sh mse 1 0 0 1 0 10 0 1 $HLS 2 5433 queries/job-joinkeys-tebow-temps
## default + no flow-feats
#bash run_default.sh mse 0 0 0 1 0 10 0 1 $HLS 2 5433 queries/job-joinkeys-tebow-temps

#bash run_default.sh mse 1 0 0 1 0 10 0 1 $HLS 2 5433 queries/job-joinkeys-tebow-temps

#bash run_default.sh mse 1 2 2 0 1 20 0 10 128 2 5431

#bash run_default.sh mse 1 2 2 0 1 20 0 10 32 1
#bash run_default.sh mse 1 2 2 0 1 20 0 10 64 1

#bash run_default.sh mse 1 2 0 0 1 20 0 10 32 1
#bash run_default.sh mse 1 2 0 0 1 20 0 10 32 1
#bash run_default.sh mse 1 2 0 0 1 20 0 10 64 1

#bash run_default.sh flowloss 1 2 2 0 1
#bash run_default.sh mse 0 0 0 1 0
#bash run_default.sh mse 1 0 0 0 1

#bash run_default.sh flowloss 1 0 0 0 0 10
#bash run_default.sh mse 0 0 0 0 0

# no dropout, no early stopping
#bash run_default.sh mse 1 0 2 0 1 20 1

# no early stopping
#bash run_default.sh mse 1 2 0 0 1 20 1

# flow-loss w/o dropout
#bash run_default.sh flowloss 1 0 2 0 1 10 1


#bash run_all_diff.sh 1
#bash run_all_diff.sh 0
#bash run_all_diff.sh 0 embeddings1.pkl

#SWA=none
#BINS=30
#BINS2=10
#BITMAP=0
#FLNORM=1
#EPOCHS=30
#EVALE=2
#QDIR=queries/imdb-unique-plans
##TRAIN_TMPS=1a
##TEST_TMPS=2a,2b
#TRAIN_TMPS=2b
#TEST_TMPS=2a
#DROPOUTKIND=0
##EMBS=(none embeddings/sampled_data_new30.pkl)
#EMBS=(none)
#DECAY=0.0
#FEAT_ALIAS_SEP=1
##REPS=(1)
##REPS=(1 1 1 1 1 1 1 1 1 1)
#REPS=(1 1 1 1 1)

#for i in "${!REPS[@]}";
#do
  #for ei in "${!EMBS[@]}";
  #do
  #CMD="time python3 main.py --algs mscn \
  #--query_dir $QDIR \
  #--eval_epoch $EVALE \
  #--loss_func_name mse \
  #--onehot_dropout $DROPOUTKIND \
  #--lr 0.0001 \
  #--normalize_flow_loss $FLNORM \
  #--embedding_fn ${EMBS[$ei]} \
  #--weight_decay $DECAY \
  #--embedding_pooling sum \
  #--max_discrete_featurizing_buckets $BINS \
  #--sample_bitmap $BITMAP \
  #-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  #--use_wandb 1 --load_padded_mscn_feats 1 \
  #--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  #--eval_fns qerr,ppc
  #--feat_onlyseen_preds 1 \
  #--feat_separate_alias $FEAT_ALIAS_SEP \
  #--table_features 1 --join_features onehot \
  #--set_column_feature onehot --feat_mcvs 0 \
  #--implied_pred_features 0"
  #echo $CMD
  #eval $CMD
  #done
#done

##for i in "${!REPS[@]}";
##do
  ##for ei in "${!EMBS[@]}";
  ##do
  ##CMD="time python3 main.py --algs mscn \
  ##--query_dir $QDIR \
  ##--eval_epoch $EVALE \
  ##--loss_func_name mse \
  ##--onehot_dropout $DROPOUTKIND \
  ##--lr 0.0001 \
  ##--normalize_flow_loss $FLNORM \
  ##--embedding_fn ${EMBS[$ei]} \
  ##--weight_decay $DECAY \
  ##--embedding_pooling sum \
  ##--max_discrete_featurizing_buckets $BINS2 \
  ##--sample_bitmap $BITMAP \
  ##-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  ##--use_wandb 1 --load_padded_mscn_feats 1 \
  ##--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  ##--eval_fns qerr,ppc
  ##--feat_onlyseen_preds 1 \
  ##--feat_separate_alias $FEAT_ALIAS_SEP \
  ##--table_features 1 --join_features onehot \
  ##--set_column_feature onehot --feat_mcvs 0 \
  ##--implied_pred_features 0"
  ##echo $CMD
  ##eval $CMD
  ##done
##done


##for i in "${!REPS[@]}";
##do
  ##for ei in "${!EMBS[@]}";
  ##do
  ##CMD="time python3 main.py --algs mscn \
  ##--query_dir $QDIR \
  ##--loss_func_name flowloss \
  ##--lr 0.00001 \
  ##--normalize_flow_loss $FLNORM \
  ##--embedding_fn ${EMBS[$ei]} \
  ##--weight_decay $DECAY \
  ##--embedding_pooling sum \
  ##--max_discrete_featurizing_buckets $BINS \
  ##-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  ##--use_wandb 1 --load_padded_mscn_feats 1 \
  ##--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  ##--eval_epoch 100 --feat_onlyseen_preds 1 \
  ##--feat_separate_alias $FEAT_ALIAS_SEP \
  ##--table_features 0 --join_features stats \
  ##--set_column_feature stats --feat_mcvs 0 \
  ##--implied_pred_features 0"
  ##echo $CMD
  ##eval $CMD
  ##done
##done

##for i in "${!REPS[@]}";
##do
  ##for ei in "${!EMBS[@]}";
  ##do
  ##CMD="time python3 main.py --algs mscn \
  ##--query_dir $QDIR \
  ##--embedding_fn ${EMBS[$ei]} \
  ##--weight_decay $DECAY \
  ##--embedding_pooling sum \
  ##--max_discrete_featurizing_buckets $BINS \
  ##-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  ##--use_wandb 1 --load_padded_mscn_feats 1 \
  ##--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  ##--eval_epoch 100 --feat_onlyseen_preds 1 \
  ##--feat_separate_alias $FEAT_ALIAS_SEP \
  ##--table_features 0 --join_features 0 \
  ##--set_column_feature 0 --feat_mcvs 0 \
  ##--implied_pred_features 0"
  ##echo $CMD
  ##eval $CMD
  ##done
##done

##for i in "${!REPS[@]}";
##do
  ##for ei in "${!EMBS[@]}";
  ##do
  ##CMD="time python3 main.py --algs mscn \
  ##--query_dir $QDIR \
  ##--embedding_fn ${EMBS[$ei]} \
  ##--weight_decay $DECAY \
  ##--embedding_pooling sum \
  ##--max_discrete_featurizing_buckets $BINS \
  ##-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  ##--use_wandb 1 --load_padded_mscn_feats 1 \
  ##--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  ##--eval_epoch 100 --feat_onlyseen_preds 1 \
  ##--feat_separate_alias $FEAT_ALIAS_SEP \
  ##--table_features 0 --join_features stats \
  ##--set_column_feature stats --feat_mcvs 0 \
  ##--implied_pred_features 0"
  ##echo $CMD
  ##eval $CMD
  ##done
##done

##for i in "${!REPS[@]}";
##do
  ##for ei in "${!EMBS[@]}";
  ##do
  ##CMD="time python3 main.py --algs mscn \
  ##--query_dir $QDIR \
  ##--embedding_fn ${EMBS[$ei]} \
  ##--weight_decay $DECAY \
  ##--embedding_pooling sum \
  ##--max_discrete_featurizing_buckets $BINS \
  ##-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  ##--use_wandb 1 --load_padded_mscn_feats 1 \
  ##--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS \
  ##--no_regex 1 \
  ##--eval_epoch 100 --feat_onlyseen_preds 1 \
  ##--feat_separate_alias $FEAT_ALIAS_SEP \
  ##--set_column_feature onehot \
  ##--table_features 1 --join_features onehot"
  ##echo $CMD
  ##eval $CMD
  ##done
##done

##for i in "${!REPS[@]}";
##do
  ##for ei in "${!EMBS[@]}";
  ##do
  ##CMD="time python3 main.py --algs mscn \
  ##--query_dir $QDIR \
  ##--embedding_fn ${EMBS[$ei]} \
  ##--weight_decay $DECAY \
  ##--embedding_pooling sum \
  ##--max_discrete_featurizing_buckets $BINS \
  ##-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  ##--use_wandb 1 --load_padded_mscn_feats 1 \
  ##--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  ##--eval_epoch 100 --feat_onlyseen_preds 1 \
  ##--feat_separate_alias $FEAT_ALIAS_SEP \
  ##--table_features 0 --join_features onehot"
  ##echo $CMD
  ##eval $CMD
  ##done
##done

##for i in "${!REPS[@]}";
##do
  ##for ei in "${!EMBS[@]}";
  ##do
  ##CMD="time python3 main.py --algs mscn \
  ##--query_dir $QDIR \
  ##--embedding_fn ${EMBS[$ei]} \
  ##--weight_decay $DECAY \
  ##--embedding_pooling sum \
  ##--max_discrete_featurizing_buckets $BINS \
  ##-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  ##--use_wandb 1 --load_padded_mscn_feats 1 \
  ##--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  ##--eval_epoch 100 --feat_onlyseen_preds 1 \
  ##--feat_separate_alias $FEAT_ALIAS_SEP \
  ##--table_features 0 --join_features onehot \
  ##--set_column_feature onehot --feat_mcvs 1 --implied_pred_features 1"
  ##echo $CMD
  ##eval $CMD
  ##done
##done

##for i in "${!REPS[@]}";
##do
  ##for ei in "${!EMBS[@]}";
  ##do
  ##CMD="time python3 main.py --algs mscn \
  ##--query_dir $QDIR \
  ##--embedding_fn ${EMBS[$ei]} \
  ##--weight_decay $DECAY \
  ##--embedding_pooling sum \
  ##--max_discrete_featurizing_buckets $BINS \
  ##-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  ##--use_wandb 1 --load_padded_mscn_feats 1 \
  ##--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  ##--eval_epoch 100 --feat_onlyseen_preds 1 \
  ##--feat_separate_alias $FEAT_ALIAS_SEP \
  ##--table_features 0 --join_features stats \
  ##--set_column_feature stats --feat_mcvs 0"
  ##echo $CMD
  ##eval $CMD
  ##done
##done

##for i in "${!REPS[@]}";
##do
  ##for ei in "${!EMBS[@]}";
  ##do
  ##CMD="time python3 main.py --algs mscn \
  ##--query_dir $QDIR \
  ##--embedding_fn ${EMBS[$ei]} \
  ##--weight_decay $DECAY \
  ##--embedding_pooling sum \
  ##--max_discrete_featurizing_buckets $BINS \
  ##-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  ##--use_wandb 1 --load_padded_mscn_feats 1 \
  ##--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  ##--eval_epoch 100 --feat_onlyseen_preds 1 \
  ##--feat_separate_alias $FEAT_ALIAS_SEP \
  ##--table_features 1 --join_features stats"
  ##echo $CMD
  ##eval $CMD
  ##done
##done

##for i in "${!REPS[@]}";
##do
  ##for ei in "${!EMBS[@]}";
  ##do
  ##CMD="time python3 main.py --algs mscn \
  ##--query_dir $QDIR \
  ##--embedding_fn ${EMBS[$ei]} \
  ##--weight_decay $DECAY \
  ##--embedding_pooling sum \
  ##--max_discrete_featurizing_buckets $BINS \
  ##-n -1 --max_epochs $EPOCHS --train_test_split_kind custom \
  ##--use_wandb 1 --load_padded_mscn_feats 1 \
  ##--train_tmps $TRAIN_TMPS --test_tmps $TEST_TMPS --no_regex 1 \
  ##--eval_epoch 100 --feat_onlyseen_preds 1 \
  ##--feat_separate_alias $FEAT_ALIAS_SEP \
  ##--table_features 0 --join_features stats \
  ##--set_column_feature stats"
  ##echo $CMD
  ##eval $CMD
  ##done
##done

