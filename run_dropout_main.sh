ONEHOT_DROP_PROB=0.8
TABLE_FEATURES=1
JOIN_FEATURES=onehot
COLUMN_FEATURES=onehot
ONEHOT_DROPOUT=2
BUCKETS=10
LIKEBUCKETS=1
MAX_EPOCHS=20
ESTOP=2
VAL=0.2
FLOW_FEATS=1
BITMAP=1
PORT=5433

MAXY=1
CLAMPT=1

SUBPLAN_LEVELS=0
DECAY=0.0
SEP=0
LOAD_PADDED=1

JOB=1
JOBM=0

LOSS_FUNC=mse
ONEHOT_REG=0
EVAL_EPOCH=200
MASK_UNSEEN=0

SWA=none
SWA_LR=0.001

ALG=mscn

LR=0.0001

FEAT_MCV=0
WANDB_TAGS=dropout
FEAT_ONLYSEEN=1
#ALG=mscn

## keep fixed
EMBEDDING_FN=none
QDIR=queries/imdb-unique-plans
#QDIR=queries/imdb
NO_REGEX=0
N=-1

SEEDS=(7 6 4 5 13 1 2 3 8 9 10 11 12 14 15 16 17 18 19 20)
#SEEDS=(7 8 13 14 1 2 3 4 5 6 9 10 11 12 15 16 17 18 19 20)
#SEEDS=(7 13 14 1 2 6 10 17 18 19 20)
#SEEDS=(7 13 14 17 19 20)
REPS=(1 1 1)
#REPS=(1)

EVAL_FNS=qerr,ppc,ppc2
RES_DIR=final_results2/mscn/dropout2

HLS=128
LOAD_QUERY_TOGTHER=0

for r in "${!REPS[@]}";
do
for i in "${!SEEDS[@]}";
  do
  CMD="time python3 main.py --algs $ALG \
   -n $N \
   --port $PORT \
   --onehot_dropout $ONEHOT_DROPOUT \
   --feat_onlyseen_maxy $MAXY \
   --flow_features $FLOW_FEATS \
   --feat_clamp_timeouts $CLAMPT \
   --early_stopping $ESTOP \
   --val_size $VAL \
   --sample_bitmap $BITMAP \
   --onehot_mask_truep $ONEHOT_DROP_PROB \
   --onehot_reg $ONEHOT_REG \
   --eval_on_job $JOB \
   --eval_on_jobm $JOBM \
   --mask_unseen_subplans $MASK_UNSEEN \
   --training_opt $SWA \
   --opt_lr $SWA_LR \
   --no_regex_templates $NO_REGEX \
   --feat_onlyseen_preds $FEAT_ONLYSEEN \
   --wandb_tags $WANDB_TAGS \
   --subplan_level_outputs $SUBPLAN_LEVELS \
   --feat_separate_alias $SEP \
   --query_dir $QDIR \
   --hidden_layer_size $HLS \
   --train_test_split_kind template \
   --diff_templates_seed ${SEEDS[$i]} \
   --max_discrete_featurizing_buckets $BUCKETS \
   --max_like_featurizing_buckets $LIKEBUCKETS \
   --weight_decay $DECAY \
   --alg $ALG \
   --load_padded_mscn_feats $LOAD_PADDED \
   --load_query_together $LOAD_QUERY_TOGTHER \
   --eval_fns $EVAL_FNS \
   --loss_func $LOSS_FUNC \
   --test_size 0.5 \
   --result_dir $RES_DIR \
   --max_epochs $MAX_EPOCHS \
   --eval_epoch $EVAL_EPOCH \
   --table_features $TABLE_FEATURES \
   --join_features $JOIN_FEATURES \
   --set_column_feature $COLUMN_FEATURES \
   --embedding_fn $EMBEDDING_FN \
   --feat_mcvs $FEAT_MCV \
   --implied_pred_features $FEAT_MCV \
   --optimizer_name adamw \
   --lr $LR"
    echo $CMD
    eval $CMD
  done
done
