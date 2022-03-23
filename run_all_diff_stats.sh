TABLE_FEATURES=$1
JOIN_FEATURES=$2
COLUMN_FEATURES=$3
ONEHOT_DROPOUT=$4
BUCKETS=$5
MAX_EPOCHS=$6
SUBPLAN_LEVELS=0
ONEHOT_DROP_PROB=$7
LOSS_FUNC=$8
ONEHOT_REG=$9
EVAL_EPOCH=100

MASK_UNSEEN=0

SWA=none
SWA_LR=0.001

DECAY=0.0
SEP=0

ALG=mscn

LR=0.0001

FEAT_MCV=0
LOAD_PADDED=1
WANDB_TAGS=stats
FEAT_ONLYSEEN=1
#ALG=mscn

## keep fixed
EMBEDDING_FN=none
QDIR=queries/imdb-unique-plans
#QDIR=queries/imdb
NO_REGEX=0
N=-1

FLOW_FEATS=0

#SEEDS=(1 2 3 4 5 6 7 8 9 10)
#SEEDS=(7 6 1 2 3 4 5 8 9 10)
#SEEDS=(2 10 7 6 1 3 4 5 8 9 13 14 19 20)

#SEEDS=(7 6 1 2 10)
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
#SEEDS=(4 5 6 7 8 9 10)


EVAL_FNS=qerr,ppc,constraints

RES_DIR=results/

HLS=128
LOAD_QUERY_TOGTHER=0

for i in "${!SEEDS[@]}";
  do
  CMD="time python3 main.py --algs $ALG \
   -n $N \
   --onehot_dropout $ONEHOT_DROPOUT \
   --onehot_mask_truep $ONEHOT_DROP_PROB \
   --onehot_reg $ONEHOT_REG \
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
   --max_like_featurizing_buckets $BUCKETS \
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
