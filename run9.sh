ALG=mscn
NORMFL=1
LRS=(0.0001 0.00001)
#DECAYS=(0.1 1.0 0.0)
DECAYS=(0.0)
#DECAYS=(1.0)
REPS=(1 1)
SEED=7

## copied over

# changed since paper
HLS=(128 256)
FLOW_FEATS=0
SEPS=(0 1)
FEAT_ONLYSEEN=1
MAX_EPOCHS=10

TABLE_FEATURES=1
JOIN_FEATURES=onehot
COLUMN_FEATURES=onehot
FEAT_MCV=0
ONEHOT_DROPOUT=0
LOAD_PADDED=1
WANDB_TAGS=flow_default
LOSS_FUNC=flowloss
BUCKETS=10

EMBEDDING_FN=none

QDIR=queries/imdb-unique-plans
#QDIR=queries/imdb
NO_REGEX=0
N=-1

#SEEDS=(1 2 3 4 5 6 7 8 9 10)
#SEEDS=(7 1 2 3 4 5 6 8 9 10)
#SEEDS=(4 5 6 7 8 9 10)
EVAL_EPOCH=200

EVAL_FNS=qerr,ppc,plancost

RES_DIR=results/

LOAD_QUERY_TOGTHER=1

for r in "${!REPS[@]}";
do
    for lr in "${!LRS[@]}";
    do
    for hls in "${!HLS[@]}";
    do
    for j in "${!DECAYS[@]}";
    do
      for s in "${!SEPS[@]}";
      do
      #bash run_all_diff_flow.sh $ALG ${DECAYS[$j]} $NORMFL $LR
      CMD="time python3 main.py --algs $ALG \
       -n $N \
       --onehot_dropout $ONEHOT_DROPOUT \
       --no_regex_templates $NO_REGEX \
       --normalize_flow_loss $NORMFL \
       --feat_onlyseen_preds $FEAT_ONLYSEEN \
       --feat_separate_alias ${SEPS[$s]} \
       --query_dir $QDIR \
       --hidden_layer_size ${HLS[$hls]} \
       --train_test_split_kind template \
       --diff_templates_seed $SEED \
       --max_discrete_featurizing_buckets $BUCKETS \
       --weight_decay ${DECAYS[$j]} \
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
       --lr ${LRS[$lr]}"
        echo $CMD
        eval $CMD
      done
      done
    done
    done
done
