
REPS=(1 1 1)
EVALE=2
#EVALQDIR=queries/job2
EVALQDIR=queries/job2,queries/imdb-regex,queries/imdb-noregex
#EVALQDIR=queries/job-joinkeys,queries/imdb-unique-plans

for r in "${!REPS[@]}";
do
  CMD="python3 main.py --algs mscn --query_dir queries/joblight_train/ \
  --save_test_preds 1 \
  -n -1 --regen_featstats 0 --save_featstats 0 --lr 0.0001 \
  --bitmap_onehotmask 1 \
  --eval_query_dir $EVALQDIR \
  --val_size 0.0 --test_size 0.0 --eval_epoch $EVALE --eval_fns qerr,ppc \
  --onehot_dropout 2 --join_bitmap 1 --feat_clamp_timeouts 1 \
  --onehot_mask_truep 0.8 --max_discrete_featurizing_buckets 1 \
  --max_epochs 80 \
  --feat_onlyseen_maxy 1"
  echo $CMD
  eval $CMD
done
