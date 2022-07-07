
REPS=(1 1 1)
EVALE=2

for r in "${!REPS[@]}";
do
  #CMD="python3 main.py --algs mstn --query_dir queries/joblight_train/ \
  #-n -1 --regen_featstats 0 --save_featstats 0 --lr 0.0001 \
  #--bitmap_onehotmask 1 \
  #--eval_query_dir queries/imdb-unique-plans,queries/job-joinkeys \
  #--val_size 0.0 --test_size 0.0 --eval_epoch 100 --eval_fns qerr,ppc \
  #--onehot_dropout 2 --join_bitmap 1 --feat_clamp_timeouts 1 \
  #--onehot_mask_truep 0.8 --max_discrete_featurizing_buckets 1 \
  #--max_epochs 20 \
  #--feat_onlyseen_maxy 1"
  #echo $CMD
  #eval $CMD

  CMD="python3 main.py --algs mstn --query_dir queries/imdb-unique-plans/ \
  -n 100 --regen_featstats 0 --save_featstats 0 --lr 0.0001 \
  --bitmap_onehotmask 1 \
  --eval_query_dir queries/job2 \
  --val_size 0.0 --test_size 0.0 --eval_epoch $EVALE --eval_fns qerr,ppc \
  --onehot_dropout 2 --join_bitmap 1 --feat_clamp_timeouts 1 \
  --onehot_mask_truep 0.8 --max_discrete_featurizing_buckets 1 \
  --max_epochs 20 \
  --feat_onlyseen_maxy 1"
  echo $CMD
  eval $CMD

  CMD="python3 main.py --algs mstn --query_dir queries/joblight_train/ \
  -n -1 --regen_featstats 0 --save_featstats 0 --lr 0.0001 \
  --bitmap_onehotmask 1 \
  --eval_query_dir queries/imdb-unique-plans,queries/job2 \
  --val_size 0.0 --test_size 0.0 --eval_epoch $EVALE --eval_fns qerr,ppc \
  --onehot_dropout 2 --join_bitmap 1 --feat_clamp_timeouts 1 \
  --onehot_mask_truep 0.8 --max_discrete_featurizing_buckets 1 \
  --max_epochs 20 \
  --feat_onlyseen_maxy 1"
  echo $CMD
  eval $CMD

done
