REPS=(1 1 1)
EVALE=2
HLS=256
OTHERHL=256
#EVALQDIR=queries/job-joinkeys

BITMAPMASK=1
RANDOMBITMAP=1
RANDOMBITMAP2=0
#QDIR=queries/joblight_train
QDIR=queries/joblight_train_1950
N=-1
#QDIR=queries/simple_imdb_train
#N=30000

#EVALQDIR=queries/job-joinkeys
EVALQDIR=queries/job2
#EVALQDIR=queries/job2,queries/imdb-regex,queries/imdb-noregex
#EVALQDIR2=queries/imdb-regex,queries/imdb-noregex
EPOCHS=80
EVALQ_EVALS=qerr,ppc
#EVALQ_EVALS=qerr

CLAMP=1
MAXY=1
DECAY=0.0
REGLOSS=0
REGDECAY=0.001
LR=0.0001

for r in "${!REPS[@]}";
do
  CMD="python3 main.py --algs mscn --query_dir $QDIR \
  -n $N --regen_featstats 0 --save_featstats 0 --lr $LR \
  --bitmap_onehotmask $BITMAPMASK \
  --random_bitmap_idx $RANDOMBITMAP \
  --evalq_eval_fns $EVALQ_EVALS \
  --weight_decay $DECAY \
  --reg_loss $REGLOSS --onehot_reg_decay $REGDECAY \
  --eval_query_dir $EVALQDIR \
  --val_size 0.0 --test_size 0.0 --eval_epoch $EVALE --eval_fns qerr,ppc \
  --sample_bitmap 0 \
  --onehot_dropout 2 --join_bitmap 1 --feat_clamp_timeouts $CLAMP \
  --onehot_mask_truep 0.8 --max_discrete_featurizing_buckets 1 \
  --hidden_layer_size $HLS --other_hid_units $OTHERHL \
  --max_epochs $EPOCHS \
  --feat_onlyseen_maxy $MAXY"
  echo $CMD
  eval $CMD

  CMD="python3 main.py --algs mscn --query_dir $QDIR \
  -n $N --regen_featstats 0 --save_featstats 0 --lr $LR \
  --bitmap_onehotmask $BITMAPMASK \
  --random_bitmap_idx $RANDOMBITMAP2 \
  --evalq_eval_fns $EVALQ_EVALS \
  --weight_decay $DECAY \
  --reg_loss $REGLOSS --onehot_reg_decay $REGDECAY \
  --eval_query_dir $EVALQDIR \
  --val_size 0.0 --test_size 0.0 --eval_epoch $EVALE --eval_fns qerr,ppc \
  --sample_bitmap 0 \
  --onehot_dropout 2 --join_bitmap 1 --feat_clamp_timeouts $CLAMP \
  --onehot_mask_truep 0.8 --max_discrete_featurizing_buckets 1 \
  --hidden_layer_size $HLS --other_hid_units $OTHERHL \
  --max_epochs $EPOCHS \
  --feat_onlyseen_maxy $MAXY"
  echo $CMD
  eval $CMD

  #CMD="python3 main.py --algs mscn --query_dir $QDIR \
  #-n $N --regen_featstats 0 --save_featstats 0 --lr $LR \
  #--random_bitmap_idx $RANDOMBITMAP \
  #--weight_decay $DECAY \
  #--evalq_eval_fns $EVALQ_EVALS \
  #--reg_loss $REGLOSS --onehot_reg_decay $REGDECAY \
  #--bitmap_onehotmask 0 \
  #--eval_query_dir $EVALQDIR \
  #--val_size 0.0 --test_size 0.0 --eval_epoch $EVALE --eval_fns qerr,ppc \
  #--onehot_dropout 0 --join_bitmap 0 --feat_clamp_timeouts $CLAMP \
  #--sample_bitmap 1 \
  #--onehot_mask_truep 0.0 --max_discrete_featurizing_buckets 1 \
  #--hidden_layer_size $HLS --other_hid_units $OTHERHL \
  #--max_epochs $EPOCHS \
  #--feat_onlyseen_maxy $MAXY"
  #echo $CMD
  #eval $CMD
done
