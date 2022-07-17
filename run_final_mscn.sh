REPS=(1 1)
EVALE=1000
HLS=256
OTHERHL=256
#EVALQDIR=queries/job-joinkeys

BITMAPMASK=1
#RANDOMBITMAP=0
QDIR=queries/joblight_train
QDIRS2=(queries/joblight_train_1950 queries/joblight_train_1980)
#QDIRS2=(queries/joblight_train_1980 queries/joblight_train_1950)

N=-1
EVALQDIR=queries/imdb-unique-plans,queries/job-joinkeys

EPOCHS=40
EVALQ_EVALS=qerr,ppc

CLAMP=1
MAXY=1
DECAY=0.0
REGLOSS=0
REGDECAY=0.01
LR=0.0001

#for r in "${!REPS[@]}";
#do
  #CMD="python3 main.py --algs mscn --query_dir queries/simple_imdb_train \
  #-n $N --regen_featstats 0 --save_featstats 0 --lr $LR \
  #--result_dir final_results \
  #--bitmap_onehotmask $BITMAPMASK \
  #--random_bitmap_idx 0 \
  #--evalq_eval_fns $EVALQ_EVALS \
  #--weight_decay $DECAY \
  #--reg_loss 1 --onehot_reg_decay $REGDECAY \
  #--eval_query_dir $EVALQDIR \
  #--val_size 0.0 --test_size 0.0 --eval_epoch $EVALE --eval_fns qerr,ppc \
  #--sample_bitmap 0 \
  #--onehot_dropout 2 --join_bitmap 1 --feat_clamp_timeouts $CLAMP \
  #--onehot_mask_truep 0.8 --max_discrete_featurizing_buckets 1 \
  #--hidden_layer_size $HLS --other_hid_units $OTHERHL \
  #--max_epochs $EPOCHS \
  #--feat_onlyseen_maxy $MAXY"
  #echo $CMD
  #eval $CMD

  #CMD="python3 main.py --algs mscn --query_dir queries/joblight_train \
  #-n $N --regen_featstats 0 --save_featstats 0 --lr $LR \
  #--result_dir final_results \
  #--bitmap_onehotmask $BITMAPMASK \
  #--random_bitmap_idx 0 \
  #--evalq_eval_fns $EVALQ_EVALS \
  #--weight_decay $DECAY \
  #--reg_loss $REGLOSS --onehot_reg_decay $REGDECAY \
  #--eval_query_dir $EVALQDIR \
  #--val_size 0.0 --test_size 0.0 --eval_epoch $EVALE --eval_fns qerr,ppc \
  #--sample_bitmap 0 \
  #--onehot_dropout 2 --join_bitmap 1 --feat_clamp_timeouts $CLAMP \
  #--onehot_mask_truep 0.8 --max_discrete_featurizing_buckets 1 \
  #--hidden_layer_size $HLS --other_hid_units $OTHERHL \
  #--max_epochs $EPOCHS \
  #--feat_onlyseen_maxy $MAXY"
  #echo $CMD
  #eval $CMD

  #CMD="python3 main.py --algs mscn --query_dir queries/imdb_train_mod \
  #-n $N --regen_featstats 0 --save_featstats 0 --lr $LR \
  #--result_dir final_results \
  #--bitmap_onehotmask $BITMAPMASK \
  #--random_bitmap_idx 0 \
  #--evalq_eval_fns $EVALQ_EVALS \
  #--weight_decay $DECAY \
  #--reg_loss $REGLOSS --onehot_reg_decay $REGDECAY \
  #--eval_query_dir $EVALQDIR \
  #--val_size 0.0 --test_size 0.0 --eval_epoch $EVALE --eval_fns qerr,ppc \
  #--sample_bitmap 0 \
  #--onehot_dropout 2 --join_bitmap 1 --feat_clamp_timeouts $CLAMP \
  #--onehot_mask_truep 0.8 --max_discrete_featurizing_buckets 1 \
  #--hidden_layer_size $HLS --other_hid_units $OTHERHL \
  #--max_epochs $EPOCHS \
  #--feat_onlyseen_maxy $MAXY"
  #echo $CMD
  #eval $CMD
#done

for r in "${!REPS[@]}";
  do
 for qi in "${!QDIRS2[@]}";
  do
  CMD="python3 main.py --algs mscn \
  -n $N --regen_featstats 0 --save_featstats 0 --lr $LR \
  --result_dir final_data_update_results \
  --bitmap_onehotmask 0 \
  --random_bitmap_idx 0 \
  --query_dir ${QDIRS2[$qi]} \
  --evalq_eval_fns $EVALQ_EVALS \
  --weight_decay $DECAY \
  --reg_loss 0 --onehot_reg_decay $REGDECAY \
  --eval_query_dir $EVALQDIR \
  --val_size 0.0 --test_size 0.0 --eval_epoch $EVALE --eval_fns qerr,ppc \
  --sample_bitmap 1 \
  --onehot_dropout 0 --join_bitmap 0 --feat_clamp_timeouts $CLAMP \
  --onehot_mask_truep 0.8 --max_discrete_featurizing_buckets 1 \
  --hidden_layer_size $HLS --other_hid_units $OTHERHL \
  --max_epochs $EPOCHS \
  --feat_onlyseen_maxy 0"
  echo $CMD
  eval $CMD

  #CMD="python3 main.py --algs mscn \
  #-n $N --regen_featstats 0 --save_featstats 0 --lr $LR \
  #--result_dir final_results \
  #--bitmap_onehotmask $BITMAPMASK \
  #--random_bitmap_idx 1 \
  #--query_dir ${QDIRS2[$qi]} \
  #--evalq_eval_fns $EVALQ_EVALS \
  #--weight_decay $DECAY \
  #--reg_loss $REGLOSS --onehot_reg_decay $REGDECAY \
  #--eval_query_dir $EVALQDIR \
  #--val_size 0.0 --test_size 0.0 --eval_epoch $EVALE --eval_fns qerr,ppc \
  #--sample_bitmap 0 \
  #--onehot_dropout 2 --join_bitmap 1 --feat_clamp_timeouts $CLAMP \
  #--onehot_mask_truep 0.8 --max_discrete_featurizing_buckets 1 \
  #--hidden_layer_size $HLS --other_hid_units $OTHERHL \
  #--max_epochs $EPOCHS \
  #--feat_onlyseen_maxy 0"
  #echo $CMD
  #eval $CMD

  #CMD="python3 main.py --algs mscn \
  #-n $N --regen_featstats 0 --save_featstats 0 --lr $LR \
  #--result_dir final_results \
  #--bitmap_onehotmask $BITMAPMASK \
  #--random_bitmap_idx 0 \
  #--query_dir ${QDIRS2[$qi]} \
  #--random_bitmap_idx 0 \
  #--evalq_eval_fns $EVALQ_EVALS \
  #--weight_decay $DECAY \
  #--reg_loss $REGLOSS --onehot_reg_decay $REGDECAY \
  #--eval_query_dir $EVALQDIR \
  #--val_size 0.0 --test_size 0.0 --eval_epoch $EVALE --eval_fns qerr,ppc \
  #--sample_bitmap 0 \
  #--onehot_dropout 2 --join_bitmap 1 --feat_clamp_timeouts $CLAMP \
  #--onehot_mask_truep 0.8 --max_discrete_featurizing_buckets 1 \
  #--hidden_layer_size $HLS --other_hid_units $OTHERHL \
  #--max_epochs $EPOCHS \
  #--feat_onlyseen_maxy 0"
  #echo $CMD
  #eval $CMD
done
done
