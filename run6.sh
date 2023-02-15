
## join bitmap stuff
REPS=(1 1 1 1 1)
FF=0
#DROP=2
BITMAPNUM=1000
EPOCHS=1000
LR=0.001
TRAIN=2
TEST=1
QDIR=queries/synth_joins3
HLS=128
#EQDIR=queries/new_synth_2d_gaussian10K_neg5,queries/new_synth_2d_gaussian10K_0,queries/new_synth_2d_gaussian10K_5

for rep in "${REPS[@]}";
do
  #CMD="""python3 main.py --algs mscn \
  #--query_dir $QDIR --eval_fns qerr \
  #--wandb_tags join_bitmap1 \
  #--sample_bitmap_num $BITMAPNUM \
  #--sample_bitmap_buckets $BITMAPNUM \
  #--train_test_split custom --train_tmps $TRAIN --test_tmps $TEST \
  #--max_epochs $EPOCHS --lr $LR --hidden_layer_size $HLS \
  #--db_name synth1 \
  #--table_features 0 --sample_bitmap 0 --pred_features 0 \
  #--join_features 1 --join_bitmap 1 \
  #--random_bitmap_idx 0 --test_random_bitmap 0 \
  #--save_featstats 0 --feat_onlyseen_maxy 0 \
  #--onehot_dropout 0 --onehot_mask_truep 0.8 --flow_features 0 \
  #-n -1 --heuristic_features 0 \
  #--use_wandb 1 --eval_epoch 100"""
  #echo $CMD
  #eval $CMD

  #CMD="""python3 main.py --algs mscn \
  #--query_dir $QDIR --eval_fns qerr \
  #--wandb_tags join_bitmap1 \
  #--sample_bitmap_num $BITMAPNUM \
  #--sample_bitmap_buckets $BITMAPNUM \
  #--train_test_split custom --train_tmps $TRAIN --test_tmps $TEST \
  #--max_epochs $EPOCHS --lr $LR --hidden_layer_size $HLS \
  #--db_name synth1 \
  #--table_features 1 --sample_bitmap 1 --pred_features 0 \
  #--join_features 0 --join_bitmap 0 \
  #--random_bitmap_idx 0 --test_random_bitmap 0 \
  #--save_featstats 0 --feat_onlyseen_maxy 0 \
  #--onehot_dropout 0 --onehot_mask_truep 0.8 --flow_features 0 \
  #-n -1 --heuristic_features 0 \
  #--use_wandb 1 --eval_epoch 100"""
  #echo $CMD
  #eval $CMD

  #CMD="""python3 main.py --algs mscn \
  #--query_dir $QDIR --eval_fns qerr \
  #--wandb_tags join_bitmap1 \
  #--sample_bitmap_num $BITMAPNUM \
  #--sample_bitmap_buckets $BITMAPNUM \
  #--train_test_split custom --train_tmps $TRAIN --test_tmps $TEST \
  #--max_epochs $EPOCHS --lr $LR --hidden_layer_size $HLS \
  #--db_name synth1 \
  #--table_features 0 --sample_bitmap 0 --pred_features 0 \
  #--join_features 1 --join_bitmap 1 \
  #--random_bitmap_idx 1 --test_random_bitmap 0 \
  #--save_featstats 0 --feat_onlyseen_maxy 0 \
  #--onehot_dropout 0 --onehot_mask_truep 0.8 --flow_features 0 \
  #-n -1 --heuristic_features 0 \
  #--use_wandb 1 --eval_epoch 100"""
  #echo $CMD
  #eval $CMD

  #CMD="""python3 main.py --algs mscn \
  #--query_dir $QDIR --eval_fns qerr \
  #--wandb_tags join_bitmap1 \
  #--sample_bitmap_num $BITMAPNUM \
  #--sample_bitmap_buckets $BITMAPNUM \
  #--train_test_split custom --train_tmps $TRAIN --test_tmps $TEST \
  #--max_epochs $EPOCHS --lr $LR --hidden_layer_size $HLS \
  #--db_name synth1 \
  #--table_features 0 --sample_bitmap 0 --pred_features 0 \
  #--join_features 1 --join_bitmap 1 \
  #--random_bitmap_idx 1 --test_random_bitmap 1 \
  #--save_featstats 0 --feat_onlyseen_maxy 0 \
  #--onehot_dropout 0 --onehot_mask_truep 0.8 --flow_features 0 \
  #-n -1 --heuristic_features 0 \
  #--use_wandb 1 --eval_epoch 100"""
  #echo $CMD
  #eval $CMD

  #CMD="""python3 main.py --algs mscn \
  #--query_dir $QDIR --eval_fns qerr \
  #--wandb_tags join_bitmap1 \
  #--sample_bitmap_num $BITMAPNUM \
  #--sample_bitmap_buckets $BITMAPNUM \
  #--train_test_split custom --train_tmps $TRAIN --test_tmps $TEST \
  #--max_epochs $EPOCHS --lr $LR --hidden_layer_size $HLS \
  #--db_name synth1 \
  #--table_features 0 --sample_bitmap 0 --pred_features 0 \
  #--join_features 1 --join_bitmap 1 \
  #--random_bitmap_idx 0 --test_random_bitmap 1 \
  #--save_featstats 0 --feat_onlyseen_maxy 0 \
  #--onehot_dropout 0 --onehot_mask_truep 0.8 --flow_features 0 \
  #-n -1 --heuristic_features 0 \
  #--use_wandb 1 --eval_epoch 100"""
  #echo $CMD
  #eval $CMD

  CMD="""python3 main.py --algs mscn \
  --query_dir $QDIR --eval_fns qerr \
  --wandb_tags join_bitmap1 \
  --sample_bitmap_num $BITMAPNUM \
  --sample_bitmap_buckets $BITMAPNUM \
  --train_test_split custom --train_tmps $TRAIN --test_tmps $TEST \
  --max_epochs $EPOCHS --lr $LR --hidden_layer_size $HLS \
  --db_name synth1 \
  --table_features 1 --sample_bitmap 1 --pred_features 0 \
  --join_features 0 --join_bitmap 0 \
  --random_bitmap_idx 0 --test_random_bitmap 1 \
  --save_featstats 0 --feat_onlyseen_maxy 0 \
  --onehot_dropout 0 --onehot_mask_truep 0.8 --flow_features 0 \
  -n -1 --heuristic_features 0 \
  --use_wandb 1 --eval_epoch 100"""
  echo $CMD
  eval $CMD

done

