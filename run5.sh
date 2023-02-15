
REPS=(1 1 1 1 1 1 1 1)
FF=0
DROP=0
EPOCHS=8000
EPOCHS2=4000
TRAIN=2a,2b,2c
#TEST=2f,2e
QDIR=queries/new_synth_2d_gaussian100K
EQDIR=queries/new_synth_2d_gaussian100K_neg5,queries/new_synth_2d_gaussian100K_0,queries/new_synth_2d_gaussian100K_5

for rep in "${REPS[@]}";
do
  #CMD="""python3 main.py --algs mscn \
  #--query_dir $QDIR --eval_fns qerr \
  #--wandb_tags gaussian_tests \
  #--train_test_split custom --train_tmps $TRAIN --test_tmps $TEST \
  #--max_epochs $EPOCHS --lr 0.0001 --hidden_layer_size 8 \
  #--db_name synth1 --table_features 0 --pred_features 1 \
  #--join_features 0 --save_featstats 0 --feat_onlyseen_maxy 0 \
  #--onehot_dropout $DROP --onehot_mask_truep 0.8 --flow_features $FF \
  #-n -1 --heuristic_features 1 \
  #--use_wandb 1 --eval_epoch 50"""
  #echo $CMD
  #eval $CMD

  #CMD="""python3 main.py --algs mscn \
  #--query_dir $QDIR --eval_fns qerr \
  #--wandb_tags gaussian_tests \
  #--train_test_split custom --train_tmps $TRAIN --test_tmps $TEST \
  #--max_epochs $EPOCHS --lr 0.0001 --hidden_layer_size 32 \
  #--db_name synth1 --table_features 0 --pred_features 1 \
  #--join_features 0 --save_featstats 0 --feat_onlyseen_maxy 0 \
  #--onehot_dropout $DROP --onehot_mask_truep 0.8 --flow_features $FF \
  #-n -1 --heuristic_features 1 \
  #--use_wandb 1 --eval_epoch 50"""
  #echo $CMD
  #eval $CMD

  CMD="""python3 main.py --algs mscn \
  --query_dir $QDIR --eval_fns qerr \
  --wandb_tags gaussian_update \
  --train_test_split query --val_size 0.2 --test_size 0.0 \
  --eval_query_dir $EQDIR \
  --query_templates $TRAIN --eval_templates $TRAIN \
  --max_epochs $EPOCHS --lr 0.0001 --hidden_layer_size 8 \
  --db_name synth1 --table_features 0 --pred_features 1 \
  --join_features 0 --save_featstats 0 --feat_onlyseen_maxy 0 \
  --onehot_dropout $DROP --onehot_mask_truep 0.8 --flow_features $FF \
  -n -1 --heuristic_features 1 \
  --use_wandb 1 --eval_epoch 50"""
  echo $CMD
  eval $CMD

  #CMD="""python3 main.py --algs mscn \
  #--query_dir $QDIR --eval_fns qerr \
  #--wandb_tags gaussian_update \
  #--train_test_split query --val_size 0.2 --test_size 0.0 \
  #--eval_query_dir $EQDIR \
  #--query_templates $TRAIN --eval_templates $TRAIN \
  #--max_epochs $EPOCHS2 --lr 0.0001 --hidden_layer_size 8 \
  #--db_name synth1 --table_features 0 --pred_features 1 \
  #--join_features 0 --save_featstats 0 --feat_onlyseen_maxy 0 \
  #--onehot_dropout $DROP --onehot_mask_truep 0.8 --flow_features $FF \
  #-n -1 --heuristic_features 1 \
  #--use_wandb 1 --eval_epoch 50"""
  #echo $CMD
  #eval $CMD
done

