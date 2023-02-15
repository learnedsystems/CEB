
REPS=(1 1 1 1 1 1 1 1 1 1)
FF=1
DROP=2
EPOCHS=16000
TRAIN=all
TEST=2c
HLS=8
QDIRS=(queries/new_synth_2d_gaussian1K queries/new_synth_2d_gaussian10K queries/new_synth_2d_gaussian100K queries/new_synth_2d_gaussian1M)
#EQDIR=(queries/new_synth_2d_gaussian1M
EQDIR=queries/new_synth_2d_gaussian1K,queries/new_synth_2d_gaussian10K,queries/new_synth_2d_gaussian100K,queries/new_synth_2d_gaussian1M

for qdir in "${QDIRS[@]}";
  do
    for rep in "${REPS[@]}";
    do
      echo ${qdir}
      CMD="""python3 main.py --algs mscn \
      --query_dir ${qdir} --eval_fns qerr \
      --wandb_tags data_update_final2 \
      --train_test_split query --val_size 0.2 --test_size 0.0 \
      --eval_query_dir $EQDIR \
      --query_templates $TRAIN --eval_templates $TRAIN \
      --max_epochs $EPOCHS --lr 0.0001 --hidden_layer_size $HLS \
      --db_name synth1 --table_features 0 --pred_features 1 \
      --join_features 0 --save_featstats 0 --feat_onlyseen_maxy 0 \
      --onehot_dropout $DROP --onehot_mask_truep 0.8 --flow_features $FF \
      -n -1 --heuristic_features 1 \
      --use_wandb 1 --eval_epoch 50"""
      echo $CMD
      eval $CMD
  done
done

