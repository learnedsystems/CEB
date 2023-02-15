REPS=(1 1 1 1 1)
FF=0
DROP=2
EPOCHS=10000
TRAIN=(2a 2b 2c)
TEST=(2c 2b)
QDIR=queries/new_synth_2d_gaussian10K

for ctrain in "${TRAIN[@]}";
do
  for ctest in "${TEST[@]}";
  do
    for rep in "${REPS[@]}";
    do
      if [ ${ctrain} == ${ctest} ];
      then
        continue
      fi
      echo ${ctrain}
      echo ${ctest}
      CMD="""python3 main.py --algs mscn \
      --query_dir $QDIR --eval_fns qerr \
      --wandb_tags gaussian_tests \
      --train_test_split custom --train_tmps ${ctrain} --test_tmps ${ctest} \
      --max_epochs $EPOCHS --lr 0.0001 --hidden_layer_size 8 \
      --db_name synth1 --table_features 0 --pred_features 1 \
      --join_features 0 --save_featstats 0 --feat_onlyseen_maxy 0 \
      --onehot_dropout $DROP --onehot_mask_truep 0.8 --flow_features $FF \
      -n -1 --heuristic_features 1 \
      --use_wandb 1 --eval_epoch 50"""
      echo $CMD
      eval $CMD
    done
  done
done

