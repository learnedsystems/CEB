TEMPLATES=(9 8 7 6 5 4 3 2 1)
TESTTMP=10
REPS=(1 1 1 1 1)
USEWANDB=1
EPOCHS=3000

QDIR=queries/synth_gaussian

for rep in "${REPS[@]}";
do
  for tmp in "${TEMPLATES[@]}";
  do
  CMD="python3 main.py --algs mscn --query_dir $QDIR \
  --db_name synth1 --eval_fns qerr --max_epochs $EPOCHS --lr 0.001 \
  --val_size 0.1 --max_discrete_featurizing_buckets 1 \
  --train_test_split custom --train_tmps ${tmp} --test_tmps $TESTTMP \
  --wandb_tags gaussian \
  --val_size 0.2 --onehot_dropout 2 --onehot_mask 0.8 \
  --hidden_layer_size 32 --heuristic_features 1 \
  --join_features 0 --table_features 0 \
  --use_wandb $USEWANDB \
  --flow_features 1 --pred_features 1"
  echo $CMD
  eval $CMD

  CMD="python3 main.py --algs mscn --query_dir $QDIR \
  --db_name synth1 --eval_fns qerr --max_epochs $EPOCHS --lr 0.001 \
  --val_size 0.1 --max_discrete_featurizing_buckets 1 \
  --train_test_split custom --train_tmps ${tmp} --test_tmps $TESTTMP \
  --wandb_tags gaussian \
  --val_size 0.2 --onehot_dropout 0 --onehot_mask 0.8 \
  --hidden_layer_size 32 --heuristic_features 1 \
  --join_features 0 --table_features 0 \
  --use_wandb $USEWANDB \
  --flow_features 0 --pred_features 1"
  echo $CMD
  eval $CMD

  CMD="python3 main.py --algs mscn --query_dir $QDIR \
  --db_name synth1 --eval_fns qerr --max_epochs $EPOCHS --lr 0.001 \
  --val_size 0.1 --max_discrete_featurizing_buckets 1 \
  --train_test_split custom --train_tmps ${tmp} --test_tmps $TESTTMP \
  --val_size 0.2 --onehot_dropout 2 --onehot_mask 0.7 \
  --wandb_tags gaussian \
  --hidden_layer_size 32 --heuristic_features 1 \
  --join_features 0 --table_features 0 \
  --use_wandb $USEWANDB \
  --flow_features 1 --pred_features 1"
  echo $CMD
  eval $CMD

  done
done

