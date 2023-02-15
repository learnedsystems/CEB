#TEMPLATES=(9 8 7 6 5 4 3 2 1)
#TESTTMP=10
REPS=(1 1 1 1 1)
USEWANDB=1
EPOCHS=100
LR=0.0001

QDIR=queries/synth_gaussian

for rep in "${REPS[@]}";
do
  #for tmp in "${TEMPLATES[@]}";
  #do
  CMD="python3 main.py --algs mscn --query_dir queries/simple_ergast_train \
  -n -1 --eval_query_dir queries/ergast --max_epochs $EPOCHS --eval_epoch 5 \
  --onehot_dropout 2 --join_bitmap 1 --sample_bitmap 0 \
  --use_wandb 1 --val_size 0.0 --test_size 0.0 \
  --lr $LR --hidden_layer_size 512"
  echo $CMD
  eval $CMD
  #done
  CMD="python3 main.py --algs mscn --query_dir queries/simple_ergast_train \
  -n -1 --eval_query_dir queries/ergast --max_epochs $EPOCHS --eval_epoch 5 \
  --onehot_dropout 0 --join_bitmap 0 --sample_bitmap 1 \
  --use_wandb 1 --val_size 0.0 --test_size 0.0 \
  --lr $LR --hidden_layer_size 512"
  echo $CMD
  eval $CMD
done
