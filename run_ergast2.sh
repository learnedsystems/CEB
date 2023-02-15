#TEMPLATES=(9 8 7 6 5 4 3 2 1)
#TESTTMP=10
REPS=(1 1 1 1 1)
USEWANDB=1
EPOCHS=50

QDIR=queries/synth_gaussian

for rep in "${REPS[@]}";
do
  #for tmp in "${TEMPLATES[@]}";
  #do
  CMD="python3 main.py --algs mscn --query_dir queries/ergast \
  -n -1 --max_epochs 50 --eval_epoch 5 \
  --onehot_dropout 2 --join_bitmap 1 --sample_bitmap 0 \
  --use_wandb 1 --val_size 0.2 --test_size 0.2 \
  --lr 0.001 --hidden_layer_size 512"
  echo $CMD
  eval $CMD
  CMD="python3 main.py --algs mscn --query_dir queries/ergast \
  -n -1 --max_epochs 50 --eval_epoch 5 \
  --onehot_dropout 0 --join_bitmap 0 --sample_bitmap 1 \
  --use_wandb 1 --val_size 0.2 --test_size 0.2 \
  --lr 0.001 --hidden_layer_size 512"
  echo $CMD
  eval $CMD


done

