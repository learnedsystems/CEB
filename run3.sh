#TEMPLATES=(9 8 7 6 5 4 3 2 1)
#TESTTMP=10
REPS=(1 1 1 1 1)
USEWANDB=1
SAMPLEB=1
LR=0.001
EPOCHS=60
SAMPLES=10000

#QDIR=queries/synth_gaussian
EVAL_DIR=queries/ergast/
EVALTMP=7,7c

for rep in "${REPS[@]}";
do
  CMD="""python3 main.py --algs mscn \
  --query_dir ../learned-cardinalities/queries/ergastf1_train \
  --max_epochs $EPOCHS --lr $LR \
  --eval_query_dir $EVAL_DIR \
  --eval_templates $EVALTMP \
  --val_size 0.0 --test_size 0.0 \
  --eval_fns qerr,ppc --db_name ergastf1 \
  --sample_bitmap 0 \
  --join_bitmap 1 \
  --onehot_dropout 2 \
  --bitmap_onehotmask 1 \
  -n $SAMPLES \
  --eval_epoch 2 --use_wandb 1"""
  echo $CMD
  eval $CMD

  #CMD="""python3 main.py --algs mscn \
  #--query_dir ../learned-cardinalities/queries/ergastf1_train \
  #--max_epochs $EPOCHS --lr $LR \
  #--eval_query_dir $EVAL_DIR \
  #--eval_templates $EVALTMP \
  #--val_size 0.0 --test_size 0.0 \
  #--eval_fns qerr,ppc --db_name ergastf1 \
  #--sample_bitmap 1 \
  #--flow_features 0 \
  #--join_bitmap 0 \
  #--onehot_dropout 0 \
  #--bitmap_onehotmask 1 \
  #-n $SAMPLES \
  #--eval_epoch 2 --use_wandb 1"""
  #echo $CMD
  #eval $CMD


  CMD="""python3 main.py --algs mscn \
  --query_dir ../learned-cardinalities/queries/ergastf1_train \
  --max_epochs $EPOCHS --lr $LR \
  --eval_query_dir $EVAL_DIR \
  --eval_templates $EVALTMP \
  --val_size 0.0 --test_size 0.0 \
  --eval_fns qerr,ppc --db_name ergastf1 \
  --sample_bitmap 0 \
  --join_bitmap 1 \
  --onehot_dropout 2 \
  --bitmap_onehotmask 0 \
  -n $SAMPLES \
  --eval_epoch 2 --use_wandb 1"""
  echo $CMD
  eval $CMD

done

