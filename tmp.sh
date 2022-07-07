#HLS=$1
#JB=$2
#SB=$3
#DROP=$4
#EVALE=100

REPS=(1 1 1 1 1)

for r in "${!REPS[@]}";
do
  #CMD="time python3 main.py --algs mscn --query_dir queries/job-joinkeys \
  #--eval_fns qerr,ppc --val_size 0.0 --test_size 0.0 -n -1 \
  #--join_bitmap $JB --sample_bitmap $SB --lr 0.0001 \
  #--max_epochs 20 --eval_query_dir queries/imdb-unique-plans \
  #--onehot_dropout $DROP --eval_epoch $EVALE \
  #--hidden_layer_size $HLS"
  CMD="python3 main.py --algs mscn --query_dir queries/stats_train_mod \
	--val_size 0.0 --test_size 0.0 --db_name stats --pwd postgres --port 5440 \
	--max_discrete_feat 1 --max_like 1 --user postgres --onehot_dropout 0 \
	--join_bitmap 1 --sample_bitmap 0 --max_epochs 60 --eval_epoch 1000 \
	--eval_query_dir queries/stats --feat_onlyseen_maxy 1 --onehot_dropout 2 \
	--loss_func mse -n -1 --max_num_tables 3 --eval_epoch 2"
  echo $CMD
  eval $CMD
done
