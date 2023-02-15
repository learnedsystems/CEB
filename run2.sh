REPS=(1 1 1 1 1)
WANDB_TAGS='gaussian'

for rep in "${REPS[@]}";
do
	CMD="python3 main.py --algs mscn --query_dir queries/synth_2d_gaussian/ \
	--eval_fns qerr --train_test_split query --onehot_dropout 2 \
  --onehot_mask_truep 0.8 \
	--max_epochs 200 --lr 0.001 --db_name synth1 --join_features 0 \
	--table_features 0 --pred_features 1 --flow_features 1 \
	--hidden_layer_size 32 --eval_query_dir queries/synth_2d_gaussian1_gt \
	--use_wandb 1 --wandb_tags $WANDB_TAGS
	"
	echo $CMD
	eval $CMD

	CMD="python3 main.py --algs mscn --query_dir queries/synth_2d_gaussian/ \
	--eval_fns qerr --train_test_split query --onehot_dropout 2 \
  --onehot_mask_truep 0.7 \
	--max_epochs 200 --lr 0.001 --db_name synth1 --join_features 0 \
	--table_features 0 --pred_features 1 --flow_features 1 \
	--hidden_layer_size 32 --eval_query_dir queries/synth_2d_gaussian1_gt \
	--use_wandb 1 --wandb_tags $WANDB_TAGS
	"
	echo $CMD
	eval $CMD

	CMD="python3 main.py --algs mscn --query_dir queries/synth_2d_gaussian/ \
	--eval_fns qerr --train_test_split query --onehot_dropout 0 \
	--max_epochs 100 --lr 0.001 --db_name synth1 --join_features 0 \
	--table_features 0 --pred_features 1 --flow_features 1 \
	--hidden_layer_size 32 --eval_query_dir queries/synth_2d_gaussian1_gt \
	--use_wandb 1 --wandb_tags $WANDB_TAGS
	"
	echo $CMD
	eval $CMD

	CMD="python3 main.py --algs mscn --query_dir queries/synth_2d_gaussian/ \
	--eval_fns qerr --train_test_split query --onehot_dropout 0 \
	--max_epochs 100 --lr 0.001 --db_name synth1 --join_features 0 \
	--table_features 0 --pred_features 1 --flow_features 0 \
	--hidden_layer_size 32 --eval_query_dir queries/synth_2d_gaussian1_gt \
	--use_wandb 1 --wandb_tags $WANDB_TAGS
	"
	echo $CMD
	eval $CMD

done
