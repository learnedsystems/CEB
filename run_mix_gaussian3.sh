REPS=(1 1 1 1 1 1 1 1 1)

for rep in "${REPS[@]}";
do
	CMD="python3 main.py --algs mscn --query_dir queries/synth_mix4 \
	--eval_fns qerr --train_test_split custom --train_tmps 1,4 \
	--test_tmps 2 --onehot_dropout 0 --max_epochs 400 \
	--lr 0.0001 --db_name synth1 --join_features 0 \
	--table_features 0 --pred_features 1 --flow_features 0 \
	--hidden_layer_size 16 -n -1 --use_wandb 1 \
	--wandb_tags mix4
	--eval_epoch 10 --onehot_mask_truep 0.8 --use_wandb 1"
	echo $CMD
	eval $CMD

	CMD="python3 main.py --algs mscn --query_dir queries/synth_mix4 \
	--eval_fns qerr --train_test_split custom --train_tmps 1,4 \
	--test_tmps 2 --onehot_dropout 2 --max_epochs 400 \
	--lr 0.0001 --db_name synth1 --join_features 0 \
	--table_features 0 --pred_features 1 --flow_features 1 \
	--hidden_layer_size 16 -n -1 --use_wandb 1 \
	--wandb_tags mix4
	--eval_epoch 10 --onehot_mask_truep 0.8 --use_wandb 1"
	echo $CMD
	eval $CMD

	#CMD="python3 main.py --algs mscn --query_dir queries/synth_mix4 \
	#--eval_fns qerr --train_test_split custom --train_tmps 1,4 \
	#--test_tmps 2 --onehot_dropout 2 --max_epochs 400 \
	#--lr 0.0001 --db_name synth1 --join_features 0 \
	#--table_features 0 --pred_features 1 --flow_features 1 \
	#--hidden_layer_size 16 -n -1 --use_wandb 1 \
	#--wandb_tags mix4
	#--eval_epoch 10 --onehot_mask_truep 0.7 --use_wandb 1"
	#echo $CMD
	#eval $CMD
done
