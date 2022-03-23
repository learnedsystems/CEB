time python3 main.py --algs mscn -n -1 --onehot_dropout 1 \
--mask_unseen_subplans 0 --training_opt swa --no_regex_templates 0 \
--feat_onlyseen_preds 1 --wandb_tags stats --subplan_level_outputs 0 \
--feat_separate_alias 0 --query_dir queries/imdb-unique-plans \
--hidden_layer_size 128 --train_test_split_kind template \
--diff_templates_seed 10 --max_discrete_featurizing_buckets 10 \
--max_like_featurizing_buckets 10 --weight_decay 0.0 \
--alg mscn --load_padded_mscn_feats 1 --load_query_together 0 \
--eval_fns qerr,ppc --loss_func mse --test_size 0.5 --result_dir results/ \
--max_epochs 10 --eval_epoch 100 --table_features 0 --join_features onehot-stats \
--set_column_feature onehot-stats

time python3 main.py --algs mscn -n -1 --onehot_dropout 1 \
--mask_unseen_subplans 0 --training_opt swa --no_regex_templates 0 \
--feat_onlyseen_preds 1 --wandb_tags stats --subplan_level_outputs 0 \
--feat_separate_alias 0 --query_dir queries/imdb-unique-plans \
--hidden_layer_size 128 --train_test_split_kind template \
--diff_templates_seed 10 --max_discrete_featurizing_buckets 10 \
--max_like_featurizing_buckets 10 --weight_decay 0.0 \
--alg mscn --load_padded_mscn_feats 1 --load_query_together 0 \
--eval_fns qerr,ppc --loss_func mse --test_size 0.5 --result_dir results/ \
--max_epochs 10 --eval_epoch 100 --table_features 0 --join_features onehot-stats \
--set_column_feature onehot-stats

time python3 main.py --algs mscn -n -1 --onehot_dropout 1 \
--mask_unseen_subplans 0 --training_opt swa --no_regex_templates 0 \
--feat_onlyseen_preds 1 --wandb_tags stats --subplan_level_outputs 0 \
--feat_separate_alias 0 --query_dir queries/imdb-unique-plans \
--hidden_layer_size 128 --train_test_split_kind template \
--diff_templates_seed 10 --max_discrete_featurizing_buckets 10 \
--max_like_featurizing_buckets 10 --weight_decay 0.0 \
--alg mscn --load_padded_mscn_feats 1 --load_query_together 0 \
--eval_fns qerr,ppc --loss_func mse --test_size 0.5 --result_dir results/ \
--max_epochs 10 --eval_epoch 100 --table_features 0 --join_features onehot-stats \
--set_column_feature onehot-stats

