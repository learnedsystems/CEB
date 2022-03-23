#bash run_all_diff_stats.sh 0 stats stats 0 10
#bash run_all_diff_stats.sh 0 stats stats 0 1

#bash run_all_diff_stats.sh 0 stats stats 0 10 1

#bash run_all_diff_stats.sh 0 stats stats 0 1 0
#bash run_all_diff_stats.sh 0 stats stats 0 1 0
#bash run_all_diff_stats.sh 0 stats stats 0 1 0

#bash run_all_diff_sel.sh 0 onehot-stats onehot-stats 1 10 0 1
#bash run_all_diff_sel.sh 0 onehot-stats onehot-stats 1 10 0 1
#bash run_all_diff_sel.sh 0 onehot-stats onehot-stats 1 10 0 1

# onehot-stats mix, with unseen masking
#bash run_all_diff_sel.sh 0 onehot-stats onehot-stats 1 10 0 1
#bash run_all_diff_sel.sh 0 onehot-stats onehot-stats 1 10 0 1
#bash run_all_diff_sel.sh 0 onehot-stats onehot-stats 1 10 0 1

# stats with no one-hots
bash run_all_diff_sel.sh 0 stats stats 0 1 0 0
bash run_all_diff_sel.sh 0 stats stats 0 1 0 0
bash run_all_diff_sel.sh 0 stats stats 0 1 0 0


## default version
#bash run_all_diff_sel.sh 1 onehot onehot 0 10 0 0
#bash run_all_diff_sel.sh 1 onehot-stats onehot-stats 0 10 0 0

#bash run_all_diff_sel.sh 1 onehot onehot 0 10 0 0
#bash run_all_diff_sel.sh 1 onehot-stats onehot-stats 0 10 0 0

#bash run_all_diff_sel.sh 1 onehot onehot 0 10 0 0
#bash run_all_diff_sel.sh 1 onehot-stats onehot-stats 0 10 0 0
