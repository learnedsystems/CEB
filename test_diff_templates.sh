
#SEEDS=(7 6 2 11 12 13 14 15 16 17 18 19 20 1 3 4 5 8 9 10)
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

for i in "${!SEEDS[@]}";
  do
  echo ${SEEDS[$i]}
  python3 main.py --algs mscn --train_test_split template \
   --diff_templates_seed ${SEEDS[$i]} \
   --query_dir queries/imdb-unique-plans -n 10 --onehot_dropout -1 --use_wandb 0
done
