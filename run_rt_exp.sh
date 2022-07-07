python3 evaluation/get_runtimes.py --port 5434 --user ceb --pwd password --result_dir results5433_512mb/True/ --reps 1

python3 evaluation/get_runtimes.py --port 5434 --user ceb --pwd password --result_dir results5433_512mb/Postgres/ --reps 1

docker restart card-db3

python3 evaluation/get_runtimes.py --port 5434 --user ceb --pwd password --result_dir results5433_512mb/True/ --reps 1

python3 evaluation/get_runtimes.py --port 5434 --user ceb --pwd password --result_dir results5433_512mb/Postgres/ --reps 1

docker restart card-db3

python3 evaluation/get_runtimes.py --port 5434 --user ceb --pwd password --result_dir results5433_512mb/True/ --reps 1

python3 evaluation/get_runtimes.py --port 5434 --user ceb --pwd password --result_dir results5433_512mb/Postgres/ --reps 1

docker restart card-db3

python3 evaluation/get_runtimes.py --port 5434 --user ceb --pwd password --result_dir results5433_512mb/True/ --reps 1

python3 evaluation/get_runtimes.py --port 5434 --user ceb --pwd password --result_dir results5433_512mb/Postgres/ --reps 1
