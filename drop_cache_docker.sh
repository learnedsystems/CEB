#!/usr/bin/env bash
echo "drop cache!"
#rm -f logfile
#pg_ctl -D $PG_DATA_DIR -m i restart -l logfile
#sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"
sudo docker stop card-db
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
echo "drop cache done!"
sudo docker start card-db

#sudo systemctl restart postgresql
