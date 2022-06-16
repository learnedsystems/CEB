#!/usr/bin/env bash
echo "drop cache!"
#rm -f logfile
#pg_ctl -D $PG_DATA_DIR -m i restart -l logfile
sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"
echo "drop cache done!"
sudo docker restart card-db

#sudo systemctl restart postgresql
