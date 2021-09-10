#!/bin/sh
#createdb -U "$CARD_USER" imdb

#autovacuum = on			# Enable autovacuum subprocess?  'on'
sed -i 's/autovacuum = on/autovacuum = off/g' /var/lib/postgresql/data/postgresql.conf

#sed -i 's/max_wal_size = 1GB/max_wal_size = 50GB/g' /var/lib/postgresql/data/postgresql.conf
sed -i 's/shared_buffers = 128MB/shared_buffers = 1GB/g' /var/lib/postgresql/data/postgresql.conf

sed -i 's/qeqo = on/geqo = off/g' /var/lib/postgresql/data/postgresql.conf
sed -i 's/max_parallel_workers = 8/max_parallel_workers = 0/g' /var/lib/postgresql/data/postgresql.conf
sed -i 's/max_parallel_workers_per_gather = 2/max_parallel_workers_per_gather = 0/g' /var/lib/postgresql/data/postgresql.conf

echo "done updating conf file"


