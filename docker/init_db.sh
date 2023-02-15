#!/bin/sh
#createdb -U "$CARD_USER" imdb

#autovacuum = on			# Enable autovacuum subprocess?  'on'
sed -i 's/autovacuum = on/autovacuum = off/g' /var/lib/postgresql/data/postgresql.conf

sed -i 's/max_wal_size = 1GB/max_wal_size = 50GB/g' /var/lib/postgresql/data/postgresql.conf

## default
#sed -i 's/shared_buffers = 128MB/shared_buffers = 4GB/g' /var/lib/postgresql/data/postgresql.conf

### these are being set for low-memory version
sed -i 's/shared_buffers = 128MB/shared_buffers = 128kB/g' /var/lib/postgresql/data/postgresql.conf

#sed -i 's/maintenance_work_mem = 64MB/maintenance_work_mem = 16MB/g' /var/lib/postgresql/data/postgresql.conf
#sed -i 's/work_mem = 4MB/work_mem = 512kB/g' /var/lib/postgresql/data/postgresql.conf

#sed -i 's/geqo = on/geqo = off/g' /var/lib/postgresql/data/postgresql.conf
#sed -i 's/max_parallel_workers = 8/max_parallel_workers = 0/g' /var/lib/postgresql/data/postgresql.conf
#sed -i 's/max_parallel_workers_per_gather = 2/max_parallel_workers_per_gather = 0/g' /var/lib/postgresql/data/postgresql.conf

echo "geqo = off" >> /var/lib/postgresql/data/postgresql.conf
echo "max_parallel_workers = 0" >> /var/lib/postgresql/data/postgresql.conf
echo "max_parallel_workers_per_gather = 0" >> /var/lib/postgresql/data/postgresql.conf

echo "done updating conf file"


