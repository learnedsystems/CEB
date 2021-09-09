#!/bin/sh

createdb -U $POSTGRES_USER imdb

wget -O /var/lib/postgresql/pg_imdb.tar https://www.dropbox.com/s/vq1owleo9nuyxdf/pg_imdb.tar.gz?dl=1
tar xfv /var/lib/postgresql/pg_imdb.tar -C /var/lib/postgresql/
#psql -d imdb -U $POSTGRES_USER -c "SHOW max_wal_size";
pg_restore -v -d imdb -U $POSTGRES_USER /var/lib/postgresql/pg_imdb
