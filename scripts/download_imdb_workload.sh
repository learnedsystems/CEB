#!/bin/sh
wget -O imdb.tar.gz https://www.dropbox.com/s/4aykbs6c8myeq2y/imdb.tar.gz?dl=1
tar -xvf imdb.tar.gz
mkdir -p queries
mv imdb queries/imdb
rm imdb.tar.gz

## downloading bitmaps

wget -O imdb_bitmaps2.tar.gz https://www.dropbox.com/s/qnzrzxnr5c6paa2/imdb_bitmaps2.tar.gz?dl=1

tar -xvf imdb_bitmaps2.tar.gz
mkdir -p queries
mkdir -p queries/allbitmaps
mv imdb_bitmaps2 queries/allbitmaps/imdb_bitmaps

rm imdb_bitmaps2.tar.gz

## downloading data about imdb schema etc.
wget -O imdb_data.json https://www.dropbox.com/s/o8m1fthow6zn1kg/imdb-unique-plans-sqls.tar.gz?dl=1
mv imdb_data.json queries/imdb/dbdata.json
