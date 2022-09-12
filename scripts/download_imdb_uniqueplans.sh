#!/bin/sh
wget -O imdb-unique-plans.tar.gz https://www.dropbox.com/s/u3trdnof6xj074f/imdb-unique-plans.tar.gz?dl=1

tar -xvf imdb-unique-plans.tar.gz
mkdir -p queries
mv imdb-unique-plans queries/imdb-unique-plans
rm imdb-unique-plans.tar.gz

## downloading bitmaps

wget -O imdb_bitmaps2.tar.gz https://www.dropbox.com/s/qnzrzxnr5c6paa2/imdb_bitmaps2.tar.gz?dl=1

tar -xvf imdb_bitmaps2.tar.gz
mkdir -p queries
mkdir -p queries/allbitmaps
mv imdb_bitmaps2 queries/allbitmaps/imdb_bitmaps

rm imdb_bitmaps2.tar.gz
