#!/bin/sh

mkdir -p queries

# JOBLight_train 1950
wget -O joblight_train_1950.tar.gz https://www.dropbox.com/s/cenbc0paq9yvjaq/joblight_train_1950.tar.gz?dl=1

tar -xvf joblight_train_1950.tar.gz
mv joblight_train_1950 queries/
rm joblight_train.tar.gz

## downloading data about imdb schema etc.
wget -O imdb_data.json https://www.dropbox.com/s/o8m1fthow6zn1kg/imdb-unique-plans-sqls.tar.gz?dl=1
cp imdb_data.json queries/joblight_train_1950/dbdata.json


# JOBLight_train 1980
wget -O joblight_train_1980.tar.gz https://www.dropbox.com/s/hcxl2455h2dmoii/joblight_train_1980.tar.gz?dl=1

tar -xvf joblight_train_1980.tar.gz
mv joblight_train_1980 queries/
rm joblight_train.tar.gz

## downloading data about imdb schema etc.
wget -O imdb_data.json https://www.dropbox.com/s/o8m1fthow6zn1kg/imdb-unique-plans-sqls.tar.gz?dl=1
cp imdb_data.json queries/joblight_train_1980/dbdata.json


