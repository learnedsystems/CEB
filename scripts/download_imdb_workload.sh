#!/bin/sh

mkdir -p queries


## full CEB-IMDB workload
wget -O imdb.tar.gz https://www.dropbox.com/s/4aykbs6c8myeq2y/imdb.tar.gz?dl=1
tar -xvf imdb.tar.gz
mv imdb queries/ceb-imdb-full
rm imdb.tar.gz

## downloading data about imdb schema etc.
wget -O imdb_data.json https://www.dropbox.com/s/o8m1fthow6zn1kg/imdb-unique-plans-sqls.tar.gz?dl=1
cp imdb_data.json queries/ceb-imdb-full/dbdata.json

## CEB-IMDb UniquePlans workload (~3k queries, subset of full CEB-IMDb)
wget -O imdb-unique-plans.tar.gz https://www.dropbox.com/s/u3trdnof6xj074f/imdb-unique-plans.tar.gz?dl=1

tar -xvf imdb-unique-plans.tar.gz
mkdir -p queries
mv imdb-unique-plans queries/ceb-imdb
rm imdb-unique-plans.tar.gz
cp imdb_data.json queries/ceb-imdb/dbdata.json

## JOB workload
wget -O job.tar.gz https://www.dropbox.com/s/i2tphhwv1u4o26k/job.tar.gz?dl=1

tar -xvf job.tar.gz
mv job queries/
rm job.tar.gz
cp imdb_data.json queries/job/dbdata.json

# JOBLight_train
wget -O joblight_train.tar.gz https://www.dropbox.com/s/iedwloyk6zse2kc/joblight_train.tar.gz?dl=1

tar -xvf joblight_train.tar.gz
mv joblight_train queries/
rm joblight_train.tar.gz
cp imdb_data.json queries/joblight_train/dbdata.json

## JOB-M
wget -O jobm.tar.gz https://www.dropbox.com/s/z8dujmx7dqpggnm/jobm.tar.gz?dl=1

mkdir -p queries
tar -xvf jobm.tar.gz
mv jobm queries/
rm jobm.tar.gz
cp imdb_data.json queries/jobm/dbdata.json

## download bitmaps for all these workloads
wget -O allbitmaps.tar.gz https://www.dropbox.com/s/eph3q5a3dcqv7io/allbitmaps.tar.gz?dl=1
tar -xvf allbitmaps.tar.gz

mv allbitmaps queries/
rm allbitmaps.tar.gz

