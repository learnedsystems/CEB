#!/bin/sh

wget -O job.tar.gz https://www.dropbox.com/s/i2tphhwv1u4o26k/job.tar.gz?dl=1

mkdir -p queries
tar -xvf job.tar.gz
mv job queries/
rm job.tar.gz

## bitmaps
wget -O job_bitmaps2.tar.gz https://www.dropbox.com/s/6zsas4gmnj7x1uu/job_bitmaps2.tar.gz?dl=1

tar -xvf job_bitmaps2.tar.gz
mkdir -p queries
mkdir -p queries/allbitmaps
#mv job_bitmaps2 queries/allbitmaps/job_bitmaps
mv job_bitmaps2 queries/allbitmaps/job

rm job_bitmaps2.tar.gz

# JOBLight
wget -O joblight_train.tar.gz https://www.dropbox.com/s/iedwloyk6zse2kc/joblight_train.tar.gz?dl=1

mkdir -p queries
tar -xvf joblight_train.tar.gz
mv joblight_train queries/
rm joblight_train.tar.gz

## bitmaps
wget -O joblight_bitmaps2.tar.gz https://www.dropbox.com/s/2bhwfmky06m2mes/joblight_bitmaps2.tar.gz?dl=1

tar -xvf joblight_bitmaps2.tar.gz
mkdir -p queries
mkdir -p queries/allbitmaps
#mv joblight_bitmaps2 queries/allbitmaps/joblight_bitmaps
mv joblight_bitmaps2 queries/allbitmaps/joblight_train

rm joblight_bitmaps2.tar.gz

## JOB-M
#wget -O jobm.tar.gz https://www.dropbox.com/s/z8dujmx7dqpggnm/jobm.tar.gz?dl=1

#mkdir -p queries
#tar -xvf jobm.tar.gz
#mv jobm queries/
#rm jobm.tar.gz

## downloading data about imdb schema etc.
wget -O imdb_data.json https://www.dropbox.com/s/o8m1fthow6zn1kg/imdb-unique-plans-sqls.tar.gz?dl=1
cp imdb_data.json queries/job/dbdata.json
cp imdb_data.json queries/joblight_train/dbdata.json
rm imdb_data.json
