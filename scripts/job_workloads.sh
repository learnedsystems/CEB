#!/bin/sh

wget -O job.tar.gz https://www.dropbox.com/s/i2tphhwv1u4o26k/job.tar.gz?dl=1

mkdir -p queries
tar -xvf job.tar.gz
mv job queries/
rm job.tar.gz

wget -O jobm.tar.gz https://www.dropbox.com/s/z8dujmx7dqpggnm/jobm.tar.gz?dl=1

mkdir -p queries
tar -xvf jobm.tar.gz
mv jobm queries/
rm jobm.tar.gz
