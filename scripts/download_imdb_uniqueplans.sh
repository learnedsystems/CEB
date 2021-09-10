#!/bin/sh
wget -O imdb-unique-plans.tar.gz https://www.dropbox.com/s/u3trdnof6xj074f/imdb-unique-plans.tar.gz?dl=1

tar -xvf imdb-unique-plans.tar.gz
mkdir -p queries
mv imdb-unique-plans queries/imdb-unique-plans
rm imdb-unique-plans.tar.gz
