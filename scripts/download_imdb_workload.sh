#!/bin/sh
#wget -O imdb.tar.gz https://www.dropbox.com/s/jva9rvd4f3yy5v0/minified_dataset.tar.gz?dl=1
wget -O imdb.tar.gz https://www.dropbox.com/s/4aykbs6c8myeq2y/imdb.tar.gz?dl=1
tar -xvf imdb.tar.gz
mkdir -p queries
mv imdb queries/imdb
rm imdb.tar.gz
