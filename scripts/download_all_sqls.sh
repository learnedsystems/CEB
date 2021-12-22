#!/bin/sh
wget -O imdb-unique-plans-sqls.tar.gz https://www.dropbox.com/s/o8m1fthow6zn1kg/imdb-unique-plans-sqls.tar.gz?dl=1
tar -xvf imdb-unique-plans-sqls.tar.gz
mkdir -p sqls
mv imdb-unique-plans-sqls sqls/imdb-unique-plans-sqls
rm imdb-unique-plans-sqls.tar.gz

wget -O imdb-sqls.tar.gz https://www.dropbox.com/s/azpofzyusz4evp6/imdb-sqls.tar.gz?dl=1
tar -xvf imdb-sqls.tar.gz
mkdir -p sqls
mv imdb-sqls sqls/imdb-sqls
rm imdb-sqls.tar.gz
