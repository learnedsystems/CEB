
mkdir -p queries

wget -O mlsys1-train.tar.gz https://www.dropbox.com/s/gvvdndcbuoo9zis/mlsys1-train.tar.gz?dl=1
tar -xvf mlsys1-train.tar.gz
mv mlsys1-train queries/
rm mlsys1-train.tar.gz

wget -O mlsys1-val.tar.gz https://www.dropbox.com/s/ottspau5x5dzxls/mlsys1-val.tar.gz?dl=1
tar -xvf mlsys1-val.tar.gz
mv mlsys1-val queries/
rm mlsys1-val.tar.gz

wget -O mlsys1-test.tar.gz https://www.dropbox.com/s/m85xmao9csdywp1/mlsys1-test.tar.gz?dl=1
tar -xvf mlsys1-test.tar.gz
mv mlsys1-test queries/
rm mlsys1-test.tar.gz

