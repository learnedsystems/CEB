#!/bin/sh
wget -O stack.tar.gz https://www.dropbox.com/s/mm4uibrzvdynmbj/stack.tar.gz?dl=1
tar -xvf stack.tar.gz
mkdir -p queries
mv stack queries
rm stack.tar.gz
