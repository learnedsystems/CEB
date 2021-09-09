#!/bin/sh
wget -O stack.tar.gz https://www.dropbox.com/s/u4xpl6tryon0h26/so_workload.tar.gz?dl=1
tar -xvf stack.tar.gz
mkdir -p queries
mkdir -p queries/stack
mv so_workload/* queries/stack
rm stack.tar.gz
