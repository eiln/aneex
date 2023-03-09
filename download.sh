#!/usr/bin/bash

cd vision/compile
curl -LO https://www.dropbox.com/s/vkoar8ay7b2to7v/hwx.tar.gz?dl=0
tar xzvf hwx.tar.gz
rm -f hwx.tar.gz
cd -

cd compute
curl -LO https://www.dropbox.com/s/m73pwwuy8j3mq80/hwx-compute.tar.gz?dl=0
tar xzvf hwx-compute.tar.gz
rm -f hwx-compute.tar.gz
cd -
