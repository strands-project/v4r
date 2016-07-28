#!/bin/bash

mkdir cfg
cd cfg

echo "Downloading network definition"
wget -c https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/deploy.prototxt
echo "Downloading trained network parameters"
wget -c http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel #trained network
echo "Downloading trained mean pixel values"
wget -c http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz

echo "Unzipping..."
tar -xf caffe_ilsvrc12.tar.gz && rm -f caffe_ilsvrc12.tar.gz train.txt test.txt val.txt *synset* imagenet.bet.pickle
cd ..

echo "Done!"
