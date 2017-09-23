#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

mkdir imagenet_models
cd imagenet_models

wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
wget https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt

wget http://cs.unc.edu/~wliu/projects/ParseNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel
wget https://gist.githubusercontent.com/weiliu89/2ed6e13bfd5b57cf81d6/raw/758667b33d1d1ff2ac86b244a662744b7bb48e01/VGG_ILSVRC_16_layers_fc_reduced_deploy.prototxt

wget http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_CNN_S.caffemodel
wget https://gist.githubusercontent.com/ksimonyan/fd8800eeb36e276cd6f9/raw/e8dbbd31fc037fdf5430d89c102619e31ca7e8ef/VGG_CNN_S_deploy.prototxt

wget http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_CNN_M_2048.caffemodel
wget https://gist.githubusercontent.com/ksimonyan/78047f3591446d1d7b91/raw/93964ea9cab699d894cb76c7989b0e76c557bf4e/VGG_CNN_M_2048_deploy.prototxt

wget http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_CNN_F.caffemodel
wget https://gist.githubusercontent.com/ksimonyan/a32c9063ec8e1118221a/raw/6a3b8af023bae65669a4ceccd7331a5e7767aa4e/VGG_CNN_F_deploy.prototxt

