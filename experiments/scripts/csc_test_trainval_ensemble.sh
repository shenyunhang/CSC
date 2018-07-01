#!/bin/bash

#
#/experiments/scripts/test_csc_ensemble.sh pascal_voc csc.yml output/vgg16_csc_0219/voc_2007_test/VGG16_iter_1 output/vgg16_csc_0219/voc_2007_test/VGG16_iter_5
#

set -x
set -e

export PYTHONUNBUFFERED="True"

DATASET=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
	pascal_voc)
		TRAIN_IMDB="voc_2007_trainval"
		TEST_IMDB="voc_2007_test"
		PT_DIR="pascal_voc"
		;;
	pascal_voc10)
		TRAIN_IMDB="voc_2010_trainval"
		TEST_IMDB="voc_2010_test"
		PT_DIR="pascal_voc"
		;;
	pascal_voc12)
		TRAIN_IMDB="voc_2012_trainval"
		TEST_IMDB="voc_2012_test"
		PT_DIR="pascal_voc"
		;;
	pascal_voc07+12)
		TRAIN_IMDB="voc_2007+2012_trainval"
		TEST_IMDB="voc_2007_test"
		PT_DIR="pascal_voc"
		;;
	coco)
		TRAIN_IMDB="coco_2014_train"
		TEST_IMDB="coco_2014_minival"
		PT_DIR="coco"
		;;
	*)
		echo "No dataset given"
		exit
		;;
esac

mkdir -p "experiments/logs/ensemble"
LOG="experiments/logs/ensemble/${0##*/}_${DATASET}_`date +'%Y-%m-%d_%H-%M-%S'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net_wsl_ensemble.py  \
	--imdb ${TRAIN_IMDB} \
	--cfg experiments/cfgs/csc.yml \
	--result ${EXTRA_ARGS}
