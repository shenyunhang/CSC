#!/bin/bash

#
#/experiments/scripts/test_cpg_ensemble.sh pascal_voc cpg.yml output/vgg16_cpg_0219/voc_2007_test/VGG16_iter_1 output/vgg16_cpg_0219/voc_2007_test/VGG16_iter_5
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
	--imdb ${TEST_IMDB} \
	--cfg experiments/cfgs/cpg.yml \
	--result ${EXTRA_ARGS}
