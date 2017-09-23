#!/bin/bash
set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

is_next=false
for var in "$@"
do
	if ${is_next}
	then
		EXP_DIR=${var}
		break
	fi
	if [ ${var} == "EXP_DIR" ]
	then
		is_next=true
	fi
done

case $DATASET in
	pascal_voc)
		TRAIN_IMDB="voc_2007_trainval"
		TEST_IMDB="voc_2007_test"
		PT_DIR="pascal_voc"
		ITERS=20
		ITERS2=10
		;;
	pascal_voc10)
		TRAIN_IMDB="voc_2010_trainval"
		TEST_IMDB="voc_2007_test"
		PT_DIR="pascal_voc"
		ITERS=20
		ITERS2=10
		;;
	pascal_voc12)
		TRAIN_IMDB="voc_2012_trainval"
		TEST_IMDB="voc_2007_test"
		PT_DIR="pascal_voc"
		ITERS=20
		ITERS2=10
		;;
	pascal_voc07+12)
		TRAIN_IMDB="voc_2007+2012_trainval"
		TEST_IMDB="voc_2007_test"
		PT_DIR="pascal_voc"
		ITERS=20
		ITERS2=10
		;;
	coco)
		TRAIN_IMDB="coco_2014_train"
		TEST_IMDB="coco_2014_minival"
		PT_DIR="coco"
		ITERS=280000
		;;
	*)
		echo "No dataset given"
		exit
		;;
esac

for((ITER=1;ITER<=ITERS;ITER++))
do
	NET_PREFIX="${NET}_iter_${ITER}"
	if [ ! -d "output/${EXP_DIR}/${TEST_IMDB}/${NET_PREFIX}" ]
	then
		./experiments/scripts/cpg_test.sh ${GPU_ID} ${NET} ${DATASET} ${NET_PREFIX} ${EXTRA_ARGS}
	fi
done

for((ITER=1;ITER<=ITERS2;ITER++))
do
	NET_PREFIX="${NET}_2_iter_${ITER}"
	if [ ! -d "output/${EXP_DIR}/${TEST_IMDB}/${NET_PREFIX}" ]
	then
		./experiments/scripts/cpg_test.sh ${GPU_ID} ${NET} ${DATASET} ${NET_PREFIX} ${EXTRA_ARGS}
	fi
done
