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

for((ITERS=1;ITERS<=20;ITERS++))
do
	NET_PREFIX="${NET}_iter_${ITERS}"
	if [ ! -d "output/${EXP_DIR}/${TEST_IMDB}/${NET_PREFIX}" ]
	then
		./experiments/scripts/cpg_test.sh ${GPU_ID} ${NET} ${DATASET} ${NET_PREFIX} ${EXTRA_ARGS}
	fi
done

for((ITERS=1;ITERS<=20;ITERS++))
do
	NET_PREFIX="${NET}_2_iter_${ITERS}"
	if [ ! -d "output/${EXP_DIR}/${TEST_IMDB}/${NET_PREFIX}" ]
	then
		./experiments/scripts/cpg_test.sh ${GPU_ID} ${NET} ${DATASET} ${NET_PREFIX} ${EXTRA_ARGS}
	fi
done
