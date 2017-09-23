#!/bin/bash
set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
DATASET=$3
NET_PREFIX=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
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
	pascal_voc10)
		TRAIN_IMDB="voc_2010_trainval"
		#TEST_IMDB="voc_2010_test"
		TEST_IMDB="voc_2007_test"
		PT_DIR="pascal_voc"
		;;
	pascal_voc12)
		TRAIN_IMDB="voc_2012_trainval"
		#TEST_IMDB="voc_2012_test"
		TEST_IMDB="voc_2007_test"
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

mkdir -p "experiments/logs/${EXP_DIR}"
EXTRA_ARGS_SLUG_=`echo "$EXTRA_ARGS_SLUG" | sed 's/\//_/g'`
LOG="experiments/logs/${EXP_DIR}/${0##*/}_${NET}_${NET_PREFIX}_${EXTRA_ARGS_SLUG_}_`date +'%Y-%m-%d_%H-%M-%S'`.log"
LOG=`echo "$LOG" | sed 's/\[//g' | sed 's/\]//g'`
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

echo ---------------------------------------------------------------------
git log -1
git submodule foreach 'git log -1'
echo ---------------------------------------------------------------------

NET_FINAL=output/${EXP_DIR}/${TRAIN_IMDB}/${NET_PREFIX}.caffemodel

time ./tools/wsl/test_net.py --gpu ${GPU_ID} \
	--def models/${PT_DIR}/${NET}/cpg/test.prototxt \
	--net ${NET_FINAL} \
	--imdb ${TEST_IMDB} \
	--cfg experiments/cfgs/cpg.yml \
	${EXTRA_ARGS}
