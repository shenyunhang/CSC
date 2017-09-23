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

if $NET ~= "VGG16"
then
	echo "only support VGG16 network"
	exit 0
fi

case $DATASET in
	pascal_voc)
		TRAIN_IMDB="voc_2007_trainval"
		TEST_IMDB="voc_2007_test"
		PT_DIR="pascal_voc"
		ITERS=30
		ITERS2=200
		;;
	pascal_voc10)
		TRAIN_IMDB="voc_2010_trainval"
		TEST_IMDB="voc_2010_test"
		PT_DIR="pascal_voc"
		ITERS=30
		;;
	pascal_voc07+12)
		TRAIN_IMDB="voc_2007+2012_trainval"
		TEST_IMDB="voc_2007_test"
		PT_DIR="pascal_voc"
		ITERS=30
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

mkdir -p "experiments/logs/${EXP_DIR}"
LOG="experiments/logs/${EXP_DIR}/${0##*/}_${NET}_${NET_PREFIX}_${EXTRA_ARGS_SLUG}_`date +'%Y-%m-%d_%H-%M-%S'`.log"
LOG=`echo "$LOG" | sed 's/\[//g' | sed 's/\]//g'`
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

echo ---------------------------------------------------------------------
git log -1
git submodule foreach 'git log -1'
echo ---------------------------------------------------------------------

python ./tools/fwsl/fwsl_pascalvoc07.py ${EXP_DIR}/FWSL_score

NET_FINAL=output/${EXP_DIR}/FWSL/${NET_PREFIX}.caffemodel


time ./tools/fwsl/test_net.py --gpu ${GPU_ID} \
	--def output/${EXP_DIR}/FWSL_score/deploy.prototxt \
	--net ${NET_FINAL} \
	--imdb ${TEST_IMDB} \
	--cfg experiments/cfgs/fwsl_fwsl.yml \
	${EXTRA_ARGS} \
	EXP_DIR ${EXP_DIR}/FWSL_score \
	TRAIN.SCALES [300] \
	TEST.SCALES [300]
