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
		ITERS=40000
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
LOG="experiments/logs/${EXP_DIR}/${0##*/}_${NET}_${EXTRA_ARGS_SLUG}_`date +'%Y-%m-%d_%H-%M-%S'`.log"
LOG=`echo "$LOG" | sed 's/\[//g' | sed 's/\]//g'`
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
	--solver models/${PT_DIR}/${NET}/fast_rcnn_wsl/solver.prototxt \
	--weights data/imagenet_models/${NET}.v2.caffemodel \
	--imdb ${TRAIN_IMDB} \
	--iters ${ITERS} \
	--cfg experiments/cfgs/fast_rcnn_wsl.yml \
	${EXTRA_ARGS}

	#--weights data/imagenet_models/${NET}.v2.caffemodel \
	#--weights output/vgg16_fast_rcnn_wsl_/voc_2007_trainval/VGG16_iter_40000.caffemodel \
	#--weights output/vgg16_cpg_0304/voc_2007_trainval/VGG16_iter_2.caffemodel \

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
	--def models/${PT_DIR}/${NET}/fast_rcnn_wsl/test.prototxt \
	--net ${NET_FINAL} \
	--imdb ${TEST_IMDB} \
	--cfg experiments/cfgs/fast_rcnn_wsl.yml \
	${EXTRA_ARGS}
