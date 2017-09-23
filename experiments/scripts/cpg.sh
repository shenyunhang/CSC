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
		TEST_IMDB="voc_2010_test"
		PT_DIR="pascal_voc"
		ITERS=20
		ITERS2=10
		;;
	pascal_voc12)
		TRAIN_IMDB="voc_2012_trainval"
		TEST_IMDB="voc_2012_test"
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

mkdir -p "experiments/logs/${EXP_DIR}"
LOG="experiments/logs/${EXP_DIR}/${0##*/}_${NET}_${EXTRA_ARGS_SLUG}_`date +'%Y-%m-%d_%H-%M-%S'`.log"
LOG=`echo "$LOG" | sed 's/\[//g' | sed 's/\]//g'`
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

echo ---------------------------------------------------------------------
git log -1
git submodule foreach 'git log -1'
echo ---------------------------------------------------------------------

echo ---------------------------------------------------------------------
echo showing the solver file:
cat "models/${PT_DIR}/${NET}/cpg/solver.prototxt"
echo ---------------------------------------------------------------------
time ./tools/wsl/train_net.py --gpu ${GPU_ID} \
	--solver models/${PT_DIR}/${NET}/cpg/solver.prototxt \
	--weights data/imagenet_models/${NET}.v2.caffemodel \
	--imdb ${TRAIN_IMDB} \
	--iters ${ITERS} \
	--cfg experiments/cfgs/cpg.yml \
	${EXTRA_ARGS}

echo ---------------------------------------------------------------------
echo showing the solver file:
cat "models/${PT_DIR}/${NET}/cpg/solver2.prototxt"
echo ---------------------------------------------------------------------
time ./tools/wsl/train_net.py --gpu ${GPU_ID} \
	--solver models/${PT_DIR}/${NET}/cpg/solver2.prototxt \
	--weights output/${EXP_DIR}/${TRAIN_IMDB}/${NET}_iter_${ITERS}.caffemodel \
	--imdb ${TRAIN_IMDB} \
	--iters ${ITERS2} \
	--cfg experiments/cfgs/cpg.yml \
	${EXTRA_ARGS}

#--------------------------------------------------------------------------------------------------
set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} |tail -n 2 | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/wsl/test_net.py --gpu ${GPU_ID} \
	--def models/${PT_DIR}/${NET}/cpg/test.prototxt \
	--net ${NET_FINAL} \
	--imdb ${TEST_IMDB} \
	--cfg experiments/cfgs/cpg.yml \
	${EXTRA_ARGS}
