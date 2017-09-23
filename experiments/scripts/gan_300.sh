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
		YEAR="2007"
		;;
	pascal_voc10)
		TRAIN_IMDB="voc_2010_trainval"
		TEST_IMDB="voc_2010_test"
		PT_DIR="pascal_voc"
		ITERS=10
		ITERS2=10
		YEAR="2010"
		;;
	pascal_voc12)
		TRAIN_IMDB="voc_2012_trainval"
		TEST_IMDB="voc_2012_test"
		PT_DIR="pascal_voc"
		ITERS=20
		ITERS2=10
		YEAR="2012"
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


start=false
for step in {0..10}
do
	if [ ${step} == 0 ]
	then
		echo "###############################################################################"
		echo "START POINT"
		start=true
	fi

	echo "###############################################################################"
	echo "current step: ${step}"

	if [ ${step} == 0  ]
	then
		use_feedback=False
		feedback_dir_test=""
		feedback_dir_trainval=""
		feedback_num=0
	else
		use_feedback=True
		feedback_dir_test=data/VOCdevkit${YEAR}/results/VOC${YEAR}/Main/${EXP_DIR}/ssd/$((${step}-1))_score_test_feedback/
		feedback_dir_trainval=data/VOCdevkit${YEAR}/results/VOC${YEAR}/Main/${EXP_DIR}/ssd/$((${step}-1))_score_trainval_feedback/
		feedback_num=512
	fi


	echo "###############################################################################"
	echo "TRAIN F:"
	if [ "$start" = true  ]
	then
		if [ ${step} == 0  ]
		then
			weights=data/imagenet_models/${NET}.v2.caffemodel
			time ./tools/wsl/train_net.py --gpu ${GPU_ID} \
				--solver models/${PT_DIR}/${NET}/cpg/solver.prototxt \
				--weights ${weights} \
				--imdb ${TRAIN_IMDB} \
				--iters ${ITERS} \
				--cfg experiments/cfgs/cpg.yml \
				${EXTRA_ARGS} \
				EXP_DIR ${EXP_DIR}/cpg/${step} \
				USE_FEEDBACK ${use_feedback} \
				FEEDBACK_DIR "${feedback_dir_trainval}" \
				FEEDBACK_NUM ${feedback_num}

			weights=output/${EXP_DIR}/cpg/${step}/${TRAIN_IMDB}/${NET}_iter_${ITERS}.caffemodel
		else
			weights=output/${EXP_DIR}/cpg/$((${step}-1))/${TRAIN_IMDB}/${NET}_2_iter_${ITERS2}.caffemodel
		fi


		time ./tools/wsl/train_net.py --gpu ${GPU_ID} \
			--solver models/${PT_DIR}/${NET}/cpg/solver2.prototxt \
			--weights ${weights} \
			--imdb ${TRAIN_IMDB} \
			--iters ${ITERS2} \
			--cfg experiments/cfgs/cpg.yml \
			${EXTRA_ARGS} \
			EXP_DIR ${EXP_DIR}/cpg/${step} \
			USE_FEEDBACK ${use_feedback} \
			FEEDBACK_DIR "${feedback_dir_trainval}" \
			FEEDBACK_NUM ${feedback_num}

	fi


	echo "###############################################################################"
	echo "TEST F:"
	if [ "$start" = true  ]
	then
		NET_FINAL=output/${EXP_DIR}/cpg/${step}/${TRAIN_IMDB}/${NET}_2_iter_${ITERS2}.caffemodel

		#use_feedback=False
		#feedback_dir_test=""
		#feedback_dir_trainval=""
		#feedback_num=0

		time ./tools/wsl/test_net.py --gpu ${GPU_ID} \
			--def models/${PT_DIR}/${NET}/cpg/test.prototxt \
			--net ${NET_FINAL} \
			--imdb ${TEST_IMDB} \
			--cfg experiments/cfgs/cpg.yml \
			${EXTRA_ARGS} \
			EXP_DIR ${EXP_DIR}/cpg/${step} \
			USE_FEEDBACK ${use_feedback} \
			FEEDBACK_DIR "${feedback_dir_test}" \
			FEEDBACK_NUM ${feedback_num}


		time ./tools/wsl/test_net.py --gpu ${GPU_ID} \
			--def models/${PT_DIR}/${NET}/cpg/test.prototxt \
			--net ${NET_FINAL} \
			--imdb ${TRAIN_IMDB} \
			--cfg experiments/cfgs/cpg.yml \
			${EXTRA_ARGS} \
			EXP_DIR ${EXP_DIR}/cpg/${step} \
			USE_FEEDBACK ${use_feedback} \
			FEEDBACK_DIR "${feedback_dir_trainval}" \
			FEEDBACK_NUM ${feedback_num}
	fi


	echo "###############################################################################"
	echo "TRAIN G:"
	if [ "$start" = true  ]
	then
		python ./tools/gan/ssd_voc_300.py ${YEAR} ${EXP_DIR}/ssd/${step} "${GPU_ID}"

		if [ ${step} == 0  ]
		then
			weights=data/imagenet_models/VGG_ILSVRC_16_layers_fc_reduced.caffemodel
		else
			weights=output/${EXP_DIR}/ssd/$((${step}-1))/VGG_VOC${YEAR}_iter_10000.caffemodel
		fi
		weights=data/imagenet_models/VGG_ILSVRC_16_layers_fc_reduced.caffemodel

		echo ---------------------------------------------------------------------
		echo showing the solver file:
		cat "output/${EXP_DIR}/ssd/${step}/solver.prototxt"
		echo ---------------------------------------------------------------------
		time ./tools/ssd/train_net.py --gpu ${GPU_ID} \
			--solver output/${EXP_DIR}/ssd/${step}/solver.prototxt \
			--weights ${weights} \
			--imdb ${TRAIN_IMDB} \
			--iters ${ITERS} \
			--cfg experiments/cfgs/gan_ssd_300.yml \
			${EXTRA_ARGS} \
			TRAIN.PSEUDO_PATH output/${EXP_DIR}/cpg/${step}/${TRAIN_IMDB}/${NET}_2_iter_${ITERS2}/detections_o.pkl
	fi


	echo "###############################################################################"
	echo "TEST G:"
	if [ "$start" = true  ]
	then
		python ./tools/gan/score_ssd_voc_300_test_feedback.py ${YEAR} ${EXP_DIR}/ssd/${step} "${GPU_ID}"
		python ./tools/gan/score_ssd_voc_300_trainval_feedback.py ${YEAR} ${EXP_DIR}/ssd/${step} "${GPU_ID}"
		python ./tools/gan/score_ssd_voc_300_test.py ${YEAR} ${EXP_DIR}/ssd/${step} "${GPU_ID}"
		python ./tools/gan/score_ssd_voc_300_trainval.py ${YEAR} ${EXP_DIR}/ssd/${step} "${GPU_ID}"

		dir_trainval=data/VOCdevkit${YEAR}/results/VOC${YEAR}/Main/${EXP_DIR}/ssd/${step}_score_trainval/
		dir_eval=data/VOCdevkit${YEAR}/results/VOC${YEAR}/Main/
		cp $dir_trainval/* $dir_eval
		rename -f -v "s/comp3_det_/comp3_F_det_/g" ${dir_eval}/*
		python tools/eval.py --salt F
	fi

done
