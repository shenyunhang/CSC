#!/bin/bash
set -x
set -e

# weight
#./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_2 --set EXP_DIR vgg16_cpg_0304 TEST.USE_FLIPPED False TEST.SCALES "[688]" DRAW True

# refine
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_10 --set EXP_DIR vgg16_cpg_0316 TEST.USE_FLIPPED False TEST.SCALES "[688]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_1 --set EXP_DIR vgg16_cpg_0316 TEST.USE_FLIPPED False TEST.SCALES "[688]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_2 --set EXP_DIR vgg16_cpg_0316 TEST.USE_FLIPPED False TEST.SCALES "[688]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_3 --set EXP_DIR vgg16_cpg_0316 TEST.USE_FLIPPED False TEST.SCALES "[688]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_4 --set EXP_DIR vgg16_cpg_0316 TEST.USE_FLIPPED False TEST.SCALES "[688]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_5 --set EXP_DIR vgg16_cpg_0316 TEST.USE_FLIPPED False TEST.SCALES "[688]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_6 --set EXP_DIR vgg16_cpg_0316 TEST.USE_FLIPPED False TEST.SCALES "[688]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_7 --set EXP_DIR vgg16_cpg_0316 TEST.USE_FLIPPED False TEST.SCALES "[688]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_8 --set EXP_DIR vgg16_cpg_0316 TEST.USE_FLIPPED False TEST.SCALES "[688]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_9 --set EXP_DIR vgg16_cpg_0316 TEST.USE_FLIPPED False TEST.SCALES "[688]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_1 --set EXP_DIR vgg16_cpg_0304 TEST.USE_FLIPPED False TEST.SCALES "[688]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_2 --set EXP_DIR vgg16_cpg_0304 TEST.USE_FLIPPED False TEST.SCALES "[688]"

./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_10 --set EXP_DIR vgg16_wsddn_0311 TEST.USE_FLIPPED False TEST.SCALES "[864]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_1 --set EXP_DIR vgg16_wsddn_0311 TEST.USE_FLIPPED False TEST.SCALES "[864]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_2 --set EXP_DIR vgg16_wsddn_0311 TEST.USE_FLIPPED False TEST.SCALES "[864]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_3 --set EXP_DIR vgg16_wsddn_0311 TEST.USE_FLIPPED False TEST.SCALES "[864]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_4 --set EXP_DIR vgg16_wsddn_0311 TEST.USE_FLIPPED False TEST.SCALES "[864]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_5 --set EXP_DIR vgg16_wsddn_0311 TEST.USE_FLIPPED False TEST.SCALES "[864]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_6 --set EXP_DIR vgg16_wsddn_0311 TEST.USE_FLIPPED False TEST.SCALES "[864]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_7 --set EXP_DIR vgg16_wsddn_0311 TEST.USE_FLIPPED False TEST.SCALES "[864]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_8 --set EXP_DIR vgg16_wsddn_0311 TEST.USE_FLIPPED False TEST.SCALES "[864]"
./experiments/scripts/cpg_test_trainval_debug.sh 1 VGG16 pascal_voc VGG16_iter_9 --set EXP_DIR vgg16_wsddn_0311 TEST.USE_FLIPPED False TEST.SCALES "[864]"

# result
#./tools/try.py --imdb voc_2007_test --set TEST.PROPOSAL_METHOD mcg
