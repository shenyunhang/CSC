#!/bin/bash
set -x
set -e

cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir

redo=1
data_root_dir="data/VOCdevkit2010"
dataset_name="VOC2010"
mapfile="$root_dir/data/$dataset_name/labelmap_voc.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
	extra_cmd="$extra_cmd --redo"
fi
for subset in trainval test
do
	if [ $subset == "trainval" ]
	then
		python $root_dir/tools/ssd/generate_noise_gt.py $data_root_dir $root_dir/data/$dataset_name/$subset.txt $root_dir/data/$dataset_name/noise_Annotations
		python $root_dir/tools/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/noise_$subset.txt $root_dir/data/$dataset_name/$db/$dataset_name"_"$subset"_"$db
	else
		python $root_dir/tools/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset.txt $root_dir/data/$dataset_name/$db/$dataset_name"_"$subset"_"$db
	fi
done
