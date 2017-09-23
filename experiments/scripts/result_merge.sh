#!/bin/bash
set -x
set -e

#我们还需要算NMS，不能直接合并结果

for entry in `ls $2`
do
	echo "$entry"
	cp $2/$entry $1/$entry
done

for entry in `ls $1`
do
	echo "$entry"
	awk '{ print $1, 0.1*$2, $3, $4, $5, $6 }' $1/$entry > $1/$entry.txt
	mv $1/$entry.txt $1/$entry
done


for entry in `ls $3`
do
	echo "$entry"
	cat $3/$entry >> $1/$entry
done
exit
