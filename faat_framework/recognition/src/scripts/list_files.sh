#!/bin/bash

cd $1
for i in $( find . -name '*.pts' );
do
	echo $i
    out_file=""
	export IFS="/"
	for word in $i; do
	  echo "$word"
	done

	export IFS=""
	out_file="/home/aitor/data/queens_dataset/pcd_scenes/$word.pcd"
	replace=""
	out_file=`echo $out_file | sed -e "s/.pts/${replace}/g"`
	echo $out_file
	/home/aitor/aldoma_employee_svn/code/faat_framework/build/bin/ops_to_pcd -ops_file $i -pcd_out_file $out_file -flip_z 1 -save_normals false
	#exit
done
