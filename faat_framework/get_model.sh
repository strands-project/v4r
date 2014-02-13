#!/bin/bash

build_dir="/qtcreator-build"
model=$1
nseq=$2
use_color=1
pairwise=$3

if [ ${pairwise} != 0 ]; then
	for (( c=1; c<=nseq; c++ ))
	do
	   echo "Executing pairwise registration for sequence $c"
	   echo ".${build_dir}/bin/object_modelling_willow_unknown_poses -pcd_files_dir /media/DATA/models/${model}/seq${c}/bin/ -Z_DIST 1 -low 100 -high 150 -iterations 20 -max_corresp_dist 0.05 -vis_final 0 -step 1 -vis_pairwise_ 0 -aligned_output_dir /media/DATA/models/${model}/seq${c}_aligned -seg_type 1"
	   .${build_dir}/bin/object_modelling_willow_unknown_poses -pcd_files_dir /media/DATA/models/${model}/seq${c}/bin/ -Z_DIST 1 -low 100 -high 150 -iterations 20 -max_corresp_dist 0.05 -vis_final 0 -step 1 -vis_pairwise_ 0 -aligned_output_dir /media/DATA/models/${model}/seq${c}_aligned -seg_type 1
	done
else
	echo "Pairwise deactivated"
fi

if [ ${nseq} -gt 1 ]; then
	
	sequences=""
	for (( c=1; c<=nseq; c++ ))
	do
		if [ ${c} == ${nseq} ]; then
			sequences="$sequences/media/DATA/models/${model}/seq${c}_aligned" 
		else
			sequences="$sequences/media/DATA/models/${model}/seq${c}_aligned," 
		fi
	done
	
    echo "Going to merge sequences"
	echo ".${build_dir}/bin/merge_multiple -sequences ${sequences} -overlap 0.5 -inliers_threshold 0.01 -output_path /media/DATA/models/${model}/merged_sequences -use_color ${use_color}"
	.${build_dir}/bin/merge_multiple -sequences ${sequences} -overlap 0.5 -inliers_threshold 0.01 -output_path /media/DATA/models/${model}/merged_sequences -use_color ${use_color} -visualize 1
	
	#multiview
    echo "Multiview of multiple sequences"
	echo ".${build_dir}/bin/multiview -pcd_files_dir /media/DATA/models/${model}/merged_sequences/ -aligned_output_dir /media/DATA/models/${model}/merged_sequences_multiview/ -mv_iterations 5"
	.${build_dir}/bin/multiview -pcd_files_dir /media/DATA/models/${model}/merged_sequences/ -aligned_output_dir /media/DATA/models/${model}/merged_sequences_multiview/ -mv_iterations 5
	
    echo "Get nice model"
	echo ".${build_dir}/bin/get_nice_model -input_dir /media/DATA/models/${model}/merged_sequences_multiview/ -organized_normals 1 -w_t 0.9 -lateral_sigma 0.001 -octree_resolution 0.0015 -mls_radius 0.002 -visualize 0 -structure_for_recognition /media/DATA/models/recognition_structure/${model}.pcd -save_model_to /media/DATA/models/nice_models/${model}.pcd -bring_to_plane 0"
	.${build_dir}/bin/get_nice_model -input_dir /media/DATA/models/${model}/merged_sequences_multiview/ -organized_normals 1 -w_t 0.9 -lateral_sigma 0.001 -octree_resolution 0.0015 -mls_radius 0.002 -visualize 0 -structure_for_recognition /media/DATA/models/recognition_structure/${model}.pcd -save_model_to /media/DATA/models/nice_models/${model}.pcd -bring_to_plane 0
	
else
    echo "Get nice model"
	echo ".${build_dir}/bin/get_nice_model -input_dir /media/DATA/models/${model}/seq1_aligned/ -organized_normals 1 -w_t 0.9 -lateral_sigma 0.001 -octree_resolution 0.0015 -mls_radius 0.002 -visualize 0 -structure_for_recognition /media/DATA/models/recognition_structure/${model}.pcd -save_model_to /media/DATA/models/nice_models/${model}.pcd -bring_to_plane 1"
	.${build_dir}/bin/get_nice_model -input_dir /media/DATA/models/${model}/seq1_aligned/ -organized_normals 1 -w_t 0.9 -lateral_sigma 0.001 -octree_resolution 0.0015 -mls_radius 0.002 -visualize 0 -structure_for_recognition /media/DATA/models/recognition_structure/${model}.pcd -save_model_to /media/DATA/models/nice_models/${model}.pcd -bring_to_plane 1
fi




