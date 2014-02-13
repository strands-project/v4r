#for entry in /home/aitor/data/willow_object_clouds/hannes_clouds2/*.pcd;
for entry in /home/aitor/data/willow_object_clouds/hannes_clouds_copy/object_*.pcd;
do
	export a=`echo "$entry" | awk '{ print substr( $0, 0, length($0) - 4 ) }'`
	echo $a
	export b=`echo "$a" | awk '{ print substr( $0, length($0) - 8, length($0)) }'`
	echo $b
    #mkdir /home/aitor/willow_challenge_ros_code/read_willow_data/aligned_clouds/$b
    #echo `./bin/object_modelling_willow -pcd_files_dir /home/aitor/willow_challenge_ros_code/read_willow_data/train/$b -single_object 1 -dt_size 0.003 -vx_size 0.003 -max_corresp_dist 0.005 -pose_estimate 1 -Z_DIST 0.8 -x_limits -0.6 -aligned_output_dir /home/aitor/data/willow_object_clouds/models_ml/$b.pcd`
    #echo `./bin/object_modelling_willow -pcd_files_dir /home/aitor/willow_challenge_ros_code/read_willow_data/train/$b -Z_DIST 0.8 -num_plane_inliers 2000 -max_corresp_dist 0.0075 -vx_size 0.002 -dt_size 0.002 -visualize 0 -fast_overlap 0 -aligned_output_dir /home/aitor/data/willow_structure_final/$b.pcd -aligned_model_saved_to /home/aitor/data/willow_object_clouds/models_ml_final/$b.pcd`
    echo "./bin/object_modelling_willow_unknown_poses -pcd_files_dir /home/aitor/data/willow_structure_final/$b.pcd/ -Z_DIST 0.9 -max_vis 50 -low 150 -high 200 -iterations 10 -max_corresp_dist 0.05 -dt_size 0.003 -use_cg 1 -organized_normals 0 -w_t 0.75 -fast_overlap 0 -aligned_output_dir new_models_with_weights_all_views/$b -vis_final 0 -min_dot 0.98 -mv_use_weights 1 -mv_iterations 15 -step 1"
    echo `./bin/object_modelling_willow_unknown_poses -pcd_files_dir /home/aitor/data/willow_structure_final/$b.pcd/ -Z_DIST 0.9 -max_vis 50 -low 150 -high 200 -iterations 10 -max_corresp_dist 0.05 -dt_size 0.003 -use_cg 1 -organized_normals 1 -w_t 0.75 -fast_overlap 0 -aligned_output_dir new_models_with_weights_all_views/$b -vis_final 0 -min_dot 0.98 -mv_use_weights 1 -mv_iterations 15 -step 1` 
done
