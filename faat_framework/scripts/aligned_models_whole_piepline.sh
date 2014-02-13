#for entry in /home/aitor/data/willow_object_clouds/hannes_clouds2/*.pcd;
for entry in /home/aitor/data/willow_object_clouds/hannes_clouds_copy/*.pcd;
do
	export a=`echo "$entry" | awk '{ print substr( $0, 0, length($0) - 4 ) }'`
	echo $a
	export b=`echo "$a" | awk '{ print substr( $0, length($0) - 8, length($0)) }'`
	echo $b
    mkdir /home/aitor/willow_challenge_ros_code/read_willow_data/aligned_clouds/$b
    #echo `./bin/object_modelling_willow -pcd_files_dir /home/aitor/willow_challenge_ros_code/read_willow_data/train/$b -single_object 1 -dt_size 0.003 -vx_size 0.003 -max_corresp_dist 0.005 -pose_estimate 1 -Z_DIST 0.8 -x_limits -0.6 -aligned_output_dir /home/aitor/data/willow_object_clouds/models_ml/$b.pcd`
    echo `./bin/object_modelling_willow -pcd_files_dir /home/aitor/willow_challenge_ros_code/read_willow_data/train/$b -Z_DIST 0.8 -num_plane_inliers 2000 -max_corresp_dist 0.0075 -vx_size 0.002 -dt_size 0.002 -visualize 0 -fast_overlap 0 -aligned_output_dir /home/aitor/data/willow_structure_final/$b.pcd -aligned_model_saved_to /home/aitor/data/willow_object_clouds/models_ml_final/$b.pcd`
done
