export BIN_PATH="/home/aitor/aldoma_employee_svn/code/faat_framework/qtcreator-build"
export MODELS_DIR_INPUT="/home/aitor/data/willow/new_models_with_weights_all_views"
export MODELS_DIR_OUTPUT="/home/aitor/data/willow/models"
export RECOGNIZER_STRUCTURE="/home/aitor/data/willow/recognizer_structure"
cd ${MODELS_DIR_INPUT}

for entry in `ls -d *`;
do
	#export a=`echo "$entry" | awk '{ print substr( $0, 0, length($0) - 4 ) }'`
	#export b=`echo "$a" | awk '{ print substr( $0, length($0) - 8, length($0)) }'`
	echo $entry
    export model_name=${entry}.pcd
    echo $model_name
    echo "${BIN_PATH}/bin/get_nice_model -input_dir ${MODELS_DIR_INPUT}/${entry}/ -lateral_sigma 0.001 -save_model_to ${MODELS_DIR_OUTPUT}/${model_name} -octree_resolution 0.0015 -visualize 0 -structure_for_recognition ${RECOGNIZER_STRUCTURE}/${entry}.pcd/"
    echo `${BIN_PATH}/bin/get_nice_model -input_dir ${MODELS_DIR_INPUT}/${entry}/ -lateral_sigma 0.001 -save_model_to ${MODELS_DIR_OUTPUT}/${model_name} -octree_resolution 0.0015 -visualize 0 -structure_for_recognition ${RECOGNIZER_STRUCTURE}/${entry}.pcd/`
    #echo `./bin/object_modelling_willow_unknown_poses -pcd_files_dir /home/aitor/data/willow_structure_final/$b.pcd/ -Z_DIST 0.9 -max_vis 50 -low 150 -high 200 -iterations 10 -max_corresp_dist 0.05 -dt_size 0.003 -use_cg 1 -organized_normals 1 -w_t 0.75 -fast_overlap 0 -aligned_output_dir new_models_with_weights_all_views/$b -vis_final 0 -min_dot 0.98 -mv_use_weights 1 -mv_iterations 15 -step 1` 
    
    #cp input directory to structure for recognizer
    #mkdir ${RECOGNIZER_STRUCTURE}/${entry}.pcd
    #cp ${MODELS_DIR_INPUT}/${entry}/* ${RECOGNIZER_STRUCTURE}/${entry}.pcd/.
done
