#!/bin/bash
function calculateAttentionPoints {

    base_dir='/media/Documents/research/db/GRAPHICON2013/willow/'
    base_name='willow_'

    base_directory_img=$base_dir'rgb/'$base_name
    base_extention='ppm'
    base_directory_attention=$base_dir$1'/'$base_name
    base_directory_output=$base_dir$2'/'$base_name

    path_to_exe=$3 #'../../../bin/ExmplWTA'

    start_idx=$4 #0
    end_idx=$5 #131

    mkdir $base_dir$2

    for (( i=$start_idx; i<=$end_idx; i++ ))
    do
        echo $base_name$i
        echo $base_directory_img$i.$base_extention
        echo $base_directory_attention$i.pgm
        echo $base_directory_output$i.txt
        $path_to_exe $base_directory_img$i.$base_extention $base_directory_attention$i.pgm $base_directory_output$i.txt
    done
}

calculateAttentionPoints 'AIM' 'AIM_MSR' '../../../bin/ExmplMSR' 0 175
calculateAttentionPoints 'AIM' 'AIM_WTA' '../../../bin/ExmplWTA' 0 175
calculateAttentionPoints 'GBVS' 'GBVS_MSR' '../../../bin/ExmplMSR' 0 175
calculateAttentionPoints 'GBVS' 'GBVS_WTA' '../../../bin/ExmplWTA' 0 175
calculateAttentionPoints 'SYM2D' 'SYM2D_MSR' '../../../bin/ExmplMSR' 0 175
calculateAttentionPoints 'SYM3D' 'SYM3D_MSR' '../../../bin/ExmplMSR' 0 175