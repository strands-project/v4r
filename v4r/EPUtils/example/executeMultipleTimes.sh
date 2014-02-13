#!/bin/bash
function drawAttentionPoints {

    base_dir='/media/Documents/research/db/GRAPHICON2013/TOSD/test/'
    base_name='test'

    base_directory_img=$base_dir'rgb/'$base_name
    base_extention='ppm'
    base_directory_attention=$base_dir$1'/'$base_name
    mkdir $base_dir$1'_images/'
    base_directory_output=$base_dir$1'_images/'$base_name

    path_to_exe=$2 #'../../../bin/ExmplDrawPoints'

    start_idx=$3 #0
    end_idx=$4 #131

    for (( i=$start_idx; i<=$end_idx; i++ )) #131
    do
        echo $base_name$i
        $path_to_exe $base_directory_attention$i.txt $base_directory_img$i.$base_extention $base_directory_output$i.ppm

        echo $base_directory_img$i.$base_extention
        echo $base_directory_attention$img.pgm
        echo $base_directory_output$i.txt
    done
}

drawAttentionPoints 'AIM_MSR' '../../../bin/ExmplDrawPoints' 0 131
drawAttentionPoints 'AIM_WTA' '../../../bin/ExmplDrawPoints' 0 131
drawAttentionPoints 'GBVS_MSR' '../../../bin/ExmplDrawPoints' 0 131
drawAttentionPoints 'GBVS_WTA' '../../../bin/ExmplDrawPoints' 0 131
drawAttentionPoints 'SYM2D_MSR' '../../../bin/ExmplDrawPoints' 0 131
drawAttentionPoints 'SYM2D_WTA' '../../../bin/ExmplDrawPoints' 0 131
drawAttentionPoints 'SYM3D_MSR' '../../../bin/ExmplDrawPoints' 0 131
drawAttentionPoints 'SYM3D_WTA' '../../../bin/ExmplDrawPoints' 0 131
drawAttentionPoints 'SYM3D_TJ' '../../../bin/ExmplDrawPoints' 0 131