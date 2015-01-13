/*
 * GT6DOF.cpp
 *
 *  Created on: May 27, 2013
 *      Author: aitor
 */

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <QtGui>
#include "main_window.h"

//../build/bin/GT6DOF -models_dir /home/aitor/data/Mians_dataset/models_with_rhino -model_scale 0.001 -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes/ -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_copy/ -training_dir /home/aitor/data/Mians_trained_models_voxelsize_0.003/
//../build/bin/GT6DOF -models_dir /home/aitor/data/queens_dataset/pcd_models/ -model_scale 0.001 -pcd_file /home/aitor/data/queens_dataset/pcd_scenes_with_gt/ -training_dir /home/aitor/data/queens_dataset/trained_models -GT_DIR /home/aitor/data/queens_dataset/gt_or_format_copy/ -training_dir /home/aitor/data/queens_dataset/trained_models
int main (int argc, char* argv[])
{

  //create qt window
  MainWindow(argc, argv);
  return 0;
}
