/*
 * main.cpp
 *
 * Multiview Object Recognition
 * Sept 2013
 * Author: Thomas Fäulhammer
 *
 *
 *Command: rosrun visual_recognizer visual_recognizer_node _models_dir_sift:=/home/thomas/willow_dataset/willow_dense_shot30_reduced/ _training_dir:=/home/thomas/willow_dataset/trained_test/ _model_path:=/home/thomas/willow_dataset/models_ml_new/ _icp_iterations:=0 _do_ourcvfh:=false _topic:=/camera/depth_registered/points _visualize_output:=true
 *
 */

#include <iostream>
#include "myGraphClasses.h"
  
int main (int argc, char **argv)
{ 
  multiviewGraph myGraph;
  myGraph.init(argc, argv);
  ROS_INFO("Starting rec");
  //myGraph.recognize();
  return 0;
}
