/*
 * multi_pipeline_recognizer.cpp
 *
 *  Created on: Feb 25, 2013
 *      Author: aitor
 */

#include <faat_pcl/3d_rec_framework/pipeline/impl/multi_pipeline_recognizer.hpp>

template class PCL_EXPORTS faat_pcl::rec_3d_framework::MultiRecognitionPipeline<pcl::PointXYZ>;
template class PCL_EXPORTS faat_pcl::rec_3d_framework::MultiRecognitionPipeline<pcl::PointXYZRGB>;
