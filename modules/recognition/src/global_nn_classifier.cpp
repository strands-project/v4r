/*
 * global_nn_classifier.cpp
 *
 *  Created on: Mar 9, 2012
 *      Author: aitor
 */

#include <v4r/recognition/impl/global_nn_classifier.hpp>
#include <v4r/recognition//metrics.h>

//Instantiation
template class V4R_EXPORTS v4r::GlobalNNPipeline<flann::L1, pcl::PointXYZ, pcl::VFHSignature308>;
template class V4R_EXPORTS v4r::GlobalNNPipeline<v4r::Metrics::HistIntersectionUnionDistance, pcl::PointXYZ, pcl::VFHSignature308>;
template class V4R_EXPORTS v4r::GlobalNNPipeline<flann::L1, pcl::PointXYZ, pcl::ESFSignature640>;

template class V4R_EXPORTS v4r::GlobalClassifier<pcl::PointXYZ>;
