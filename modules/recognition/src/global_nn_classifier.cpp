/*
 * global_nn_classifier.cpp
 *
 *  Created on: Mar 9, 2012
 *      Author: aitor
 */

#include <v4r/recognition/impl/global_nn_classifier.hpp>
#include <v4r/recognition/metrics.h>

//Instantiation
template class V4R_EXPORTS v4r::GlobalNNClassifier<flann::L1, pcl::PointXYZ>;
template class V4R_EXPORTS v4r::GlobalNNClassifier<v4r::Metrics::HistIntersectionUnionDistance, pcl::PointXYZ>;
