/*
 * global_nn_classifier.cpp
 *
 *  Created on: Mar 9, 2012
 *      Author: aitor
 */

#include "v4r/recognition/impl/voxel_based_correspondence_estimation.hpp"

template class v4r::VoxelBasedCorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ >;
template class v4r::VoxelBasedCorrespondenceEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB >;
template class v4r::VoxelBasedCorrespondenceEstimation<pcl::PointXYZRGBA, pcl::PointXYZRGBA >;
