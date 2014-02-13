/*
 * multiplane_segmentation.cpp
 *
 *  Created on: Sep 25, 2013
 *      Author: aitor
 */

#include <faat_pcl/3d_rec_framework/segmentation/impl/multiplane_segmentation.hpp>

template class faat_pcl::MultiPlaneSegmentation<pcl::PointXYZ>;
template class faat_pcl::MultiPlaneSegmentation<pcl::PointXYZRGB>;
