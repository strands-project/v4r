/*
 * multiplane_segmentation.cpp
 *
 *  Created on: Sep 25, 2013
 *      Author: aitor
 */

#include "v4r/recognition/impl/multiplane_segmentation.hpp"

template class faat_pcl::MultiPlaneSegmentation<pcl::PointXYZ>;
template class faat_pcl::MultiPlaneSegmentation<pcl::PointXYZRGB>;
