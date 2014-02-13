/*
 * fast_icp_with_gc.cpp
 *
 *  Created on: Sep 8, 2013
 *      Author: aitor
 */

#include <faat_pcl/registration/fast_icp_with_gc.h>
#include <faat_pcl/registration/impl/fast_icp_with_gc.hpp>

template class faat_pcl::registration::ICPNode<pcl::PointXYZRGB>;
template class faat_pcl::registration::FastIterativeClosestPointWithGC<pcl::PointXYZRGB>;

