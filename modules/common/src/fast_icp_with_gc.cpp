/*
 * fast_icp_with_gc.cpp
 *
 *  Created on: Sep 8, 2013
 *      Author: aitor
 */

#include <v4r/common/fast_icp_with_gc.h>
#include <v4r/common/impl/fast_icp_with_gc.hpp>

template class v4r::common::ICPNode<pcl::PointXYZRGB>;
template class v4r::common::FastIterativeClosestPointWithGC<pcl::PointXYZRGB>;

