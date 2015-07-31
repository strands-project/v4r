/*
 * fast_icp_with_gc.cpp
 *
 *  Created on: Sep 8, 2013
 *      Author: aitor
 */

#include <v4r/common/fast_icp_with_gc.h>
#include <v4r/common/impl/fast_icp_with_gc.hpp>

#include <v4r/core/macros.h>

template class V4R_EXPORTS v4r::common::ICPNode<pcl::PointXYZRGB>;
template class V4R_EXPORTS v4r::common::FastIterativeClosestPointWithGC<pcl::PointXYZRGB>;

