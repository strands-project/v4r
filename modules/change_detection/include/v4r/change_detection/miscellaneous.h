/*
 * miscellaneous.h
 *
 *  Created on: 26 Aug 2015
 *      Author: martin
 */

#ifndef MISCELLANEOUS_CHANGEDET_H_
#define MISCELLANEOUS_CHANGEDET_H_

#include <pcl/point_cloud.h>
#include <pcl/common/eigen.h>

namespace v4r {

template <class PointType>
V4R_EXPORTS typename pcl::PointCloud<PointType>::Ptr downsampleCloud(
		typename pcl::PointCloud<PointType>::Ptr input,
		double resolution = 0.005f);

template <class PointType>
V4R_EXPORTS Eigen::Affine3f resetViewpoint(typename pcl::PointCloud<PointType>::Ptr input);

}

#include <v4r/change_detection/impl/miscellaneous.hpp>

#endif /* MISCELLANEOUS_CHANGEDET_H_ */
