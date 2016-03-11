/*
 * miscellaneous.hpp
 *
 *  Created on: 26 Aug 2015
 *      Author: martin
 */

#ifndef MISCELLANEOUS_CHANGEDET_HPP_
#define MISCELLANEOUS_CHANGEDET_HPP_

#include <pcl/io/pcd_io.h>

#include <v4r/change_detection/miscellaneous.h>
#include <v4r/io/filesystem.h>

namespace v4r {

template <class PointType>
typename pcl::PointCloud<PointType>::Ptr downsampleCloud(
		typename pcl::PointCloud<PointType>::Ptr input,
        double resolution) {

    pcl::VoxelGrid<PointType> vg;
    typename pcl::PointCloud<PointType>::Ptr cloud_filtered (new pcl::PointCloud<PointType>());
    vg.setInputCloud (input);
    vg.setLeafSize (resolution, resolution, resolution);
    vg.filter (*cloud_filtered);

    return cloud_filtered;
}

template <class PointType>
Eigen::Affine3f resetViewpoint(typename pcl::PointCloud<PointType>::Ptr input) {
	Eigen::Vector3f offset(input->sensor_origin_(0),
			input->sensor_origin_(1),
			input->sensor_origin_(2));

	Eigen::Translation3f translation(offset);
	Eigen::Affine3f t(translation * input->sensor_orientation_);
	transformPointCloud (*input, *input, t);

	input->sensor_origin_ = Eigen::Vector4f::Zero();
	input->sensor_orientation_ = Eigen::Quaternionf::Identity();

	return t;
}

}

#endif /* MISCELLANEOUS_CHANGEDET_HPP_ */
