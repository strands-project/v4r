/*
 * Object.h
 *
 *  Created on: 12 Aug 2015
 *      Author: martin
 */

#ifndef OBJECT_DETECTION_H_
#define OBJECT_DETECTION_H_

#include <iostream>
#include <fstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <boost/filesystem.hpp>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>

namespace v4r {

template <class PointType>
class ObjectDetection {

public:
	ObjectDetection(const std::string &claz_, const int id_,
			typename pcl::PointCloud<PointType>::ConstPtr cloud_, const Eigen::Affine3f &pose_) :
		claz(claz_), id(id_), cloud(cloud_), pose(pose_) {
	}

	const std::string& getClass() const {
		return claz;
	}

	typename pcl::PointCloud<PointType>::Ptr getCloud(bool transformed = true) const {
		typename pcl::PointCloud<PointType>::Ptr result(
			new pcl::PointCloud<PointType>());
		if(transformed) {
			pcl::transformPointCloud(*cloud, *result, pose);
		} else {
			*result += *cloud;
		}
		return result;
	}

	int getId() const {
		return id;
	}

	const Eigen::Affine3f& getPose() const {
		return pose;
	}

private:

	std::string claz;
	int id;
	typename pcl::PointCloud<PointType>::ConstPtr cloud;
	Eigen::Affine3f pose;
};

}

#endif /* OBJECT_DETECTION_H_ */
