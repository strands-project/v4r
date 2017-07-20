/******************************************************************************
 * Copyright (c) 2016 Martin Velas, Thomas Faeulhammer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/


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
