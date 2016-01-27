/*
 * View.h
 *
 *  Created on: 15.12.2015
 *      Author: ivelas
 */

#ifndef VIEW_H_
#define VIEW_H_

#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#include <v4r/io/filesystem.h>

#include <glog/logging.h>

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdlib.h>
#include <algorithm>

#include "Types.h"

class View {
public:
	typedef boost::shared_ptr<View> Ptr;
	typedef boost::shared_ptr<const View> ConstPtr;
	typedef std::map<int, View> Db;

	int id;
	Cloud::Ptr cloud;
	Eigen::Matrix4f pose;

	View(int id_) : id(id_), cloud(new Cloud) {
	}

	static void loadFrom(const std::string &dir, View::Db &output) {
		std::vector<std::string> view_files = v4r::io::getFilesInDirectory(dir, ".*.pcd", false);
		std::sort(view_files.begin(), view_files.end());
		for (size_t i = 0; i < view_files.size(); i++) {
			View v(getId(view_files[i]));
			const std::string fn = dir + "/" + view_files[i];
			LOG(INFO) << "Adding view " << fn << " (id " << v.id << ")";
			pcl::io::loadPCDFile(fn, *v.cloud);

			v.pose = v4r::RotTrans2Mat4f(v.cloud->sensor_orientation_, v.cloud->sensor_origin_);
			// reset view point otherwise pcl visualization is potentially messed up
			Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
			v.cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
			v.cloud->sensor_origin_ = zero_origin;

			output.insert(std::make_pair(v.id, v));
		}
	}

	static int getId(const std::string &filename) {
		size_t find_pos = filename.find("cloud_");
		if(find_pos == std::string::npos) {
			return atoi(filename.c_str());
		} else {
			return atoi(filename.substr(6).c_str());
		}
	}
};

#endif /* VIEW_H_ */
