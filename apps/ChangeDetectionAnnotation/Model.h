/*
 * Model.h
 *
 *  Created on: 15.12.2015
 *      Author: ivelas
 */

#ifndef MODEL_H_
#define MODEL_H_

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

class Model {
public:
	typedef boost::shared_ptr<Model> Ptr;
	typedef boost::shared_ptr<const Model> ConstPtr;
	typedef std::map<std::string, Model> Db;

	std::string name;
	Cloud::Ptr cloud;

	Model(const std::string &name_) : name(name_), cloud(new Cloud) {
	}

	static void loadFrom(const std::string &dir, Model::Db &output) {
		std::vector<std::string> model_files = v4r::io::getFilesInDirectory(dir, ".*.pcd", false);
		for(size_t i = 0; i < model_files.size(); i++) {
			std::string name = model_files[i].substr(0, model_files[i].find('.'));
			Model m(name);
			const std::string fn = dir + "/" + model_files[i];
			LOG(INFO) << "Adding model " << fn << " (id " << m.name << ")";

			pcl::PointCloud<pcl::PointXYZRGBNormal> cloud_with_normals;
			pcl::io::loadPCDFile(fn, cloud_with_normals);
			pcl::copyPointCloud(cloud_with_normals, *m.cloud);

			/*pcl::visualization::PCLVisualizer vis("Model loaded");
			vis.addCoordinateSystem(0.5);
			vis.initCameraParameters();
		    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_vis(m.cloud);
		    std::string id = "cloud";
		    vis.addPointCloud<pcl::PointXYZRGB>(m.cloud, rgb_vis, id);
		    vis.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, id);
		    vis.spin();*/

		    output.insert(make_pair(name, m));
		}
	}
};


#endif /* MODEL_H_ */
