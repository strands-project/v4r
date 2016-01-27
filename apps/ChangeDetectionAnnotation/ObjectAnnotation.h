/*
 * ObjectAnnotation.h
 *
 *  Created on: 15.12.2015
 *      Author: ivelas
 */

#ifndef OBJECTANNOTATION_H_
#define OBJECTANNOTATION_H_

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
#include "Model.h"
#include "View.h"

class ObjectAnnotation {
public:
	std::string filename;
	int relative_id;
	const Model *model;
	const View *view;
	Eigen::Matrix4f relative_pose;

	ObjectAnnotation(std::string filename_, int relative_id_, const Model *model_, const View *view_) :
		filename(filename_), relative_id(relative_id_), model(model_), view(view_) {
	}

	static void loadFrom(const std::string &dir, const View::Db &views, const Model::Db &models,
			std::vector<ObjectAnnotation> &output, const std::string &since_model = "") {
		std::vector<std::string> gt_files = v4r::io::getFilesInDirectory(dir, ".*.txt", false);
		for(size_t i = 0; i < gt_files.size(); i++) {
			int view_id, rel_id;
			std::string model_name;
			splitFilename(gt_files[i], view_id, model_name, rel_id);
			if(gt_files[i].find("occlusion") != std::string::npos ||			// ignore occlusion files
					gt_files[i].find("transformation") != std::string::npos ||	// ignore transformation files
					model_name < since_model) {									// skip models of previous objects (e.g. already done)
				continue;
			}
			LOG(INFO) << "Reading annotation " << dir << "/" << gt_files[i] << " (view: " << view_id << ", name: '" << model_name << "' rel_id: " << rel_id << ")";

			ObjectAnnotation ann(gt_files[i], rel_id, &models.find(model_name)->second, &views.find(view_id)->second);
			ann.readPoseFrom(dir + "/" + gt_files[i]);

			output.push_back(ann);
		}
		sort(output.begin(), output.end());
	}

	// returns format of GT filename
	// substitution order: view_id, model_name, relative_id
	static std::string splitFilename(const std::string &filename, int &view_id, std::string &object_name, int &relative_id) {
		std::vector<std::string> tokens;
		boost::split(tokens, filename, boost::is_any_of("_."));

		int to_skip = 0;
		std::string result;
		if(filename.find("cloud_") == 0) {
			to_skip = 1;
			result = "cloud_%02d_%s_%d.txt";
		} else {
			result = "%05d_%s_%d.txt";
		}

		// e.g. cloud_10_jasmine_green_tea_0.txt
		view_id = atoi(tokens[to_skip].c_str());

		std::vector<std::string>::iterator rel_id_token = tokens.begin() + tokens.size() - 2;
		relative_id = atoi(rel_id_token->c_str());

		std::vector<std::string>::iterator name_start = tokens.begin() + to_skip + 1;
		std::vector<std::string> name_tokens(name_start, rel_id_token);
		object_name = boost::join(name_tokens, "_");

		return result;
	}

	void readPoseFrom(const std::string &fn) {
		std::ifstream f(fn.c_str());
		if(!f.is_open()) {
		  std::cerr << "Unable to read matrix: " << filename << std::endl;
		  exit(1);
		}

		for(int i = 0; i < 16; i++) {
			f >> relative_pose(i/4, i%4);
		}
		f.close();
	}

	bool operator < (const ObjectAnnotation &other) const {
		if(this->model->name != other.model->name) {
			return this->model->name < other.model->name;
		} else if(this->view->id != other.view->id) {
			return this->view->id < other.view->id;
		} else {
			return this->relative_id < other.relative_id;
		}
	}

	void getVisualization(Cloud &output, bool highlight_r = true, bool highlight_g = true, bool highlight_b = true) const {
		pcl::transformPointCloud(*view->cloud, output, view->pose);
		for(Cloud::iterator pt = output.begin(); pt < output.end(); pt++) {
			pt->r /= 2;
			pt->g /= 2;
			pt->b /= 2;
		}

		Cloud model_transformed;
		pcl::transformPointCloud(*model->cloud, model_transformed, view->pose * relative_pose);
		for(Cloud::iterator pt = model_transformed.begin(); pt < model_transformed.end(); pt++) {
			if(highlight_r)
				pt->r = std::min((pt->r*1.5), 255.0);
			if(highlight_g)
				pt->g = std::min((pt->g*1.5), 255.0);
			if(highlight_b)
				pt->b = std::min((pt->b*1.5), 255.0);
		}
		output += model_transformed;
	}
};


#endif /* OBJECTANNOTATION_H_ */
