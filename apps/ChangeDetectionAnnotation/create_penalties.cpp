/*
 * create_penalties.cpp
 *
 *  Created on: 15.12.2015
 *      Author: Martin Velas (ivelas@fit.vutbr.cz)
 *
 *  Application creates "false" annotations for previously removed
 *  objects. This is a bit hack which enables evaluation of false
 *  detections in this special case - after the object removal.
 */

#include <boost/make_shared.hpp>
#include <boost/algorithm/string.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#include <v4r_config.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/io/filesystem.h>

#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdlib.h>
#include <algorithm>

#include "Types.h"
#include "Model.h"
#include "View.h"
#include "ObjectAnnotation.h"
#include "GtLinksFile.h"

using namespace std;
namespace po = boost::program_options;

typedef struct ArgumentsT {
	string model_dir;
	string sequence_dir;
	string gt_dir;
	string gt_links_file;
	string output_dir;
	bool visualization;

	ArgumentsT() :
		model_dir("/media/files/TUW_DATA/TUW_dynamic_dataset_icra15/models/"),
		sequence_dir("/media/files/TUW_DATA/tuw_dyn_data/set_00016/"),
		gt_dir("/media/files/TUW_DATA/tuw_dyn_data_gt_changes/set_00016/"),
		gt_links_file("/media/files/TUW_DATA/tuw_dyn_data_gt_changes/set_00016.changes"),
		output_dir("/media/files/TUW_DATA/tuw_dyn_data_gt_changes/penalties/set_00016"),
		visualization(false) {
	}
} Arguments;

class PenalitiesGenerator {
public:
	PenalitiesGenerator(Arguments arg) :
		gt_links_file(arg.gt_links_file),
		gt_dir(arg.gt_dir),
		output_dir(arg.output_dir),
		visualize(arg.visualization) {
	    View::loadFrom(arg.sequence_dir, views);
	    Model::loadFrom(arg.model_dir, models);
	}

	void run() {
		deleteTxtFilesInOutputDir();

	    for(int new_rel_id = 0; true; new_rel_id++) {
	    	vector<string> annotation_sequence = gt_links_file.getSequence();
	    	string ann_file = annotation_sequence.back();
			if(ann_file.empty()) {
				break;
			}
			int view_id, rel_id;
			string model_name;
			gt_filename_format = ObjectAnnotation::splitFilename(ann_file, view_id, model_name, rel_id);

			ObjectAnnotation ann(ann_file, rel_id, &models.find(model_name)->second, &views.find(view_id)->second);
			ann.readPoseFrom(gt_dir + "/" + ann_file);

			if(visualize) {
				vis_clouds.clear();
				Cloud *original_scene = new Cloud;
				pcl::transformPointCloud(*ann.model->cloud, *original_scene, ann.relative_pose);
				colorCloud(*original_scene, 255, 100, 255);
				*original_scene += *ann.view->cloud;
				vis_clouds.push_back(Cloud::Ptr(original_scene));
			}

			for(View::Db::iterator v = views.begin(); v != views.end(); v++) {
				if(v->first > view_id) {
					Eigen::Matrix4f new_pose = v->second.pose.inverse() * views.find(view_id)->second.pose * ann.relative_pose;

					string penalisation_filename = getPenalisationFilename(v->first, model_name, new_rel_id);
					ofstream penalisation_file((output_dir + "/" + penalisation_filename).c_str());
					for(int r = 0; r < 4; r++) {
						for(int c = 0; c < 4; c++) {
							penalisation_file << new_pose(r, c) << " ";
						}
					}
					penalisation_file << endl;

					string penalisation_occ_filename = getPenalisationFilename(v->first, "occlusion_" + model_name, new_rel_id);
					ofstream penalisation_occ_file((output_dir + "/" + penalisation_occ_filename).c_str());
					penalisation_occ_file << 0.0 << endl;

					LOG(INFO) << "Created " << penalisation_filename << " and " << penalisation_occ_filename;

					if(visualize) {
						Cloud *new_scene = new Cloud;
						pcl::transformPointCloud(*ann.model->cloud, *new_scene, new_pose);
						colorCloud(*new_scene, 255, 100, 255);
						*new_scene += *v->second.cloud;
						vis_clouds.push_back(Cloud::Ptr(new_scene));
					}
				}
			}
			if(visualize && vis_clouds.size() > 1) {
				cerr << "Penalties of " << model_name << endl;
				showResult();
			}
	    }
	}

protected:

	void deleteTxtFilesInOutputDir() {
		LOG(INFO) << "cleaning directory '" << output_dir << "' from all txt files.";
		vector<string> txtFiles = v4r::io::getFilesInDirectory(output_dir, ".*.txt", false);
		for(vector<string>::iterator f = txtFiles.begin(); f < txtFiles.end(); f++) {
			remove((output_dir + "/" + *f).c_str());
		}
	}

	void showResult() {
		static pcl::visualization::PCLVisualizer vis("Penalties GT");
		vis.initCameraParameters();

		size_t rows = 1;
		size_t cols = 2;
		while(rows*cols < vis_clouds.size()) {
			rows++;
			cols++;
		}

		vector<int> viewports;
		for(int i = 0; i < (int)vis_clouds.size(); i++) {
			viewports.push_back(i);
		}

		float width = 1.0/cols;
		float height = 1.0/rows;
		for(size_t i = 0; i < viewports.size(); i++) {
			int r = i/cols;
			int c = i%cols;
			vis.createViewPort(width*c, height*r, width*(c+1), height*(r+1), viewports[i]);
			vis.setBackgroundColor(0, 0, 0, viewports[i]);
			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_prev(vis_clouds[i]);
			stringstream ss;
			ss << "view_" << i;
			vis.addPointCloud<pcl::PointXYZRGB>(vis_clouds[i], handler_prev, ss.str(), viewports[i]);
		}

	    vis.spin();

	    for(size_t i = 0; i < viewports.size(); i++) {
	    	vis.removeAllPointClouds(viewports[i]);
	    }
	}

	void colorCloud(Cloud &cloud, int r, int g, int b) {
		for(Cloud::iterator pt = cloud.begin(); pt < cloud.end(); pt++) {
			pt->r = r;
			pt->g = g;
			pt->b = b;
		}
	}

	inline bool penalisationFileExists(const std::string& fn) {
		return v4r::io::existsFile(output_dir + "/" + fn);
	}

	string getPenalisationFilename(int view_id, const string &model_name, int rel_id) {
		char penalisation_filename[64];
		sprintf(penalisation_filename, gt_filename_format.c_str(), view_id, model_name.c_str(), rel_id);
		return string(penalisation_filename);
	}

private:
    View::Db views;
    Model::Db models;
    GtLinksFile gt_links_file;
    string gt_dir;
    string output_dir;
    string gt_filename_format;
    bool visualize;
    vector<Cloud::Ptr> vis_clouds;
};

int main(int argc, char *argv[]) {
	Arguments arg;
    po::options_description desc("Generator of panelization GTs after the object change for evaluation");
    desc.add_options()
            ("help,h", "produce help message")
            ("test_seq_dir,t", po::value<string>(&arg.sequence_dir)->default_value(arg.sequence_dir), "Directory of the point cloud data sequence")
            ("gt_dir,g", po::value<string>(&arg.gt_dir)->default_value(arg.gt_dir), "Directory of the GT for point cloud data sequence")
            ("gt_links_file,l", po::value<string>(&arg.gt_links_file)->default_value(arg.gt_links_file), "Output file of change annotator")
            ("visualization,v", po::bool_switch(&arg.visualization)->default_value(arg.visualization), "Turn on/off visualization")
    		("output_dir,o", po::value<string>(&arg.output_dir)->default_value(arg.output_dir), "Output directory for penalties");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return false;
    }
    try {
        po::notify(vm);
    } catch(std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        return false;
    }

    PenalitiesGenerator generator(arg);
    generator.run();
}
