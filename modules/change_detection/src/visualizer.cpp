/*
 * Visualiser3D.cpp
 *
 *  Created on: 23.1.2015
 *      Author: ivelas
 */

#include <iostream>

#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#include <v4r/change_detection/visualizer.h>
#include <v4r/change_detection/viewport_checker.h>

using namespace cv;
using namespace pcl;
using namespace Eigen;

namespace v4r {

Visualizer3D::Visualizer3D(const std::string &win_name) :
    rng(cv::theRNG()),
    color_index(0),
    viewer(new pcl::visualization::PCLVisualizer(win_name)),
    identifier(0)
{
  viewer->setBackgroundColor(255, 255, 255);
  viewer->addCoordinateSystem(0.5);
  viewer->initCameraParameters();
  viewer->setCameraPosition(5, -5, 0, 0, 0, 0);
}

Visualizer3D::~Visualizer3D() {
  close();
}

Visualizer3D& Visualizer3D::saveSnapshot(const std::string &filename) {
  viewer->saveScreenshot(filename);
  return *this;
}

Visualizer3D& Visualizer3D::addSenzor(PointXYZ position) {
  string name = "senzor";
  viewer->removeShape(name);
  viewer->addSphere(position, 0.1, 1.0, 1.0, 0, name);
  return *this;
}

Visualizer3D& Visualizer3D::keepOnlyClouds(int clouds_to_preserve) {
  vector<string> old_ids = all_identifiers;
  all_identifiers.clear();
  int preserved = 0;
  for(int i = old_ids.size() - 1; i >= 0; i--) {
    string id = old_ids[i];
    if(id.find("cloud") != string::npos) {
      if(preserved < clouds_to_preserve) {
        preserved++;
        all_identifiers.push_back(id);
      } else {
        viewer->removePointCloud(id);
      }
    } else {
      all_identifiers.push_back(id);
    }
  }
  reverse(all_identifiers.begin(), all_identifiers.end());
  return *this;
}

}
