/**
 * $Id$
 * 
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl, Aitor Aldoma
 *
 */

#ifndef _FUNCTIONS_H
#define _FUNCTIONS_H

#include <QThread>
#include <QMutex>
#include <queue>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/shared_ptr.hpp>
#include <pcl/common/transforms.h>
#include "params.h"
#include "v4r/KeypointSlam/Object.hpp"
#include "v4r/KeypointTools/DataMatrix2D.hpp"
#include "v4r/KeypointConversions/convertCloud.hpp"
#include "v4r/KeypointSlam/ArticulatedObject.hh"
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "v4r/KeypointTools/RigidTransformationRANSAC.hh"
#include "v4r/KeypointTools/projectPointToImage.hpp"
#include "v4r/KeypointTools/invPose.hpp"
#include "v4r/KeypointConversions/convertImage.hpp"
#include "v4r/KeypointBase/FeatureDetectorHeaders.hh"
#include "v4r/KeypointSlam/io.hh"
#include "v4r/KeypointTools/ZAdaptiveNormals.hh"



class Functions : public QThread
{
  Q_OBJECT

public:
  enum Command
  {
    STORE_TRACKING_MODEL,
    MAX_COMMAND,
    UNDEF = MAX_COMMAND
  };

  Functions();
  ~Functions();

  void start();
  void stop();
  bool isRunning();

  void storeTrackingModel(const std::string &_folder,
                          const std::string &_objectname,
                          const std::vector<Eigen::Matrix4f> &_cameras,
                          const boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > &_clouds,
                          const std::vector< cv::Mat_<unsigned char> > &_masks);


public slots:
  void cam_params_changed(const RGBDCameraParameter &_cam_params);

signals:
  void printStatus(const std::string &_txt);
  void finishedStoreTrackingModel();

private:
  void run();

  void createTrackingModel();
  void saveTrackingModel();
  void addObjectView(const kp::DataMatrix2D<Eigen::Vector3f> &cloud, const kp::DataMatrix2D<Eigen::Vector3f> &normals,
                     const cv::Mat_<unsigned char> &im, const cv::Mat_<unsigned char> &mask,
                     const Eigen::Matrix4f &pose, kp::ArticulatedObject &model);
  void detectCoordinateSystem(Eigen::Matrix4f &pose);

  Command cmd;
  bool m_run;

  cv::Mat_<double> intrinsic;
  cv::Mat_<double> dist_coeffs;

  // data to compute the tracking model
  std::string folder;
  std::string objectname;
  std::vector<Eigen::Matrix4f> cameras;
  boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > clouds;
  std::vector< cv::Mat_<unsigned char> > masks;

  kp::ArticulatedObject::Ptr model;
  kp::FeatureDetector::Ptr keyDet;
  kp::FeatureDetector::Ptr keyDesc;

};

#endif
