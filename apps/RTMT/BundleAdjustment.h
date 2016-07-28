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

#ifndef _BUNDLE_ADJUSTMENT_H
#define _BUNDLE_ADJUSTMENT_H

#ifndef Q_MOC_RUN
#include <QThread>
#include <QMutex>
#include <queue>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/shared_ptr.hpp>
#include <pcl/common/transforms.h>
#include "params.h"
#include "sensor.h"
#include <v4r/keypoints/impl/Object.hpp>
#include <v4r/common/impl/SmartPtr.hpp>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/reconstruction/ProjBundleAdjuster.h>
#endif



class BundleAdjustment : public QThread
{
  Q_OBJECT

public:
  enum Command
  {
    PROJ_BA_CAM_STRUCT,
    MAX_COMMAND,
    UNDEF = MAX_COMMAND
  };

  BundleAdjustment();
  ~BundleAdjustment();

  void start();
  void stop();
  bool isRunning();

  void optimizeCamStructProj(v4r::Object::Ptr &_model, boost::shared_ptr< std::vector<Sensor::CameraLocation> > &_cam_trajectory,
                             boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > &_log_clouds,
                             boost::shared_ptr< Sensor::AlignedPointXYZRGBVector > &_oc_cloud);
  bool restoreCameras();


public slots:
  void cam_tracker_params_changed(const CamaraTrackerParameter &_cam_tracker_params);

signals:
  void printStatus(const std::string &_txt);
  void update_model_cloud(const boost::shared_ptr< Sensor::AlignedPointXYZRGBVector > &_oc_cloud);
  void update_cam_trajectory(const boost::shared_ptr< std::vector<Sensor::CameraLocation> > &_cam_trajectory);
  void update_visualization();
  void finishedOptimizeCameras(int num_cameras);

private:
  void run();
  void optimizeCamStructProj();
  void renewPrevCloud(const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &poses, const std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > &clouds);

  Command cmd;
  bool m_run;

  CamaraTrackerParameter cam_tracker_params;

  v4r::Object::Ptr model;
  boost::shared_ptr< std::vector<Sensor::CameraLocation> > cam_trajectory;
  boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > log_clouds;
  boost::shared_ptr< Sensor::AlignedPointXYZRGBVector > oc_cloud;

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > stored_cameras;
  std::vector< std::vector<double> > stored_camera_parameter;
  std::vector<v4r::GlobalPoint> stored_points;

};

#endif
