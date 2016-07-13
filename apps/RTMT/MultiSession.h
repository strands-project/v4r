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

#ifndef _MULTI_SESSION_H
#define _MULTI_SESSION_H

#ifndef Q_MOC_RUN
#include <QObject>
#include <QThread>
#include <QMutex>
#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/shared_ptr.hpp>
#include "sensor.h"
#endif



class MultiSession : public QThread
{
  Q_OBJECT

public:
  enum Command
  {
    MULTI_SESSION_ALIGNMENT,
    MULTI_SESSION_MULTI_VIEW,
    MAX_COMMAND,
    UNDEF = MAX_COMMAND
  };

  MultiSession();
  ~MultiSession();

  void start();
  void stop();
  bool isRunning();
  void alignSequences();
  void optimizeSequences();
  void addSequences(const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &_cameras,
                    const boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > &_clouds, 
                    const std::vector<std::vector<int> > &_object_indices, const Eigen::Matrix4f &_object_base_transform=Eigen::Matrix4f::Identity());
  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &getCameras() {return cameras;}
  const boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > &getClouds() {return clouds; }
  const std::vector< cv::Mat_<unsigned char> > &getMasks(){ return masks; }
  void clear();
  bool savePointClouds(const std::string &_folder, const std::string &_modelname);
  void setUseFeatures(bool b) { use_features_ = b; }
  void setUseStablePlanes(bool b) { use_stable_planes_ = b; }

public slots:
  void object_modelling_parameter_changed(const ObjectModelling& param);

signals:
  void printStatus(const std::string &_txt);
  void finishedAlignment(bool ok);
  void update_model_cloud(const boost::shared_ptr< Sensor::AlignedPointXYZRGBVector > &_oc_cloud);
  void update_visualization();


private:

  Command cmd;
  bool m_run;

  ObjectModelling om_params;

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > cameras;
  boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > clouds;
  std::vector< cv::Mat_<unsigned char> > masks;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > output_poses;
  
  std::vector< pcl::PointCloud<pcl::Normal>::Ptr > normals;
  std::vector<std::pair<int,int> > session_ranges_;
  std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > sessions_clouds_;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > sessions_cloud_poses_;
  std::vector<std::vector<int> > sessions_cloud_indices_;

  bool use_features_;
  bool use_stable_planes_;

  double vx_size;
  double max_dist;
  int max_iterations;
  int diff_type;

  pcl::PointCloud<pcl::Normal>::Ptr big_normals;
  boost::shared_ptr< Sensor::AlignedPointXYZRGBVector > oc_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree_cloud;

  void run();

  void createMask(const std::vector<int> &indices, cv::Mat_<unsigned char> &mask, int width, int height);
  void optimizeDenseMultiview();
  void createObjectCloudFiltered();

  inline bool isnan(const Eigen::Vector3f &pt);
};

/**
 * @brief MultiSession::isnan
 * @param pt
 * @return
 */
inline bool MultiSession::isnan(const Eigen::Vector3f &pt)
{
  if (std::isnan(pt[0]) || std::isnan(pt[1]) || std::isnan(pt[2]))
    return true;
  return false;
}



#endif
