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

#ifndef _GRAB_PCD_SENSOR_H_
#define _GRAB_PCD_SENSOR_H_

#ifndef Q_MOC_RUN
#include <QThread>
#include <QMutex>
#include <queue>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <boost/shared_ptr.hpp>
#include <pcl/common/transforms.h>
#include "params.h"
#include <v4r/keypoints/impl/Object.hpp>
#include <v4r/reconstruction/KeypointSlamRGBD2.h>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/keypoints/impl/triple.hpp>
#include <v4r/common/convertCloud.h>
#include <v4r/keypoints/ClusterNormalsToPlanes.h>
#include "OctreeVoxelCentroidContainerXYZRGB.hpp"
#endif


class Sensor : public QThread
{
  Q_OBJECT

public:
  class CameraLocation
  {
  public:
    int idx;
    int type;
    Eigen::Vector3f pt;
    Eigen::Vector3f vr;
    CameraLocation() {}
    CameraLocation(int _idx, int _type, const Eigen::Vector3f &_pt, const Eigen::Vector3f &_vr) : idx(_idx), type(_type), pt(_pt), vr(_vr) {}
  };

  Sensor();
  ~Sensor();

  typedef pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGB,pcl::octree::OctreeVoxelCentroidContainerXYZRGB<pcl::PointXYZRGB> >::AlignedPointTVector AlignedPointXYZRGBVector;

  void start(int cam_id=0);
  void stop();
  void startTracker(int cam_id);
  void stopTracker();
  bool isRunning();
  void reset();
  void storeKeyframes(const std::string &_folder);
  void storeCurrentFrame(const std::string& _folder);
  void storePointCloudModel(const std::string &_folder);
  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &getCameras();
  void showDepthMask(bool _draw_depth_mask);
  void selectROI(int _seed_x, int _seed_y);
  void activateROI(bool enable);

  v4r::Object::Ptr &getModel() {return camtracker->getModelPtr(); }
  boost::shared_ptr< std::vector<CameraLocation> > &getTrajectory() {return cam_trajectory;}
  boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > &getClouds() { return log_clouds; }
  boost::shared_ptr< AlignedPointXYZRGBVector > &getAlignedCloud() {return oc_cloud;}




public slots:
  void cam_params_changed(const RGBDCameraParameter &_cam_params);
  void cam_tracker_params_changed(const CamaraTrackerParameter &_cam_tracker_params);
  void bundle_adjustment_parameter_changed(const BundleAdjustmentParameter& param);
  void select_roi(int x, int y);
  void set_roi_params(const double &_bbox_scale_xy, const double &_bbox_scale_height, const double &_seg_offs);

signals:
  void new_image(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud, const cv::Mat_<cv::Vec3b> &image);
  void new_pose(const Eigen::Matrix4f &_pose);
  void update_model_cloud(const boost::shared_ptr< Sensor::AlignedPointXYZRGBVector > &_oc_cloud);
  void update_cam_trajectory(const boost::shared_ptr< std::vector<Sensor::CameraLocation> > &_cam_trajectory);
  void update_visualization();
  void printStatus(const std::string &_txt);
  void finishedOptimizeCameras(int num_cameras);
  void update_boundingbox(const std::vector<Eigen::Vector3f> &edges, const Eigen::Matrix4f &pose);
  void set_roi(const Eigen::Vector3f &_bb_min, const Eigen::Vector3f &_bb_max, const Eigen::Matrix4f &_roi_pose);


private:
  void run();

  void CallbackCloud (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &_cloud);
  int selectFrames(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, int cam_id, const Eigen::Matrix4f &pose,
                   std::vector<CameraLocation> &traj);
  void renewPrevCloud(const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &poses, const std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > &clouds);
  void drawConfidenceBar(cv::Mat &im, const double &conf);
  void drawDepthMask(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &im);
  void detectROI(const v4r::DataMatrix2D<Eigen::Vector3f> &cloud);
  void getInplaneTransform(const Eigen::Vector3f &pt, const Eigen::Vector3f &normal, Eigen::Matrix4f &pose);
  void getBoundingBox(const v4r::DataMatrix2D<Eigen::Vector3f> &cloud, const std::vector<int> &indices, const Eigen::Matrix4f &pose,
                      std::vector<Eigen::Vector3f> &bbox, Eigen::Vector3f &bb_min, Eigen::Vector3f &bb_max);
  void maskCloud(const Eigen::Matrix4f &pose, const Eigen::Vector3f &bb_min, const Eigen::Vector3f &bb_max, v4r::DataMatrix2D<Eigen::Vector3f> &cloud);

  inline bool isNaN(const Eigen::Vector3f &pt);


  // status
  bool m_run;
  bool m_run_tracker;
  int m_cam_id;
  bool m_draw_mask;
  bool m_select_roi;
  bool m_activate_roi;

  int roi_seed_x, roi_seed_y;
  v4r::ClusterNormalsToPlanes::Plane plane;

  // parameter
  unsigned u_idle_time;
  unsigned max_queue_size;

  RGBDCameraParameter cam_params;
  CamaraTrackerParameter cam_tracker_params;
  BundleAdjustmentParameter ba_params;

  // data logging
  double cos_min_delta_angle, sqr_min_cam_distance;
  boost::shared_ptr< std::vector<CameraLocation> > cam_trajectory;
  boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > log_clouds;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > cameras;

  // preview
  double prev_voxel_size, prev_filter_z;
  std::vector<int> indices;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud, tmp_cloud2;
  boost::shared_ptr< AlignedPointXYZRGBVector > oc_cloud;

  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGB,pcl::octree::OctreeVoxelCentroidContainerXYZRGB<pcl::PointXYZRGB> >::Ptr octree;

  // camera tracker
  QMutex shm_mutex;
  std::queue< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > shm_clouds;

  QMutex cloud_mutex;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  v4r::DataMatrix2D<Eigen::Vector3f> kp_cloud;
  cv::Mat_<cv::Vec3b> image;

  Eigen::Matrix4f pose;
  double conf;
  int cam_id;

  v4r::KeypointSlamRGBD2::Parameter ct_param;
  v4r::KeypointSlamRGBD2::Ptr camtracker;

  boost::shared_ptr<pcl::Grabber> interface;

  //bounding box filter
  double bbox_scale_xy, bbox_scale_height, seg_offs;
  Eigen::Vector3f bb_min, bb_max;
  std::vector<Eigen::Vector3f> edges;
  Eigen::Matrix4f bbox_base_transform;

  v4r::ClusterNormalsToPlanes::Ptr pest;
  v4r::ZAdaptiveNormals::Ptr nest;
};

/**
 * @brief Sensor::isNaN
 * @param pt
 * @return
 */
inline bool Sensor::isNaN(const Eigen::Vector3f &pt)
{
  if (std::isnan(pt[0]) || std::isnan(pt[1]) || std::isnan(pt[2]))
    return true;
  return false;
}

#endif // _GRAB_PCD_SENSOR_H_
