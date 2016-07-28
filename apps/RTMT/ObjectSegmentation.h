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

#ifndef _OBJECT_SEGMENTATION_H
#define _OBJECT_SEGMENTATION_H

#ifndef Q_MOC_RUN
#include <QObject>
#include <QThread>
#include <QMutex>
#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/shared_ptr.hpp>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/io/pcd_io.h>
#include "OctreeVoxelCentroidContainerXYZRGB.hpp"
#include <v4r/common/ZAdaptiveNormals.h>
#include <v4r/keypoints/ClusterNormalsToPlanes.h>
#include <v4r/common/impl//DataMatrix2D.hpp>
#include <v4r/keypoints/impl/triple.hpp>
#include "params.h"
#include "sensor.h"
#endif


class ObjectSegmentation : public QThread
{
  Q_OBJECT

public:
  enum Command
  {
    FINISH_OBJECT_MODELLING,
    OPTIMIZE_MULTIVIEW,
    MAX_COMMAND,
    UNDEF = MAX_COMMAND
  };

  ObjectSegmentation();
  ~ObjectSegmentation();

  void start();
  void stop();
  bool isRunning();

  void setData(const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &_cameras, const boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > &_clouds);
  const std::vector< cv::Mat_<unsigned char> > &getMasks() {return masks; }
  void drawObjectCloud();
  void storeMasks(const std::string &_folder);
  bool storePointCloudModel(const std::string &_folder);
  bool savePointClouds(const std::string &_folder, const std::string &_modelname);
  void activateROI(bool enable);
  void finishSegmentation();
  bool optimizeMultiview();
  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &getCameras() { return cameras; }
  const std::vector<std::vector<int> > &getObjectIndices() {return indices; }
  const Eigen::Matrix4f &getObjectBaseTransform() {return object_base_transform; }

public slots:
  void segment_image(int x, int y);
  void set_image(int idx);
  void cam_params_changed(const RGBDCameraParameter &_cam_params);
  void segmentation_parameter_changed(const SegmentationParameter& param);
  void object_modelling_parameter_changed(const ObjectModelling& param);
  void set_roi(const Eigen::Vector3f &_bb_min, const Eigen::Vector3f &_bb_max, const Eigen::Matrix4f &_roi_pose);
  void set_segmentation_params(bool use_roi_segm, const double &offs, bool _use_dense_mv, const double &_edge_radius_px);

signals:
  void new_image(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud, const cv::Mat_<cv::Vec3b> &image);
  void update_model_cloud(const boost::shared_ptr< Sensor::AlignedPointXYZRGBVector > &_oc_cloud);
  void printStatus(const std::string &_txt);
  void update_visualization();
  void set_object_base_transform(const Eigen::Matrix4f &_object_base_transform);
  void finishedObjectSegmentation();


private:

  Command cmd;
  bool m_run;

  cv::Mat_<double> intrinsic;
  cv::Mat_<double> dist_coeffs;

  SegmentationParameter seg_params;
  ObjectModelling om_params;

  int image_idx;
  bool first_click;

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > cameras;
  boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > clouds;
  std::vector< pcl::PointCloud<pcl::Normal>::Ptr > normals;
  std::vector< cv::Mat_<int> > labels;
  std::vector< cv::Mat_<unsigned char> > masks;
  std::vector< std::vector<v4r::ClusterNormalsToPlanes::Plane::Ptr> > planes;

  cv::Mat_<cv::Vec3b> image;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud1, tmp_cloud2;

  boost::shared_ptr< Sensor::AlignedPointXYZRGBVector > oc_cloud;

  pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGB,pcl::octree::OctreeVoxelCentroidContainerXYZRGB<pcl::PointXYZRGB> >::Ptr octree;

  pcl::PointCloud<pcl::Normal>::Ptr big_normals;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree_cloud;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > inv_poses;
  std::vector<std::vector<int> > indices;

  bool have_roi;
  Eigen::Matrix4f roi_pose;
  Eigen::Vector3f bb_min, bb_max;
  Eigen::Matrix4f object_base_transform;

  bool use_roi_segmentation;
  double roi_offs;
  bool use_dense_mv;

  double vx_size;
  double max_dist;
  int max_iterations;
  int diff_type;

  v4r::ZAdaptiveNormals::Ptr nest;
  v4r::ClusterNormalsToPlanes::Ptr pest;


  void run();

  void postProcessingSegmentation(bool do_mv=false);
  void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat_<cv::Vec3b> &image);
  void getInplaneTransform(const Eigen::Vector3f &pt, const Eigen::Vector3f &normal, Eigen::Matrix4f &pose);
  void getMaskedImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, cv::Mat_<cv::Vec3b> &image, float alpha=.5);
  void createObjectCloud();
  void createObjectCloudFiltered();
  void segmentObject(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, pcl::PointCloud<pcl::PointXYZRGB> &seg_cloud);
  void detectCoordinateSystem(Eigen::Matrix4f &pose);
  void createMaskFromROI(const v4r::DataMatrix2D<Eigen::Vector3f> &cloud, cv::Mat_<unsigned char> &mask, const Eigen::Matrix4f &object_base_transform,
                         const Eigen::Vector3f &bb_min, const Eigen::Vector3f &bb_max, const double &roi_offs);


  inline bool isnan(const Eigen::Vector3f &pt);
};

/**
 * @brief ObjectSegmentation::isnan
 * @param pt
 * @return
 */
inline bool ObjectSegmentation::isnan(const Eigen::Vector3f &pt)
{
  if (std::isnan(pt[0]) || std::isnan(pt[1]) || std::isnan(pt[2]))
    return true;
  return false;
}



#endif
