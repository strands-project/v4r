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
 * @author Johann Prankl
 *
 */

#ifndef KP_GLOBAL_CLOUD_FILTERING_HH
#define KP_GLOBAL_CLOUD_FILTERING_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/octree/octree_pointcloud.h>
#include <v4r/camera_tracking_and_mapping/OctreeVoxelCentroidContainerXYZRGBNormal.hpp>
#include <boost/smart_ptr.hpp>
#include <v4r/keypoints/impl/triple.hpp>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/camera_tracking_and_mapping/TSFData.h>
#include <v4r/camera_tracking_and_mapping/TSFFrame.hh>
#include <v4r/core/macros.h>


namespace v4r
{



/**
 * TSFGlobalCloudFiltering
 */
class V4R_EXPORTS TSFGlobalCloudFiltering 
{
public:

  /**
   * Parameter
   */
  class Parameter
  {
  public:
    float thr_angle;
    float max_weight;
    float sigma_pts;
    float sigma_depth;
    float z_cut_off_integration;
    float voxel_size;
    float max_dist_integration;
    Parameter()
      : thr_angle(80), max_weight(20), sigma_pts(10), sigma_depth(0.008), z_cut_off_integration(0.02), voxel_size(0.001), max_dist_integration(7.) {}
  };

 

private:
  Parameter param;

  float cos_thr_angle;
  float neg_inv_sqr_sigma_pts;

  cv::Mat_<double> intrinsic;

  std::vector<cv::Vec4i> npat;
  std::vector<float> exp_error_lookup;

  std::vector< cv::Mat_<float> > reliability;

  pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGBNormal,pcl::octree::OctreeVoxelCentroidContainerXYZRGBNormal<pcl::PointXYZRGBNormal> >::Ptr octree;
  typedef pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGBNormal,pcl::octree::OctreeVoxelCentroidContainerXYZRGBNormal<pcl::PointXYZRGBNormal> >::AlignedPointTVector AlignedPointXYZRGBNormalVector;
  boost::shared_ptr< AlignedPointXYZRGBNormalVector > oc_cloud;

//  void integrateData(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Eigen::Matrix4f &pose, const Eigen::Matrix4f &filt_pose, v4r::DataMatrix2D<TSFData::Surfel> &filt_cloud);
  void computeReliability(const std::vector<TSFFrame::Ptr> &frames);
  void maxReliabilityIndexing(const std::vector<TSFFrame::Ptr> &frames);
  void getMaxPoints(const std::vector<TSFFrame::Ptr> &frames, pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud);
  void computeNormals(v4r::DataMatrix2D<v4r::Surfel> &sf_cloud);

  inline float sqr(const float &d) {return d*d;}


public:
  cv::Mat dbg;

  TSFGlobalCloudFiltering(const Parameter &p=Parameter());
  ~TSFGlobalCloudFiltering();

  void start();
  void stop();

  void computeRadius(v4r::DataMatrix2D<v4r::Surfel> &sf_cloud);

  void filter(const std::vector<TSFFrame::Ptr> &frames, pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud, const Eigen::Matrix4f &base_transform=Eigen::Matrix4f::Identity());

  void setCameraParameter(const cv::Mat &_intrinsic);
  void setParameter(const Parameter &p);

  typedef boost::shared_ptr< ::v4r::TSFGlobalCloudFiltering> Ptr;
  typedef boost::shared_ptr< ::v4r::TSFGlobalCloudFiltering const> ConstPtr;
};



/*************************** INLINE METHODES **************************/


} //--END--

#endif

