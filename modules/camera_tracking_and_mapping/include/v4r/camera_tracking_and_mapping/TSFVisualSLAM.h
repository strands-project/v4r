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

#ifndef KP_TSF_VISUAL_SLAM_HH
#define KP_TSF_VISUAL_SLAM_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <boost/shared_ptr.hpp>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/camera_tracking_and_mapping/TSFData.h>
#include <v4r/camera_tracking_and_mapping/TSFDataIntegration.hh>
#include <v4r/camera_tracking_and_mapping/TSFPoseTrackerKLT.hh>
#include <v4r/camera_tracking_and_mapping/TSFMapping.hh>
#include <v4r/core/macros.h>


namespace v4r
{



/**
 * TSFVisualSLAM
 */
class V4R_EXPORTS TSFVisualSLAM 
{
public:

  /**
   * Parameter
   */
  class Parameter
  {
  public:
    bool tsf_filtering;
    bool tsf_mapping;
    TSFDataIntegration::Parameter di_param;
    TSFPoseTrackerKLT::Parameter pt_param;
    TSFMapping::Parameter map_param;
    Parameter()
      : tsf_filtering(true), tsf_mapping(true) {}
  };


private:
  Parameter param;

  cv::Mat_<double> intrinsic;

  TSFData data;

  TSFDataIntegration tsfDataIntegration;
  TSFPoseTrackerKLT tsfPoseTracker;
  TSFMapping tsfMapping;

  cv::Mat dbg;


public:

  TSFVisualSLAM(const Parameter &p=Parameter());
  ~TSFVisualSLAM();

  void reset();
  void stop();
  bool track(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const uint64_t &timestamp, Eigen::Matrix4f &pose, double &conf_ransac_iter, double &conf_tracked_points);

  void setDebugImage(const cv::Mat &_dbg) {dbg = _dbg;}

  void getFilteredCloudNormals(pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud, Eigen::Matrix4f &pose, uint64_t &timestamp);
  void getFilteredCloudNormals(pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud, std::vector<float> &radius, Eigen::Matrix4f &pose, uint64_t &timestamp);
  void getFilteredCloudNormals(pcl::PointCloud<pcl::PointXYZRGB> &cloud, pcl::PointCloud<pcl::Normal> &normals, Eigen::Matrix4f &pose, uint64_t &timestamp);
  void getFilteredCloud(pcl::PointCloud<pcl::PointXYZRGB> &cloud, Eigen::Matrix4f &pose, uint64_t &timestamp);
  void getSurfelCloud(v4r::DataMatrix2D<Surfel> &cloud, Eigen::Matrix4f &pose, uint64_t &timestamp, bool need_normals=false);

  void setCameraParameter(const cv::Mat &_intrinsic);
  void setParameter(const Parameter &p);
  void setDetectors(const FeatureDetector::Ptr &_detector, const FeatureDetector::Ptr &_descEstimator);

  void optimizeMap();
  void transformMap(const Eigen::Matrix4f &transform) { tsfMapping.transformMap(transform); }
  void getCameraParameter(cv::Mat &_intrinsic, cv::Mat &_dist_coeffs) { tsfMapping.getCameraParameter(_intrinsic,_dist_coeffs); }


  inline const std::vector< std::vector<Eigen::Vector3d> > &getOptiPoints() const {return tsfMapping.getOptiPoints();}
  inline const std::vector<TSFFrame::Ptr> &getMap() const { return tsfMapping.getMap(); }

  typedef boost::shared_ptr< ::v4r::TSFVisualSLAM> Ptr;
  typedef boost::shared_ptr< ::v4r::TSFVisualSLAM const> ConstPtr;
};



/*************************** INLINE METHODES **************************/


} //--END--

#endif

