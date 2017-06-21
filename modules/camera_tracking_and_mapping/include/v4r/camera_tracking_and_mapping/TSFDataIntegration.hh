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

#ifndef KP_TSF_DATA_INTEGRATION_HH
#define KP_TSF_DATA_INTEGRATION_HH

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
#include <boost/shared_ptr.hpp>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/camera_tracking_and_mapping/TSFData.h>
#include <v4r/camera_tracking_and_mapping/OcclusionClustering.hh>
#include <v4r/core/macros.h>


namespace v4r
{



/**
 * TSFDataIntegration
 */
class V4R_EXPORTS TSFDataIntegration 
{
public:

  /**
   * Parameter
   */
  class Parameter
  {
  public:
    int max_integration_frames;
    float sigma_depth;
    bool filter_occlusions;
    double diff_cam_distance_map; // minimum distance the camera must move to select a keyframe
    double diff_delta_angle_map;  // or minimum angle a camera need to rotate to be selected
    int min_frames_integrated;
    Parameter()
      : max_integration_frames(20), sigma_depth(0.008), filter_occlusions(true), diff_cam_distance_map(0.5), diff_delta_angle_map(7.), min_frames_integrated(10) {}
  };

 

private:
  Parameter param;

  double sqr_diff_cam_distance_map;
  double cos_diff_delta_angle_map;

  static std::vector<cv::Vec4i> npat;

  cv::Mat_<double> intrinsic;

  bool run, have_thread;

  boost::thread th_obectmanagement;
  boost::thread th_init;

  std::vector<float> exp_error_lookup;

  TSFData *data;

  cv::Mat_<float> depth_norm;
  cv::Mat_<float> depth_weight;
  cv::Mat_<float> tmp_z;
  cv::Mat_<float> nan_z;

  Eigen::Matrix4f inv_pose0, inv_pose1;

  cv::Mat_<unsigned char> occ_mask;

  OcclusionClustering occ;

  void operate();

  bool selectFrame(const Eigen::Matrix4f &pose0, const Eigen::Matrix4f &pose1);
  void integrateData(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Eigen::Matrix4f &pose, const Eigen::Matrix4f &filt_pose, v4r::DataMatrix2D<Surfel> &filt_cloud);
  inline float sqr(const float &d) {return d*d;}


public:
  cv::Mat dbg;

  TSFDataIntegration(const Parameter &p=Parameter());
  ~TSFDataIntegration();

  void start();
  void stop();

  inline bool isStarted() {return have_thread;}

  inline void lock(){ data->lock(); }        // threaded object management, so we need to lock
  inline void unlock() { data->unlock(); }

  void reset();
  void setData(TSFData *_data) { data = _data; }

  void initCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, v4r::DataMatrix2D<Surfel> &sf_cloud);

  static void computeRadius(v4r::DataMatrix2D<Surfel> &sf_cloud, const cv::Mat_<double> &intrinsic);
  static void computeNormals(v4r::DataMatrix2D<Surfel> &sf_cloud, int nb_dist=1);

  void setCameraParameter(const cv::Mat &_intrinsic);
  void setParameter(const Parameter &p);

  typedef boost::shared_ptr< ::v4r::TSFDataIntegration> Ptr;
  typedef boost::shared_ptr< ::v4r::TSFDataIntegration const> ConstPtr;
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

