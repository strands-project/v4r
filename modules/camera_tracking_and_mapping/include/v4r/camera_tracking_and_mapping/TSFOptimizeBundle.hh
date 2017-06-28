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

#ifndef KP_TSF_OPTIMIZE_BUNDLE_HH
#define KP_TSF_OPTIMIZE_BUNDLE_HH

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <v4r/camera_tracking_and_mapping/TSFFrame.hh>
#include <v4r/core/macros.h>


namespace v4r
{



/**
 * TSFOptimizeBundle
 */
class V4R_EXPORTS TSFOptimizeBundle 
{
public:

  /**
   * Parameter
   */
  class Parameter
  {
  public:
    double depth_error_scale;
    bool use_robust_loss;
    double loss_scale;
    bool optimize_focal_length;
    bool optimize_principal_point;
    bool optimize_radial_k1;
    bool optimize_radial_k2;
    bool optimize_radial_k3;
    bool optimize_tangential_p1;
    bool optimize_tangential_p2;
    bool optimize_delta_cloud_rgb_pose_global;
    bool optimize_delta_cloud_rgb_pose;
    Parameter()
      : depth_error_scale(100), use_robust_loss(true), loss_scale(2.),
        optimize_focal_length(true), optimize_principal_point(true),
        optimize_radial_k1(false), optimize_radial_k2(false), optimize_radial_k3(false),
        optimize_tangential_p1(false), optimize_tangential_p2(false),
        optimize_delta_cloud_rgb_pose_global(false), optimize_delta_cloud_rgb_pose(false) {}
  };

private:
  Parameter param;

  cv::Mat_<double> dist_coeffs;
  cv::Mat_<double> intrinsic;
  std::vector<double> lm_intrinsics;

  Eigen::Matrix<double, 6, 1> delta_pose;
  std::vector<Eigen::Matrix<double, 6, 1> > poses_Rt;
  std::vector<Eigen::Matrix<double, 6, 1> > poses_Rt_RGB;
  std::vector< std::vector<Eigen::Vector3d> > points3d;

  std::vector<int> const_intrinsics;
  bool const_all_intrinsics;

  void convertPosesToRt(const std::vector<TSFFrame::Ptr> &map);
  void convertPosesFromRt(std::vector<TSFFrame::Ptr> &map);
  void convertPosesFromRtRGB(std::vector<TSFFrame::Ptr> &map);
  void convertPoints(const std::vector<TSFFrame::Ptr> &map);
  void optimizePoses(std::vector<TSFFrame::Ptr> &map);
  void optimizeCloudPosesRGBPoses(std::vector<TSFFrame::Ptr> &map);
  void optimizeCloudPosesDeltaRGBPose(std::vector<TSFFrame::Ptr> &map);


public:
  TSFOptimizeBundle( const Parameter &p=Parameter());
  ~TSFOptimizeBundle();

  void optimize(std::vector<TSFFrame::Ptr> &map);

  void getCameraParameter(cv::Mat &_intrinsic, cv::Mat &_dist_coeffs);

  inline const std::vector< std::vector<Eigen::Vector3d> > &getOptiPoints() const {return points3d;}

  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);
  void setParameter(const Parameter &p=Parameter());
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

