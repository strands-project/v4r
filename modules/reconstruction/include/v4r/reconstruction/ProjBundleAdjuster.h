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

#ifndef KP_PROJ_BUNDLE_ADJUSTER_HH
#define KP_PROJ_BUNDLE_ADJUSTER_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#ifndef KP_NO_CERES_AVAILABLE
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#endif

#include <v4r/core/macros.h>
#include <v4r/keypoints/impl/Object.hpp>

namespace v4r
{

class V4R_EXPORTS ProjBundleAdjuster
{
public:
  class Parameter
  {
  public:
    bool optimize_intrinsic;
    bool optimize_dist_coeffs;
    bool use_depth_prior;
    double depth_error_weight;
    double depth_inl_dist;
    double depth_cut_off;
    Parameter(bool _optimize_intrinsic=false, bool _optimize_dist_coeffs=false, 
      bool _use_depth_prior=true, double _depth_error_weight=100., 
      double _depth_inl_dist=0.02, double _depth_cut_off=2.) 
    : optimize_intrinsic(_optimize_intrinsic), optimize_dist_coeffs(_optimize_dist_coeffs),
      use_depth_prior(_use_depth_prior), depth_error_weight(_depth_error_weight), 
      depth_inl_dist(_depth_inl_dist), depth_cut_off(_depth_cut_off)  {}
  };
  class Camera
  {
  public:
    int idx;
    Eigen::Matrix<double, 6, 1> pose_Rt;
  };

private:
  Parameter param;

  double sqr_depth_inl_dist;

  std::vector<Camera> cameras;

  void getCameras(const Object &data, std::vector<Camera> &cameras);
  void setCameras(const std::vector<Camera> &cameras, Object &data);
  void bundle(Object &data, std::vector<Camera> &cameras);


  inline void getR(const Eigen::Matrix4f &pose, Eigen::Matrix3d &R);
  inline void getT(const Eigen::Matrix4f &pose, Eigen::Vector3d &t);
  inline void setPose(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, Eigen::Matrix4f &pose);
  inline bool isnan(const Eigen::Vector3f &pt);


public:
  cv::Mat dbg;

  ProjBundleAdjuster(const Parameter &p=Parameter());
  ~ProjBundleAdjuster();

  void optimize(Object &data);

  typedef SmartPtr< ::v4r::ProjBundleAdjuster> Ptr;
  typedef SmartPtr< ::v4r::ProjBundleAdjuster const> ConstPtr;
};



/*************************** INLINE METHODES **************************/

inline void ProjBundleAdjuster::getR(const Eigen::Matrix4f &pose, Eigen::Matrix3d &R)
{
  R(0,0) = pose(0,0); R(0,1) = pose(0,1); R(0,2) = pose(0,2);
  R(1,0) = pose(1,0); R(1,1) = pose(1,1); R(1,2) = pose(1,2);
  R(2,0) = pose(2,0); R(2,1) = pose(2,1); R(2,2) = pose(2,2);
}

inline void ProjBundleAdjuster::getT(const Eigen::Matrix4f &pose, Eigen::Vector3d &t)
{
  t(0) = pose(0,3);
  t(1) = pose(1,3);
  t(2) = pose(2,3);
}

inline void ProjBundleAdjuster::setPose(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, Eigen::Matrix4f &pose)
{
  pose.setIdentity();
  pose(0,0) = R(0,0); pose(0,1) = R(0,1); pose(0,2) = R(0,2); pose(0,3) = t(0);
  pose(1,0) = R(1,0); pose(1,1) = R(1,1); pose(1,2) = R(1,2); pose(1,3) = t(1);
  pose(2,0) = R(2,0); pose(2,1) = R(2,1); pose(2,2) = R(2,2); pose(2,3) = t(2);
}

inline bool ProjBundleAdjuster::isnan(const Eigen::Vector3f &pt)
{
  if (std::isnan(pt[0]) || std::isnan(pt[1]) || std::isnan(pt[2]))
    return true;
  return false;
}

} //--END--

#endif

