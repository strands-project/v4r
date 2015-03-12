/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_PROJ_LK_POSE_TRACKER_RT_HH
#define KP_PROJ_LK_POSE_TRACKER_RT_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Dense>
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "Object.hpp"
#include "v4r/KeypointBase/RefineProjectedPointLocationLK.hh"
#include "v4r/KeypointTools/RigidTransformationRANSAC.hh"
#include "v4r/KeypointTools/DataMatrix2D.hpp"


namespace kp
{


/**
 * ProjLKPoseTrackerRT
 */
class ProjLKPoseTrackerRT
{
public:
  class Parameter
  {
  public:
    bool compute_global_pose;
    RigidTransformationRANSAC::Parameter rt_param; // 0.04 (slam: 0.08)
    RefineProjectedPointLocationLK::Parameter plk_param;
    Parameter(bool _compute_global_pose=true,
      const RigidTransformationRANSAC::Parameter &_rt_param=RigidTransformationRANSAC::Parameter(0.04),
      const RefineProjectedPointLocationLK::Parameter &_plk_param = RefineProjectedPointLocationLK::Parameter())
    : compute_global_pose(_compute_global_pose),
      rt_param(_rt_param), 
      plk_param(_plk_param) {}
  };

private:
  Parameter param;

  float sqr_inl_dist;

  cv::Mat_<double> src_dist_coeffs, tgt_dist_coeffs;
  cv::Mat_<double> src_intrinsic, tgt_intrinsic;
  
  cv::Mat_<unsigned char> im_gray;
  std::vector< cv::Point2f > im_points;
  std::vector< int > inliers, converged;
  std::vector<Eigen::Vector3f> model_pts;
  std::vector<Eigen::Vector3f> query_pts;


  ObjectView::Ptr model;

  RefineProjectedPointLocationLK::Ptr plk;
  RigidTransformationRANSAC::Ptr rt;


  



public:
  cv::Mat dbg;

  ProjLKPoseTrackerRT(const Parameter &p=Parameter());
  ~ProjLKPoseTrackerRT();

  double detect(const cv::Mat &image, const DataMatrix2D<Eigen::Vector3f> &cloud, Eigen::Matrix4f &pose);

  ObjectView::Ptr getModel() { return model; }

  void setModel(const ObjectView::Ptr &_model, const Eigen::Matrix4f &_pose);
  void getProjections(std::vector< std::pair<int,cv::Point2f> > &im_pts);

  void setSourceCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);
  void setTargetCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);

  typedef SmartPtr< ::kp::ProjLKPoseTrackerRT> Ptr;
  typedef SmartPtr< ::kp::ProjLKPoseTrackerRT const> ConstPtr;
};


/***************************** inline methods *******************************/





} //--END--

#endif

