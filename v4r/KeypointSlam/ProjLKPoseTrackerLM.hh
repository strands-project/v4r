/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_PROJ_LK_POSE_TRACKER_LM_HH
#define KP_PROJ_LK_POSE_TRACKER_LM_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "Object.hpp"
#include "v4r/KeypointBase/RefineProjectedPointLocationLK.hh"
#ifndef KP_NO_CERES_AVAILABLE
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#endif


namespace kp
{


/**
 * ProjLKPoseTrackerLM
 */
class ProjLKPoseTrackerLM
{
public:
  class Parameter
  {
  public:
    double inl_dist;
    double eta_ransac;               // eta for pose ransac
    unsigned max_rand_trials;        // max. number of trials for pose ransac
    bool use_robust_loss;
    double loss_scale;
    bool use_ransac;
    RefineProjectedPointLocationLK::Parameter plk_param;
    Parameter(double _inl_dist=2, double _eta_ransac=0.01, unsigned _max_rand_trials=1000,
      int _use_robust_loss=true, double _loss_scale=1., bool _use_ransac=false,
      const RefineProjectedPointLocationLK::Parameter &_plk_param = RefineProjectedPointLocationLK::Parameter())
    : inl_dist(_inl_dist), eta_ransac(_eta_ransac), max_rand_trials(_max_rand_trials),
      use_robust_loss(_use_robust_loss), loss_scale(_loss_scale), use_ransac(_use_ransac),
      plk_param(_plk_param) {}
  };

private:
  Parameter param;

  float sqr_inl_dist;

  cv::Mat_<double> src_dist_coeffs, tgt_dist_coeffs;
  cv::Mat_<double> src_intrinsic, tgt_intrinsic;
  std::vector<double> lm_intrinsics;
  
  cv::Mat_<unsigned char> im_gray;
  std::vector< cv::Point2f > im_points;
  std::vector< int > inliers, converged;
  Eigen::Matrix3f pose_R;
  Eigen::Vector3f pose_t;

  ObjectView::Ptr model;

  RefineProjectedPointLocationLK::Ptr plk;

  void getRandIdx(int size, int num, std::vector<int> &idx);
  unsigned countInliers(const std::vector<Eigen::Vector3d> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4d &pose);
  void getInliers(const std::vector<Eigen::Vector3d> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4d &pose, std::vector<int> &inliers);
  void ransacPoseLM(const std::vector<Eigen::Vector3d> &points, const std::vector<cv::Point2f> &im_points, Eigen::Matrix4d &pose, std::vector<int> &inliers);
  void optimizePoseLM(const std::vector<Eigen::Vector3d> &points, const std::vector<int> &pt_indices, const std::vector<cv::Point2f> &im_points, const std::vector<int> &im_indices, Eigen::Matrix4d &pose);
  void optimizePoseRobustLossLM(const std::vector<Eigen::Vector3d> &points, const std::vector<int> &pt_indices, const std::vector<cv::Point2f> &im_points, const std::vector<int> &im_indices, Eigen::Matrix4d &pose);

  inline bool contains(const std::vector<int> &idx, int num);

  



public:
  cv::Mat dbg;

  ProjLKPoseTrackerLM(const Parameter &p=Parameter());
  ~ProjLKPoseTrackerLM();

  double detect(const cv::Mat &image, Eigen::Matrix4f &pose);

  void setModel(const ObjectView::Ptr &_model, const Eigen::Matrix4f &_pose);
  void getProjections(std::vector< std::pair<int,cv::Point2f> > &im_pts);

  void setSourceCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);
  void setTargetCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);

  typedef SmartPtr< ::kp::ProjLKPoseTrackerLM> Ptr;
  typedef SmartPtr< ::kp::ProjLKPoseTrackerLM const> ConstPtr;
};


/***************************** inline methods *******************************/

inline bool ProjLKPoseTrackerLM::contains(const std::vector<int> &idx, int num)
{
  for (unsigned i=0; i<idx.size(); i++)
    if (idx[i]==num)
      return true;
  return false;
}




} //--END--

#endif

