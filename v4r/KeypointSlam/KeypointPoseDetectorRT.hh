/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_KEYPOINT_POSE_DETECTOR_RT_HH
#define KP_KEYPOINT_POSE_DETECTOR_RT_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "v4r/KeypointBase/FeatureDetectorHeaders.hh"
#include "v4r/KeypointTools/RigidTransformationRANSAC.hh"
#include "v4r/KeypointTools/DataMatrix2D.hpp"
#include "Object.hpp"


namespace kp
{


/**
 * KeypointPoseDetectorRT
 */
class KeypointPoseDetectorRT
{
public:
  class Parameter
  {
  public:
    float nnr;
    bool compute_global_pose;
    RigidTransformationRANSAC::Parameter rt_param; // 0.01 (slam: 0.03)
    Parameter(float _nnr=.9, bool _compute_global_pose=true,
      const RigidTransformationRANSAC::Parameter &_rt_param=RigidTransformationRANSAC::Parameter(0.01))
    : nnr(_nnr), compute_global_pose(_compute_global_pose),
      rt_param(_rt_param) {}
  };

private:
  Parameter param;

  cv::Mat_<unsigned char> im_gray;
  cv::Mat descs;
  std::vector<cv::KeyPoint> keys;
  std::vector< std::vector<cv::DMatch> > matches;
  std::vector< Eigen::Vector3f > query_pts;
  std::vector< Eigen::Vector3f > model_pts;
  std::vector< int> inliers;

  ObjectView::Ptr model;

  //cv::Ptr<cv::BFMatcher> matcher;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  RigidTransformationRANSAC::Ptr rt;
  
  kp::FeatureDetector::Ptr detector;
  kp::FeatureDetector::Ptr descEstimator;




public:
  cv::Mat dbg;

  KeypointPoseDetectorRT(const Parameter &p=Parameter(), 
    const kp::FeatureDetector::Ptr &_detector=kp::FeatureDetector::Ptr(),
    const kp::FeatureDetector::Ptr &_descEstimator=new kp::FeatureDetector_KD_FAST_IMGD(kp::FeatureDetector_KD_FAST_IMGD::Parameter(1000, 1.44, 2, 17)));
  ~KeypointPoseDetectorRT();

  double detect(const cv::Mat &image, const kp::DataMatrix2D<Eigen::Vector3f> &cloud, Eigen::Matrix4f &pose);

  void setModel(const ObjectView::Ptr &_model);


  typedef SmartPtr< ::kp::KeypointPoseDetectorRT> Ptr;
  typedef SmartPtr< ::kp::KeypointPoseDetectorRT const> ConstPtr;
};


/***************************** inline methods *******************************/






} //--END--

#endif

