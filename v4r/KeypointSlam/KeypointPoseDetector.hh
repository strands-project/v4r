/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_KEYPOINT_POSE_DETECTOR_HH
#define KP_KEYPOINT_POSE_DETECTOR_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "v4r/KeypointBase/FeatureDetectorHeaders.hh"
#include "Object.hpp"


namespace kp
{


/**
 * KeypointPoseDetector
 */
class KeypointPoseDetector
{
public:
  class Parameter
  {
  public:
    int iterationsCount;       // 1000
    float reprojectionError;   // 3
    int minInliersCount;       // 100
    float nnr;
    Parameter(int _iterationsCount=5000, float _reprojectionError=3,
      int _minInliersCount=50, float _nnr=.9)
    : iterationsCount(_iterationsCount),
      reprojectionError(_reprojectionError), minInliersCount(_minInliersCount), 
      nnr(_nnr) {}
  };

private:
  Parameter param;

  cv::Mat_<double> dist_coeffs;
  cv::Mat_<double> intrinsic;
  
  cv::Mat_<unsigned char> im_gray;
  cv::Mat descs;
  std::vector<cv::KeyPoint> keys;
  std::vector< std::vector<cv::DMatch> > matches;
  std::vector< cv::Point2f > query_pts;
  std::vector< cv::Point3f > model_pts;
  std::vector< int> inliers;

  ObjectView::Ptr model;

  //cv::Ptr<cv::BFMatcher> matcher;
  cv::Ptr<cv::DescriptorMatcher> matcher;
  
  kp::FeatureDetector::Ptr detector;
  kp::FeatureDetector::Ptr descEstimator;

  inline void cvToEigen(const cv::Mat_<double> &R, const cv::Mat_<double> &t, Eigen::Matrix4f &pose); 



public:
  cv::Mat dbg;

  KeypointPoseDetector(const Parameter &p=Parameter(), 
    const kp::FeatureDetector::Ptr &_detector=kp::FeatureDetector::Ptr(),
    const kp::FeatureDetector::Ptr &_descEstimator=new kp::FeatureDetector_KD_FAST_IMGD(kp::FeatureDetector_KD_FAST_IMGD::Parameter(10000, 1.44, 2, 17)));
  ~KeypointPoseDetector();

  double detect(const cv::Mat &image, Eigen::Matrix4f &pose);

  void setModel(const ObjectView::Ptr &_model);

  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);

  typedef SmartPtr< ::kp::KeypointPoseDetector> Ptr;
  typedef SmartPtr< ::kp::KeypointPoseDetector const> ConstPtr;
};


/***************************** inline methods *******************************/
/**
 * cvToEigen
 */
inline void KeypointPoseDetector::cvToEigen(const cv::Mat_<double> &R, const cv::Mat_<double> &t, Eigen::Matrix4f &pose)
{
  pose.setIdentity();

  pose(0,0) = R(0,0); pose(0,1) = R(0,1); pose(0,2) = R(0,2);
  pose(1,0) = R(1,0); pose(1,1) = R(1,1); pose(1,2) = R(1,2);
  pose(2,0) = R(2,0); pose(2,1) = R(2,1); pose(2,2) = R(2,2);

  pose(0,3) = t(0,0);
  pose(1,3) = t(1,0);
  pose(2,3) = t(2,0);
}





} //--END--

#endif

