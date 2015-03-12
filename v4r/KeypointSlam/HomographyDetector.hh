/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_HOMOGRAPHY_DETECTOR_HH
#define KP_HOMOGRAPHY_DETECTOR_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "v4r/KeypointTools/ImageTransformRANSAC.hh"
#include "v4r/KeypointBase/FeatureDetectorHeaders.hh"
#include "Object.hpp"


namespace kp
{


/**
 * HomographyDetector
 */
class HomographyDetector
{
public:
  class Parameter
  {
  public:
    float nnr;
    int outlier_rejection_method; // 1..affine model, 2..full homography
    ImageTransformRANSAC::Parameter imr_param;
    Parameter(float _nnr=.9, int _outlier_rejection_method=1,
      const ImageTransformRANSAC::Parameter &_imr_param=ImageTransformRANSAC::Parameter(20,0.01,500))
    : nnr(_nnr), outlier_rejection_method(_outlier_rejection_method),
      imr_param(_imr_param) {}
  };

private:
  Parameter param;

  cv::Mat_<unsigned char> im_gray;
  cv::Mat descs;
  std::vector<cv::KeyPoint> keys;
  std::vector< std::vector<cv::DMatch> > matches;
  std::vector<Eigen::Vector2f,Eigen::aligned_allocator<Eigen::Vector2f> > pts0, pts1;
  std::vector< int > inliers;
  std::vector<int> lt;
  std::vector<int> inls;

  ObjectView::Ptr model;


  ImageTransformRANSAC::Ptr imr;

  cv::Ptr<cv::DescriptorMatcher> matcher;

  kp::FeatureDetector::Ptr detector;
  kp::FeatureDetector::Ptr descEstimator;


public:
  cv::Mat dbg;

  HomographyDetector(const Parameter &p=Parameter(), 
    const kp::FeatureDetector::Ptr &_detector=kp::FeatureDetector::Ptr(),
    const kp::FeatureDetector::Ptr &_descEstimator=new kp::FeatureDetector_KD_FAST_IMGD(kp::FeatureDetector_KD_FAST_IMGD::Parameter(10000, 1.44, 2, 17)));
  ~HomographyDetector();

  double detect(const cv::Mat &image, Eigen::Matrix3f &transform);

  void setModel(const ObjectView::Ptr &_model);
  void getProjections(std::vector< std::pair<int,cv::Point2f> > &im_pts);

  typedef SmartPtr< ::kp::HomographyDetector> Ptr;
  typedef SmartPtr< ::kp::HomographyDetector const> ConstPtr;
};


/***************************** inline methods *******************************/



} //--END--

#endif

