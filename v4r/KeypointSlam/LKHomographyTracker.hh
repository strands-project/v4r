/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_LK_HOMOGRAPHY_TRACKER_HH
#define KP_LK_HOMOGRAPHY_TRACKER_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Dense>
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "v4r/KeypointTools/ImageTransformRANSAC.hh"
#include "Object.hpp"


namespace kp
{


/**
 * LKHomographyTracker
 */
class LKHomographyTracker
{
public:
  class Parameter
  {
  public:
    cv::Size win_size;
    int max_level;
    cv::TermCriteria termcrit;
    float max_error;
    int outlier_rejection_method; //0..no outlier rejection, 1..affine model, 2..full homography
    ImageTransformRANSAC::Parameter imr_param;
    Parameter(const cv::Size &_win_size=cv::Size(21,21), int _max_level=2,
      const cv::TermCriteria &_termcrit=cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03),
      float _max_error=100, int _outlier_rejection_method=1,
      const ImageTransformRANSAC::Parameter &_imr_param=ImageTransformRANSAC::Parameter(20,0.01,500))
    : win_size(_win_size), max_level(_max_level),
      termcrit(_termcrit),
      max_error(_max_error), outlier_rejection_method(_outlier_rejection_method),
      imr_param(_imr_param) {}
  };

private:
  Parameter param;

  cv::Mat_<unsigned char> im_gray;
  std::vector< cv::Point2f > im_points0, im_points1;
  std::vector<Eigen::Vector2f,Eigen::aligned_allocator<Eigen::Vector2f> > pts0, pts1;
  std::vector< int > inliers;

  std::vector<unsigned char> status;
  std::vector<float> error;


  ObjectView::Ptr model;


  ImageTransformRANSAC::Ptr imr;



public:
  cv::Mat dbg;

  LKHomographyTracker(const Parameter &p=Parameter());
  ~LKHomographyTracker();

  double detect(const cv::Mat &image, Eigen::Matrix3f &transform);

  void setModel(const ObjectView::Ptr &_model);
  ObjectView::Ptr getModel() { return model; }
  void getProjections(std::vector< std::pair<int,cv::Point2f> > &im_pts);

  typedef SmartPtr< ::kp::LKHomographyTracker> Ptr;
  typedef SmartPtr< ::kp::LKHomographyTracker const> ConstPtr;
};


/***************************** inline methods *******************************/



} //--END--

#endif

