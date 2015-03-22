/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_OBJECT_TRACKER_MONO_HH
#define KP_OBJECT_TRACKER_MONO_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "Object.hpp"
#include "ProjLKPoseTrackerR2.hh"
#include "LKPoseTracker.hh"
#include "KeypointPoseDetector.hh"
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "v4r/KeypointBase/FeatureDetector_KD_FAST_IMGD.hh"
#include "v4r/KeypointBase/CodebookMatcher.hh"
#include "KeypointObjectRecognizerR2.hh"

namespace kp
{

/**
 * ObjectTrackerMono
 */
class ObjectTrackerMono 
{
public:

  /**
   * Parameter
   */
  class Parameter
  {
  public:
    double conf_reinit;  // 0.05
    bool do_inc_pyr_lk;  // true
    double min_conf;     // 0.3
    int min_conf_cnt;    // 2
    KeypointPoseDetector::Parameter kd_param;
    LKPoseTracker::Parameter lk_param;
    ProjLKPoseTrackerR2::Parameter kt_param;
    FeatureDetector_KD_FAST_IMGD::Parameter det_param;
    KeypointObjectRecognizerR2::Parameter or_param;
    Parameter( double _conf_reinit=0.05, bool _do_inc_pyr_lk=true,
      double _min_conf=0.3, int _min_conf_cnt=3,
      const KeypointPoseDetector::Parameter &_kd_param = KeypointPoseDetector::Parameter(),
      const LKPoseTracker::Parameter &_lk_param = LKPoseTracker::Parameter(),
      const ProjLKPoseTrackerR2::Parameter &_kt_param = ProjLKPoseTrackerR2::Parameter(),
      const FeatureDetector_KD_FAST_IMGD::Parameter &_det_param = FeatureDetector_KD_FAST_IMGD::Parameter(250, 1.44, 2, 17, 2),
      const KeypointObjectRecognizerR2::Parameter &_or_param=KeypointObjectRecognizerR2::Parameter())
    : conf_reinit(_conf_reinit), do_inc_pyr_lk(_do_inc_pyr_lk),
      min_conf(_min_conf), min_conf_cnt(_min_conf_cnt),
      kd_param(_kd_param), lk_param(_lk_param), kt_param(_kt_param), det_param(_det_param), or_param(_or_param) { }
  };

  

private:
  Parameter param;

  cv::Mat_<double> dist_coeffs;
  cv::Mat_<double> intrinsic;
  
  cv::Mat_<unsigned char> im_gray;

  ObjectView::Ptr view;
  Object::Ptr model;

  double conf;
  int conf_cnt;
  int not_conf_cnt;

  KeypointPoseDetector::Ptr kpDetector;
  ProjLKPoseTrackerR2::Ptr projTracker;
  LKPoseTracker::Ptr lkTracker;
  KeypointObjectRecognizerR2::Ptr kpRecognizer;

//  CodebookMatcher::Ptr cbMatcher;

  double viewPointChange(const Eigen::Vector3f &pt, const Eigen::Matrix4f &inv_pose1,
                         const Eigen::Matrix4f &inv_pose2);
  double reinit(const cv::Mat_<unsigned char> &im, Eigen::Matrix4f &pose, ObjectView::Ptr &view);
  void updateView(const Eigen::Matrix4f &pose, const Object &model, ObjectView::Ptr &view);


public:
  cv::Mat dbg;
  CodebookMatcher::Ptr cbMatcher;

  ObjectTrackerMono(const ObjectTrackerMono::Parameter &p=ObjectTrackerMono::Parameter());
  ~ObjectTrackerMono();

  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);
  void setObjectCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);
  void setObjectModel(const Object::Ptr &_model);
  void reset();

  bool track(const cv::Mat &image, Eigen::Matrix4f &pose, double &out_conf);

  const Object::Ptr &getModelPtr() { return model; }
  inline const Object &getModel() { return *model; }

  typedef SmartPtr< ::kp::ObjectTrackerMono> Ptr;
  typedef SmartPtr< ::kp::ObjectTrackerMono const> ConstPtr;
};



/*************************** INLINE METHODES **************************/

} //--END--

#endif

