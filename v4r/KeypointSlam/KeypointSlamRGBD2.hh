/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_KEYPOINT_SLAM_RGBD2_HH
#define KP_KEYPOINT_SLAM_RGBD2_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include "Object.hpp"
#include "KeyframeManagementRGBD2.hh"
#include "ProjLKPoseTrackerRT.hh"
#include "LKPoseTrackerRT.hh"
#include "KeypointPoseDetectorRT.hh"
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "v4r/KeypointTools/DataMatrix2D.hpp"
#include "v4r/KeypointBase/FeatureDetector_KD_FAST_IMGD.hh"

namespace kp
{

/**
 * KeypointSlamRGBD2
 */
class KeypointSlamRGBD2 
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
    bool close_loops;    // true
    double min_conf;     // 0.3
    int min_conf_cnt;    // 2
    double add_keyframe_conf; // 0.6
    double add_keyframe_angle; // 10 (slam 5)
                // deviation of the mean view ray (e.g. 10°, needed for reinit with keypoints)
    double add_keyframe_view_overlap;// 0.8, (slam 0.9) overlap of the keyframes (e.g. 50%) 
    KeyframeManagementRGBD2::Parameter om_param;
    KeypointPoseDetectorRT::Parameter kd_param;
    LKPoseTrackerRT::Parameter lk_param;
    ProjLKPoseTrackerRT::Parameter kt_param;
    FeatureDetector_KD_FAST_IMGD::Parameter det_param;
    Parameter( double _conf_reinit=0.05, bool _do_inc_pyr_lk=true, bool _close_loops=true,
      double _min_conf=0.3, int _min_conf_cnt=2, double _add_keyframe_conf=0.4,
      double _add_keyframe_angle=10, double _add_keyframe_view_overlap=0.8,
      const KeyframeManagementRGBD2::Parameter &_om_param = KeyframeManagementRGBD2::Parameter(),
      const KeypointPoseDetectorRT::Parameter &_kd_param = KeypointPoseDetectorRT::Parameter(),
      const LKPoseTrackerRT::Parameter &_lk_param = LKPoseTrackerRT::Parameter(),
      const ProjLKPoseTrackerRT::Parameter &_kt_param = ProjLKPoseTrackerRT::Parameter(),
      const FeatureDetector_KD_FAST_IMGD::Parameter &_det_param = FeatureDetector_KD_FAST_IMGD::Parameter(250, 1.44, 2, 17, 2) ) 
    : conf_reinit(_conf_reinit), do_inc_pyr_lk(_do_inc_pyr_lk), close_loops(_close_loops),
      min_conf(_min_conf), min_conf_cnt(_min_conf_cnt), add_keyframe_conf(_add_keyframe_conf),
      add_keyframe_angle(_add_keyframe_angle), add_keyframe_view_overlap(_add_keyframe_view_overlap),
      om_param(_om_param), kd_param(_kd_param), lk_param(_lk_param), kt_param(_kt_param), det_param(_det_param) { }
  };

  

private:
  Parameter param;
  double rad_add_keyframe_angle;

  cv::Mat_<double> dist_coeffs;
  cv::Mat_<double> intrinsic;
  
  cv::Mat_<unsigned char> im_gray;

  ObjectView::Ptr view;
  Eigen::Matrix4f view_pose, delta_pose;

  std::vector< std::pair<int,cv::Point2f> > im_pts;

  double conf;
  int conf_cnt;

  int new_kf_1st_frame, new_kf_2nd_frame;
  
  KeyframeManagementRGBD2::Ptr om;

  KeypointPoseDetectorRT::Ptr kpDetector;
  ProjLKPoseTrackerRT::Ptr kpTracker;
  LKPoseTrackerRT::Ptr lkTracker;

  bool addKeyframe(const ObjectView &view, const Eigen::Matrix4f &delta_pose, const Eigen::Matrix4f &view_pose, 
        const Eigen::Matrix4f &pose, int width, int height, double conf);
  double viewPointChange(const Eigen::Vector3f &pt, const Eigen::Matrix4f &inv_pose1, 
        const Eigen::Matrix4f &inv_pose2);


public:
  cv::Mat dbg;

  KeypointSlamRGBD2(const KeypointSlamRGBD2::Parameter &p=KeypointSlamRGBD2::Parameter());
  ~KeypointSlamRGBD2();

  bool track(const cv::Mat &image, const DataMatrix2D<Eigen::Vector3f> &cloud, Eigen::Matrix4f &pose, double &conf, int &cam_id);

  void setKeyframe(int idx, Eigen::Matrix4f &pose);
  int getKeyframe() { return (view.get()==0?-1:view->idx); }

  inline Object::Ptr &getModelPtr() { return om->getModelPtr(); }
  inline Object &getModel() { return om->getModel(); }
  
  inline void lock(){ om->lock(); }        // threaded object management, so we need to lock 
  inline void unlock() { om->unlock(); }
  void stopObjectManagement() { om->stop(); }

  void reset();

  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);
  void setMinDistAddProjections(const double &dist) { om->setMinDistAddProjections(dist); }

  typedef SmartPtr< ::kp::KeypointSlamRGBD2> Ptr;
  typedef SmartPtr< ::kp::KeypointSlamRGBD2 const> ConstPtr;
};



/*************************** INLINE METHODES **************************/

} //--END--

#endif

