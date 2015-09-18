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

#ifndef KP_KEYFRAME_MANAGEMENT_RGBD2_HH
#define KP_KEYFRAME_MANAGEMENT_RGBD2_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include <v4r/keypoints/impl/Object.hpp>
#include <v4r/reconstruction/ProjLKPoseTrackerRT.h>
#include <v4r/reconstruction/KeypointPoseDetectorRT.h>
#include <v4r/common/impl/SmartPtr.hpp>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/features/FeatureDetector_KD_FAST_IMGD.h>
#include <v4r/common/ZAdaptiveNormals.h>
//#include "v4r/TomGine/tgTomGineThread.h"
#include <v4r/common/impl/RandomNumbers.hpp>
#include <v4r/core/macros.h>

namespace v4r
{

/**
 * KeyframeManagementRGBD2
 */
class V4R_EXPORTS KeyframeManagementRGBD2
{
public:

  /**
   * Parameter
   */
  class Parameter
  {
  public:
    unsigned min_model_points;
    double max_dist_tracking_view; // depens on the scene size e.g. 1m
    int min_not_reliable_poses;
    float inl_dist_px;             // inlier dist for global point tracks
    double min_dist_add_proj;
    double min_conf;
    double dist_err_loop;          // 0.02 tests the error deviation of 2 subsequent frames 
                                   // ... it's not the loop closure error!
    FeatureDetector_KD_FAST_IMGD::Parameter det_param; // (300,1.44,2,17,2) slam 200,1.44,3,17,3
    ZAdaptiveNormals::Parameter n_param;
    KeypointPoseDetectorRT::Parameter kd_param;
    ProjLKPoseTrackerRT::Parameter kt_param;
    Parameter(unsigned _min_model_points=50, double _max_dist_tracking_view=2., 
      int _min_not_reliable_poses=5, float _inl_dist_px=2, 
      double _min_dist_add_proj=0.02, double _min_conf=.2, double _dist_err_loop=0.02,
      const FeatureDetector_KD_FAST_IMGD::Parameter &_det_param= FeatureDetector_KD_FAST_IMGD::Parameter(300,1.44,3,17,3),
      const ZAdaptiveNormals::Parameter &_n_param= ZAdaptiveNormals::Parameter(0.02,5,true,0.005125,0.003),
      const KeypointPoseDetectorRT::Parameter &_kd_param = KeypointPoseDetectorRT::Parameter(),
      const ProjLKPoseTrackerRT::Parameter &_kt_param= ProjLKPoseTrackerRT::Parameter() )
    : min_model_points(_min_model_points), max_dist_tracking_view(_max_dist_tracking_view),
      min_not_reliable_poses(_min_not_reliable_poses), inl_dist_px(_inl_dist_px),
      min_dist_add_proj(_min_dist_add_proj), min_conf(_min_conf), dist_err_loop(_dist_err_loop),
      det_param(_det_param), n_param(_n_param), kd_param(_kd_param), kt_param(_kt_param) {}
  };

  /**
   * Shared memory
   */
  class V4R_EXPORTS Shm
  {
  public:
    boost::mutex mtx_shm;
    Eigen::Matrix4f pose;
    cv::Mat_<unsigned char> image;
    DataMatrix2D<Eigen::Vector3f> cloud;
  
    int view_idx;
    std::vector< std::pair<int,cv::Point2f> > im_pts;

    unsigned nb_add;
    bool process_view;

    inline void lock() { mtx_shm.lock(); }
    inline void unlock() { mtx_shm.unlock(); }
    inline void copyTo(Shm &data) {
      data.pose = pose;
      image.copyTo(data.image);
      data.cloud = cloud;
      data.view_idx = view_idx;
      data.im_pts = im_pts;
    }

    Shm() : nb_add(0), process_view(false) {}
  };

 

private:
  Parameter param;

  cv::Mat_<double> dist_coeffs;
  cv::Mat_<double> intrinsic;

  // ---- dbg draw ----
  //SmartPtr<TomGine::tgTomGineThread> tgwin2;
  //std::vector<Eigen::Vector3f> cams;
  // ---- end dbg ----

  bool run, have_thread;
  unsigned nb_add;
  double sqr_max_dist_tracking_view;
  double sqr_dist_err_loop;
  int cnt_not_reliable_pose;

  double sqr_min_dist_add_proj; 
  Eigen::Matrix4f inv_last_add_proj_pose;
  Eigen::Matrix4f last_reliable_pose;

  Shm shm;
  Shm local_data;

  // create view links (loops)
  bool loop_in_progress;
  int have_loop_data;
  int last_view;
  int new_view;
  Eigen::Matrix4f last_pose[2];
  Eigen::Matrix4f new_pose[2];
  cv::Mat_<unsigned char> loop_image[2];
  DataMatrix2D<Eigen::Vector3f> loop_cloud[2];
  int cam_ids[2];

  ObjectView::Ptr view;
  Object::Ptr model;

  boost::thread th_obectmanagement;

  FeatureDetector::Ptr det;
  FeatureDetector::Ptr estDesc;
  ZAdaptiveNormals::Ptr nest;
  ProjLKPoseTrackerRT::Ptr kpTracker;
  KeypointPoseDetectorRT::Ptr kpDetector;

  std::vector< std::pair<int,cv::Point2f> > im_pts;

  void operate();

  double viewPointChange(const Eigen::Vector3f &pt, const Eigen::Matrix4f &inv_pose1, const Eigen::Matrix4f &inv_pose2);
  double cameraRotationZ(const Eigen::Matrix4f &inv_pose1, const Eigen::Matrix4f &inv_pose2);
  void getPoints3D(const DataMatrix2D<Eigen::Vector3f> &cloud, 
          const std::vector< std::pair<int,cv::Point2f> > &im_pts, 
          std::vector<Eigen::Vector3f> &points);
  bool createView(Shm &data, ObjectView::Ptr &view_ptr);
  int getGlobalCorrespondences(const std::vector<unsigned> glob_indices,  
          const std::vector< std::pair<int,cv::Point2f> > &im_pts, 
          std::vector<cv::KeyPoint> &keys, std::vector<unsigned> &points,
          std::vector<cv::Point2f> &im_points);
  int selectGuidedRandom(const Eigen::Matrix4f &pose);
  bool closeLoops();




public:
  cv::Mat dbg;

  KeyframeManagementRGBD2(const Parameter &p=Parameter());
  ~KeyframeManagementRGBD2();

  void start();
  void stop();

  inline bool isStarted() {return have_thread;}

  inline Object::Ptr &getModelPtr() { return model; }
  inline Object &getModel() { return *model; }

  inline void lock(){ shm.lock(); }        // threaded object management, so we need to lock 
  inline void unlock() { shm.unlock(); }

  void addKeyframe(const cv::Mat &image, const DataMatrix2D<Eigen::Vector3f> &cloud, 
        const Eigen::Matrix4f &pose, int view_idx, 
        const std::vector< std::pair<int,cv::Point2f> > &im_pts);
  bool getTrackingModel(ObjectView &view, Eigen::Matrix4f &view_pose, const Eigen::Matrix4f &current_pose, bool is_reliable_pose);

  int addProjections(const DataMatrix2D<Eigen::Vector3f> &cloud, const Eigen::Matrix4f &pose, int view_idx, const std::vector< std::pair<int,cv::Point2f> > &im_pts);

  int addLinkHyp1(const cv::Mat &image, const DataMatrix2D<Eigen::Vector3f> &cloud, 
        const Eigen::Matrix4f &pose, int last_view_idx, 
        const std::vector< std::pair<int,cv::Point2f> > &im_pts, int new_view_idx);
  int addLinkHyp2(const cv::Mat &image, const DataMatrix2D<Eigen::Vector3f> &cloud, 
        const Eigen::Matrix4f &pose, int last_view_idx, int new_view_idx, 
        const std::vector< std::pair<int,cv::Point2f> > &im_pts);

  void reset();
  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);
  void setMinDistAddProjections(const double &dist);

  typedef SmartPtr< ::v4r::KeyframeManagementRGBD2> Ptr;
  typedef SmartPtr< ::v4r::KeyframeManagementRGBD2 const> ConstPtr;
};



/*************************** INLINE METHODES **************************/

} //--END--

#endif

