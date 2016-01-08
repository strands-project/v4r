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

#include <v4r/reconstruction/KeypointSlamRGBD2.h>
#include <boost/thread.hpp>
#include <v4r/features/FeatureDetector_K_HARRIS.h>
#include <v4r/common/impl/ScopeTime.hpp>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/reconstruction/impl/projectPointToImage.hpp>

namespace v4r
{


using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
KeypointSlamRGBD2::KeypointSlamRGBD2(const KeypointSlamRGBD2::Parameter &p)
 : param(p), view_pose(Eigen::Matrix4f::Identity()), delta_pose(Eigen::Matrix4f::Identity()), conf(0.), conf_cnt(0), pose(Eigen::Matrix4f::Identity()), new_kf_1st_frame(-1), new_kf_2nd_frame(-1)
{ 
  rad_add_keyframe_angle = param.add_keyframe_angle*M_PI/180.;
  view.reset(new ObjectView(0));
  om.reset(new KeyframeManagementRGBD2(param.om_param));
  FeatureDetector::Ptr estDesc(new FeatureDetector_KD_FAST_IMGD(param.det_param));
  FeatureDetector::Ptr det = estDesc;//(new FeatureDetector_K_HARRIS());// = estDesc;
  param.kd_param.compute_global_pose=false;
  param.kt_param.compute_global_pose=false;
  param.lk_param.compute_global_pose=false;
  kpDetector.reset(new KeypointPoseDetectorRT(param.kd_param,det,estDesc));
  kpTracker.reset(new ProjLKPoseTrackerRT(param.kt_param));
  lkTracker.reset(new LKPoseTrackerRT(param.lk_param));
}

KeypointSlamRGBD2::~KeypointSlamRGBD2()
{
}

/**
 * viewPointChange
 */
double KeypointSlamRGBD2::viewPointChange(const Eigen::Vector3f &pt, const Eigen::Matrix4f &inv_pose1, const Eigen::Matrix4f &inv_pose2)
{
  Eigen::Vector3f v1 = (inv_pose1.block<3,1>(0,3)-pt).normalized();
  Eigen::Vector3f v2 = (inv_pose2.block<3,1>(0,3)-pt).normalized();

  float a = v1.dot(v2);

  if (a>0.9999) a=0;
  else a=acos(a);

  return a;
}

/**
 * addKeyframe
 */
bool KeypointSlamRGBD2::addKeyframe(const ObjectView &view, const Eigen::Matrix4f &delta_pose, const Eigen::Matrix4f &view_pose, const Eigen::Matrix4f &pose, int width, int height, double conf)
{
  bool add_kf = false;

  if (view.cam_points.size()<4) 
  {
    //cout<<"[KeypointSlamRGBD2::addKeyframe] add keyframes no points!"<<endl;
    return true;
  }

  if (conf_cnt<param.min_conf_cnt)
  {
    //cout<<"[KeypointSlamRGBD2::addKeyframe] do not add keyframe (conf_cnt<param.min_conf_cnt)"<<endl;
    return false;
  }

  if (conf < param.add_keyframe_conf)
    return true;

  // check angle deviation
  Eigen::Matrix4f inv1, inv2;

  invPose(view_pose,inv1);
  invPose(pose, inv2);

  double angle = viewPointChange(view.center, inv1, inv2);
  //cout<<"[KeypointSlamRGBD2::addKeyframe] angle="<<angle*180./M_PI<<endl;

  if (angle > rad_add_keyframe_angle)
    return true;

  // check viewpoint overlap
  Eigen::Vector2f im_pt;
  Eigen::Vector3f pt3;
  bool have_dist = !dist_coeffs.empty();
  int cnt = 0.;

  Eigen::Matrix3f R = delta_pose.topLeftCorner<3, 3>();
  Eigen::Vector3f t = delta_pose.block<3,1>(0, 3);

  for (unsigned i=0; i<view.cam_points.size(); i++)
  {
    pt3 = R*view.cam_points[i] + t;

    if (have_dist)
      projectPointToImage(&pt3[0], intrinsic.ptr<double>(), dist_coeffs.ptr<double>(), &im_pt[0]);
    else projectPointToImage(&pt3[0], intrinsic.ptr<double>(), &im_pt[0]);

    if (im_pt[0]>=0 && im_pt[0]<width && im_pt[1]>=0 && im_pt[1]<height)
    {
      cnt++;
    }
  }
  
  //cout<<"[KeypointSlamRGBD2::addKeyframe] overlap="<<double(cnt) / double(view.cam_points.size())<<endl;
  if (double(cnt) / double(view.cam_points.size()) < param.add_keyframe_view_overlap)
    return true;

  //cout<<"[KeypointSlamRGBD2::addKeyframe] add keyframe: "<<add_kf<<endl;

  return add_kf;
}


/***************************************************************************************/

/**
 * track
 * @brief camera tracking ....
 * @param image input image
 * @param cloud input point cloud
 * @param pose estimate without confidence 0value (verification of poses is done async)
 * @return camera id
 */
//int cnt_reinit=0;
bool KeypointSlamRGBD2::track(const cv::Mat &image, const DataMatrix2D<Eigen::Vector3f> &cloud, Eigen::Matrix4f &current_pose, double &current_conf, int &cam_id)
{
  //v4r::ScopeTime t("tracking");
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else image.copyTo(im_gray);

  if (!dbg.empty()) kpTracker->dbg = dbg;
  if (!dbg.empty()) kpDetector->dbg = dbg;
  //if (!dbg.empty()) lkTracker->dbg = dbg;

  if (!om->isStarted()) om->start();

  new_kf_2nd_frame = (new_kf_1st_frame!=-1?new_kf_1st_frame:-1);
  new_kf_1st_frame = -1;

  cam_id = -1;

  // do refinement
  if (view->points.size()>=4)
  {
    im_pts.clear();

    if (param.do_inc_pyr_lk && conf > param.conf_reinit) 
      conf = lkTracker->detectIncremental(im_gray, cloud, delta_pose);
    conf = kpTracker->detect(im_gray, cloud, delta_pose);

    if (conf < param.conf_reinit)
    {
      //cnt_reinit++;
      if (!dbg.empty()) cout<<"REINIT!!!!!!!!!!"<<endl;
      conf = kpDetector->detect(im_gray, cloud, delta_pose);
      if (conf>0.001) conf = kpTracker->detect(im_gray, cloud, delta_pose);
    }
  }
  //cout<<"cnt_reinit="<<cnt_reinit<<endl;

  if (conf>=param.min_conf) conf_cnt++;
  else conf_cnt=0;

  pose = delta_pose*view_pose;
  int tracked_view_idx = view->idx;

  // check view update/ handover
  if(om->getTrackingModel(*view, view_pose, pose, (conf>param.min_conf)))
  {
    kpDetector->setModel(view);
    kpTracker->setModel(view, view_pose);
    lkTracker->setModel(view);
    
    Eigen::Matrix4f inv_view_pose;
    invPose(view_pose, inv_view_pose);
    delta_pose = pose*inv_view_pose;

    new_kf_1st_frame = tracked_view_idx;
  }

  // check add new keyframe
  if ( addKeyframe(*view, delta_pose, view_pose, pose, im_gray.cols, im_gray.rows, conf) )
  {
    kpTracker->getProjections(im_pts);
    om->addKeyframe(im_gray, cloud, (view->points.size()<4?Eigen::Matrix4f::Identity():pose), tracked_view_idx, im_pts);
  }  

  if (conf>param.conf_reinit)
    lkTracker->setLastFrame(im_gray, delta_pose);

  // add projections (and add simple loops)
  if (conf_cnt>param.min_conf_cnt) 
  {
    kpTracker->getProjections(im_pts);

    if (new_kf_1st_frame != -1)
      cam_id = om->addLinkHyp1(im_gray,cloud,pose, tracked_view_idx, im_pts, view->idx);
    else if (new_kf_2nd_frame != -1)
      cam_id = om->addLinkHyp2(im_gray,cloud,pose, new_kf_2nd_frame, tracked_view_idx, im_pts);
    else cam_id = om->addProjections(cloud, pose, tracked_view_idx, im_pts);
  }

  current_pose = pose;
  current_conf = conf;

  return (conf_cnt>param.min_conf_cnt);
}

/**
 * setKeyframe
 */
void KeypointSlamRGBD2::setKeyframe(int idx, Eigen::Matrix4f &pose)
{
  cout<<"[KeypointSlamRGBD2::setKeyframe] TODO"<<endl;
}


/**
 * setCameraParameter
 */
void KeypointSlamRGBD2::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  reset();
  dist_coeffs = cv::Mat_<double>::zeros(1,8);

  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else _intrinsic.copyTo(intrinsic);

  if (!_dist_coeffs.empty())
  {
      for (int row_id=0; row_id<_dist_coeffs.rows; row_id++)
      {
          for (int col_id=0; col_id<_dist_coeffs.cols; col_id++)
          {
              dist_coeffs(0, row_id * dist_coeffs.rows + col_id) = _dist_coeffs.at<double>(row_id, col_id);
          }
      }
  }

  kpTracker->setTargetCameraParameter(intrinsic, dist_coeffs);
  kpTracker->setSourceCameraParameter(intrinsic, dist_coeffs);
  lkTracker->setCameraParameter(intrinsic, dist_coeffs);
  
  om->reset();
  om->setCameraParameter(intrinsic, dist_coeffs);
}

/**
 * reset()
 */
void KeypointSlamRGBD2::reset()
{
  view_pose = delta_pose = Eigen::Matrix4f::Identity();

  conf = 0.;
  conf_cnt=0;
  new_kf_1st_frame = -1;
  new_kf_2nd_frame = -1;
  pose.setIdentity();

  view.reset(new ObjectView(0));

  if (om.get()!=0)
  {
    om->stop();
    om->reset();
  }
}


}












