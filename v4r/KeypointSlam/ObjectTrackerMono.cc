/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 *
 * TODO: We should store the camera parameter of the learning camera!!!!!!
 */

#include "ObjectTrackerMono.hh"
#include <boost/thread.hpp>
#include "v4r/KeypointBase/FeatureDetector_K_HARRIS.hh"
#include "v4r/KeypointTools/ScopeTime.hpp"
#include "v4r/KeypointTools/invPose.hpp"
#include "v4r/KeypointTools/projectPointToImage.hpp"



namespace kp
{


using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
ObjectTrackerMono::ObjectTrackerMono(const ObjectTrackerMono::Parameter &p)
 : param(p), conf(0.), conf_cnt(0), not_conf_cnt(1000)
{ 
  view.reset(new ObjectView(0));
  FeatureDetector::Ptr estDesc(new FeatureDetector_KD_FAST_IMGD(param.det_param));
  FeatureDetector::Ptr det = estDesc;//(new FeatureDetector_K_HARRIS());// = estDesc;
  kpDetector.reset(new KeypointPoseDetector(param.kd_param,det,estDesc));
  projTracker.reset(new ProjLKPoseTrackerR2(param.kt_param));
  lkTracker.reset(new LKPoseTracker(param.lk_param));
}

ObjectTrackerMono::~ObjectTrackerMono()
{
}

/**
 * viewPointChange
 */
double ObjectTrackerMono::viewPointChange(const Eigen::Vector3f &pt, const Eigen::Matrix4f &inv_pose1, const Eigen::Matrix4f &inv_pose2)
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
//bool ObjectTrackerMono::addKeyframe(const ObjectView &view, const Eigen::Matrix4f &delta_pose, const Eigen::Matrix4f &view_pose, const Eigen::Matrix4f &pose, int width, int height, double conf)
//{
//  bool add_kf = false;

//  if (view.cam_points.size()<4)
//  {
//    //cout<<"[ObjectTrackerMono::addKeyframe] add keyframes no points!"<<endl;
//    return true;
//  }

//  if (conf_cnt<param.min_conf_cnt)
//  {
//    //cout<<"[ObjectTrackerMono::addKeyframe] do not add keyframe (conf_cnt<param.min_conf_cnt)"<<endl;
//    return false;
//  }

//  if (conf < param.add_keyframe_conf)
//    return true;

//  // check angle deviation
//  Eigen::Matrix4f inv1, inv2;

//  invPose(view_pose,inv1);
//  invPose(pose, inv2);

//  double angle = viewPointChange(view.center, inv1, inv2);
//  //cout<<"[ObjectTrackerMono::addKeyframe] angle="<<angle*180./M_PI<<endl;

//  if (angle > rad_add_keyframe_angle)
//    return true;

//  // check viewpoint overlap
//  Eigen::Vector2f im_pt;
//  Eigen::Vector3f pt3;
//  bool have_dist = !dist_coeffs.empty();
//  int cnt = 0.;

//  Eigen::Matrix3f R = delta_pose.topLeftCorner<3, 3>();
//  Eigen::Vector3f t = delta_pose.block<3,1>(0, 3);

//  for (unsigned i=0; i<view.cam_points.size(); i++)
//  {
//    pt3 = R*view.cam_points[i] + t;

//    if (have_dist)
//      projectPointToImage(&pt3[0], intrinsic.ptr<double>(), dist_coeffs.ptr<double>(), &im_pt[0]);
//    else projectPointToImage(&pt3[0], intrinsic.ptr<double>(), &im_pt[0]);

//    if (im_pt[0]>=0 && im_pt[0]<width && im_pt[1]>=0 && im_pt[1]<height)
//    {
//      cnt++;
//    }
//  }
  
//  //cout<<"[ObjectTrackerMono::addKeyframe] overlap="<<double(cnt) / double(view.cam_points.size())<<endl;
//  if (double(cnt) / double(view.cam_points.size()) < param.add_keyframe_view_overlap)
//    return true;

//  //cout<<"[ObjectTrackerMono::addKeyframe] add keyframe: "<<add_kf<<endl;

//  return add_kf;
//}

/**
 * @brief ObjectTrackerMono::updateView
 * @param model
 * @param pose
 * @param view
 */
void ObjectTrackerMono::updateView(const Eigen::Matrix4f &pose, const Object &model, ObjectView::Ptr &view)
{
  double angle, min_angle = DBL_MAX;
  int idx = -1;
  Eigen::Matrix4f inv_pose, inv_pose2;

  invPose(pose, inv_pose);

  for (unsigned i=0; i<model.views.size(); i++)
  {
    const ObjectView &view = *model.views[i];
    invPose(model.cameras[view.camera_id],inv_pose2);
    angle = viewPointChange(view.center, inv_pose2, inv_pose);

    if ( angle < min_angle)
    {
      min_angle = angle;
      idx = (int)i;
    }
  }

  if (idx != view->idx && idx != -1)
  {
    view = model.views[idx];

    kpDetector->setModel(view);
    projTracker->setModel(view, model.cameras[view->camera_id]);
    lkTracker->setModel(view);
  }
}

/**
 * @brief ObjectTrackerMono::reinit
 * @param im
 * @param pose
 * @param view
 * @return confidence value
 */
double ObjectTrackerMono::reinit(const cv::Mat_<unsigned char> &im, Eigen::Matrix4f &pose, ObjectView::Ptr &view)
{
  view = model->views[rand()%model->views.size()];

  kpDetector->setModel(view);
  projTracker->setModel(view, model->cameras[view->camera_id]);
  lkTracker->setModel(view);

  return kpDetector->detect(im, pose);
}


/***************************************************************************************/

/**
 * @brief ObjectTrackerMono::track
 * @param image input image
 * @param pose estimated pose
 * @param conf confidence value
 * @return
 */
bool ObjectTrackerMono::track(const cv::Mat &image, Eigen::Matrix4f &pose, double &out_conf)
{
  if (model.get()==0 || model->views.size()==0)
    throw std::runtime_error("[ObjectTrackerMono::track] No model available!");

  //kp::ScopeTime t("tracking");
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else image.copyTo(im_gray);

  if (!dbg.empty()) projTracker->dbg = dbg;
  if (!dbg.empty()) kpDetector->dbg = dbg;
  //if (!dbg.empty()) lkTracker->dbg = dbg;

  // do refinement
  if (not_conf_cnt>=param.min_conf_cnt)
  {
    conf = reinit(im_gray, pose, view);
  }

  if (conf > 0.001)
  {
    if (param.do_inc_pyr_lk && conf > param.conf_reinit)
      /*conf =*/lkTracker->detectIncremental(im_gray, pose);
    conf = projTracker->detect(im_gray, pose);
  }

  if (conf>param.conf_reinit)
  {
    lkTracker->setLastFrame(im_gray, pose);
  }

  // compute confidence
  if (conf>=param.min_conf) conf_cnt++;
  else conf_cnt=0;

  if (conf<param.min_conf) not_conf_cnt++;
  else not_conf_cnt=0;

  out_conf = conf;

  // update pose
  if (conf_cnt>=param.min_conf_cnt)
  {
    updateView(pose,*model, view);
    return true;
  }

  return false;
}


/**
 * setCameraParameter
 */
void ObjectTrackerMono::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  dist_coeffs = cv::Mat_<double>();

  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else _intrinsic.copyTo(intrinsic);

  if (!_dist_coeffs.empty())
  {
    dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols*_dist_coeffs.rows; i++)
      dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }

  kpDetector->setCameraParameter(intrinsic, dist_coeffs);
  lkTracker->setCameraParameter(intrinsic, dist_coeffs);
  projTracker->setTargetCameraParameter(intrinsic, dist_coeffs);
}

/**
 * setObjectCameraParameter
 * TODO: We should store the camera parameter of the learning camera
 */
void ObjectTrackerMono::setObjectCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  cv::Mat_<double> intrinsic;
  cv::Mat_<double> dist_coeffs = cv::Mat_<double>();

  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else _intrinsic.copyTo(intrinsic);

  if (!_dist_coeffs.empty())
  {
    dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols*_dist_coeffs.rows; i++)
      dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }

  projTracker->setSourceCameraParameter(intrinsic, dist_coeffs);
}



/**
 * @brief ObjectTrackerMono::setObjectModel
 * @param _model
 */
void ObjectTrackerMono::setObjectModel(const Object::Ptr &_model)
{
  model = _model;

  // set camera parameter
  if (model->camera_parameter.size()==1)
  {
    std::vector<double> &param = model->camera_parameter.back();
    cv::Mat_<double> cam(cv::Mat_<double>::eye(3,3));
    cv::Mat_<double> dcoeffs;

    if (param.size()==9)
    {
      dcoeffs = cv::Mat_<double>::zeros(1,5);
      dcoeffs(0,0) = param[4];
      dcoeffs(0,1) = param[5];
      dcoeffs(0,5) = param[6];
      dcoeffs(0,2) = param[7];
      dcoeffs(0,4) = param[8];
    }

    if (param.size()==4 || param.size()==9)
    {
      cam(0,0) = param[0];
      cam(1,1) = param[1];
      cam(0,2) = param[2];
      cam(1,2) = param[3];

      setObjectCameraParameter(cam, dcoeffs);
    }
  }
}

}












