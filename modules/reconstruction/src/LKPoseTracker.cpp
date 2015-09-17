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

#include "v4r/reconstruction/LKPoseTracker.h"
#include "opencv2/video/tracking.hpp"
#include <v4r/reconstruction/impl/projectPointToImage.hpp>

namespace v4r
{

using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
LKPoseTracker::LKPoseTracker(const Parameter &p)
 : param(p), last_pose(Eigen::Matrix4f::Identity()), have_im_last(false)
{ 
  sqr_inl_dist = param.inl_dist*param.inl_dist;
}

LKPoseTracker::~LKPoseTracker()
{
}

/**
 * getRandIdx
 */
void LKPoseTracker::getRandIdx(int size, int num, std::vector<int> &idx)
{
  int temp;
  idx.clear();
  for (int i=0; i<num; i++)
  {
    do{
      temp = rand()%size;
    }while(contains(idx,temp));
    idx.push_back(temp);
  }
}

/**
 * countInliers
 */
unsigned LKPoseTracker::countInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4f &pose)
{
  unsigned cnt=0;

  Eigen::Vector2f im_pt;
  Eigen::Vector3f pt3;
  bool have_dist = !dist_coeffs.empty();

  Eigen::Matrix3f R = pose.topLeftCorner<3, 3>();
  Eigen::Vector3f t = pose.block<3,1>(0, 3);

  for (unsigned i=0; i<points.size(); i++)
  {
    pt3 = R*Eigen::Map<const Eigen::Vector3f>(&points[i].x) + t;

    if (have_dist)
      projectPointToImage(&pt3[0], intrinsic.ptr<double>(), dist_coeffs.ptr<double>(), &im_pt[0]);
    else projectPointToImage(&pt3[0], intrinsic.ptr<double>(), &im_pt[0]);

    if ((im_pt - Eigen::Map<const Eigen::Vector2f>(&im_points[i].x)).squaredNorm() < sqr_inl_dist)
    {
      cnt++;
    }
  }

  return cnt;
}

/**
 * getInliers
 */
void LKPoseTracker::getInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4f &pose, std::vector<int> &inliers)
{
  Eigen::Vector2f im_pt;
  Eigen::Vector3f pt3;
  bool have_dist = !dist_coeffs.empty();

  Eigen::Matrix3f R = pose.topLeftCorner<3, 3>();
  Eigen::Vector3f t = pose.block<3,1>(0, 3);

  inliers.clear();

  for (unsigned i=0; i<points.size(); i++)
  {
    pt3 = R*Eigen::Map<const Eigen::Vector3f>(&points[i].x) + t;

    if (have_dist)
      projectPointToImage(&pt3[0], intrinsic.ptr<double>(), dist_coeffs.ptr<double>(), &im_pt[0]);
    else projectPointToImage(&pt3[0], intrinsic.ptr<double>(), &im_pt[0]);

    if ((im_pt - Eigen::Map<const Eigen::Vector2f>(&im_points[i].x)).squaredNorm() < sqr_inl_dist)
    {
      inliers.push_back(i);
    }
  }
}

/**
 * ransacSolvePnP
 */
void LKPoseTracker::ransacSolvePnP(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, Eigen::Matrix4f &pose, std::vector<int> &inliers)
{
  int k=0;
  float sig=param.nb_ransac_points, sv_sig=0.;
  float eps = sig/(float)points.size();
  std::vector<int> indices;
  std::vector<cv::Point3f> model_pts(param.nb_ransac_points);
  std::vector<cv::Point2f> query_pts(param.nb_ransac_points);
  cv::Mat_<double> R(3,3), rvec, tvec, sv_rvec, sv_tvec;

  while (pow(1. - pow(eps,param.nb_ransac_points), k) >= param.eta_ransac && k < (int)param.max_rand_trials)
  {
    getRandIdx(points.size(), param.nb_ransac_points, indices);

    for (unsigned i=0; i<indices.size(); i++)
    {
      model_pts[i] = points[indices[i]];
      query_pts[i] = im_points[indices[i]];
    }

    cv::solvePnP(cv::Mat(model_pts), cv::Mat(query_pts), intrinsic, dist_coeffs, rvec, tvec, false, param.pnp_method);

    cv::Rodrigues(rvec, R);
    cvToEigen(R, tvec, pose);

    sig = countInliers(points, im_points, pose);

    if (sig > sv_sig)
    {
      sv_sig = sig;
      rvec.copyTo(sv_rvec);
      tvec.copyTo(sv_tvec);

      eps = sv_sig / (float)points.size();
    }

    k++;
  }

  if (sv_sig<4) return;

  cv::Rodrigues(sv_rvec, R);
  cvToEigen(R, sv_tvec, pose);
  getInliers(points, im_points, pose, inliers);

  model_pts.resize(inliers.size());
  query_pts.resize(inliers.size());

  for (unsigned i=0; i<inliers.size(); i++)
  {
    model_pts[i] = points[inliers[i]];
    query_pts[i] = im_points[inliers[i]];
  }

  cv::solvePnP(cv::Mat(model_pts), cv::Mat(query_pts), intrinsic, dist_coeffs, sv_rvec, sv_tvec, true, cv::ITERATIVE );

  cv::Rodrigues(sv_rvec, R);
  cvToEigen(R, sv_tvec, pose);


  if (!dbg.empty()) cout<<"Num ransac trials: "<<k<<endl;
}





/******************************* PUBLIC ***************************************/

/**
 * setLastFrame
 */
void LKPoseTracker::setLastFrame(const cv::Mat &image, const Eigen::Matrix4f &pose)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_last, CV_RGB2GRAY );
  else image.copyTo(im_last);

  last_pose = pose;
  have_im_last = true;
}

/**
 * detect
 */
double LKPoseTracker::detectIncremental(const cv::Mat &image, Eigen::Matrix4f &pose)
{
  if (model.get()==0)
    throw std::runtime_error("[LKPoseTracker::detect] No model available!");
  if (intrinsic.empty())
    throw std::runtime_error("[LKPoseTracker::detect] Intrinsic camera parameter not set!");

  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;

  if (!have_im_last || im_last.empty()) {
    //last_pose = pose;
    //im_gray.copyTo(im_last);
    return 0.;
  }

  have_im_last = false;

  ObjectView &m = *model;

  cv::Mat_<double> R(3,3), rvec, tvec;
  std::vector<Eigen::Vector3f> points;
  std::vector<cv::Point3f> model_pts;
  std::vector<cv::Point2f> query_pts;
  std::vector<int> lk_inliers, pnp_inliers;

  m.getPoints(points);

  inliers.clear();

  Eigen::Matrix3f pose_R = last_pose.topLeftCorner<3, 3>();
  Eigen::Vector3f pose_t = last_pose.block<3,1>(0, 3);
  Eigen::Vector3f pt3;
  bool have_dist = !dist_coeffs.empty();

  for (unsigned i=0; i<points.size(); i++)
  {
    pt3 = pose_R*points[i] + pose_t;

    if (have_dist)
      projectPointToImage(&pt3[0], intrinsic.ptr<double>(), dist_coeffs.ptr<double>(), &im_points0[i].x);
    else projectPointToImage(&pt3[0], intrinsic.ptr<double>(), &im_points0[i].x);
  }

  cv::calcOpticalFlowPyrLK(im_last, im_gray, im_points0, im_points1, status, error, param.win_size, param.max_level, param.termcrit, 0, 0.001 );

  for (unsigned i=0; i<im_points0.size(); i++)
  {
    if (status[i]!=0 && error[i]<param.max_error)
    {
      lk_inliers.push_back(i);
      const Eigen::Vector3d &pt = m.getPt(i).pt;
      model_pts.push_back(cv::Point3f(pt[0],pt[1],pt[2]));
      query_pts.push_back(im_points1[i]);
      if (!dbg.empty()) cv::line(dbg,im_points0[i], im_points1[i],CV_RGB(0,0,0));
    }
  }

  if (int(query_pts.size())<4) return 0.;

  ransacSolvePnP(model_pts, query_pts, pose, pnp_inliers);

  if (int(pnp_inliers.size())<4) return 0.;

  for (unsigned i=0; i<pnp_inliers.size(); i++) {
    inliers.push_back(lk_inliers[pnp_inliers[i]]);
    if (!dbg.empty()) cv::circle(dbg,im_points1[inliers.back()],2,CV_RGB(255,255,0));
  }

  return double(inliers.size())/double(model->points.size());
}


/**
 * detect
 */
double LKPoseTracker::detect(const cv::Mat &image, Eigen::Matrix4f &pose)
{
  if (model.get()==0)
    throw std::runtime_error("[LKPoseTracker::detect] No model available!");
  if (intrinsic.empty())
    throw std::runtime_error("[LKPoseTracker::detect] Intrinsic camera parameter not set!");

  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;

  ObjectView &m = *model;

  cv::Mat_<double> R(3,3), rvec, tvec;
  std::vector<cv::Point3f> model_pts;
  std::vector<cv::Point2f> query_pts;
  std::vector<int> lk_inliers, pnp_inliers;

  inliers.clear();

  Eigen::Matrix3f pose_R = pose.topLeftCorner<3, 3>();
  Eigen::Vector3f pose_t = pose.block<3,1>(0, 3);
  Eigen::Vector3f pt3;
  bool have_dist = !dist_coeffs.empty();

  for (unsigned i=0; i<m.points.size(); i++)
  {
    pt3 = pose_R*m.getPt(i).pt.cast<float>() + pose_t;

    if (have_dist)
      projectPointToImage(&pt3[0], intrinsic.ptr<double>(), dist_coeffs.ptr<double>(), &im_points1[i].x);
    else projectPointToImage(&pt3[0], intrinsic.ptr<double>(), &im_points1[i].x);
  }

  cv::calcOpticalFlowPyrLK(model->image, im_gray, im_points0, im_points1, status, error, param.win_size, param.max_level, param.termcrit, cv::OPTFLOW_USE_INITIAL_FLOW, 0.001 );


  for (unsigned i=0; i<im_points0.size(); i++)
  {
    if (status[i]!=0 && error[i]<param.max_error)
    {
      lk_inliers.push_back(i);
      const Eigen::Vector3d &pt = m.getPt(i).pt;
      model_pts.push_back(cv::Point3f(pt[0],pt[1],pt[2]));
      query_pts.push_back(im_points1[i]);
      if (!dbg.empty()) cv::line(dbg,im_points0[i], im_points1[i],CV_RGB(0,0,0));
    }
  }

  if (int(query_pts.size())<4) return 0.;

  ransacSolvePnP(model_pts, query_pts, pose, pnp_inliers);

  if (int(pnp_inliers.size())<4) return 0.;

  for (unsigned i=0; i<pnp_inliers.size(); i++) {
    inliers.push_back(lk_inliers[pnp_inliers[i]]);
    if (!dbg.empty()) cv::circle(dbg,im_points1[inliers.back()],2,CV_RGB(255,255,0));
  }

  return double(inliers.size())/double(model->points.size());
}

/**
 * getProjections
 * @param im_pts <model_point_index, projection>
 */
void LKPoseTracker::getProjections(std::vector< std::pair<int,cv::Point2f> > &im_pts)
{
  im_pts.clear();
  if (model.get()==0 || im_points1.size()!=model->keys.size() || inliers.size()>im_points1.size())
    return;

  for (unsigned i=0; i<inliers.size(); i++)
    im_pts.push_back(make_pair(inliers[i],im_points1[inliers[i]]));
}

/**
 * setModel
 */
void LKPoseTracker::setModel(const ObjectView::Ptr &_model) 
{  
  if (_model->points.size()!=_model->keys.size())
    throw std::runtime_error("[LKPoseTracker::setModel] Invalid model!");

  model=_model; 
  ObjectView &view = *model;

  im_points0.resize(view.keys.size());
  im_points1.resize(view.keys.size());

  for (unsigned i=0; i<view.keys.size(); i++)
    im_points0[i] = view.keys[i].pt;
}

/**
 * setCameraParameter
 */
void LKPoseTracker::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  dist_coeffs = cv::Mat_<double>();
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;
  if (!_dist_coeffs.empty())
  {
    dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols*_dist_coeffs.rows; i++)
      dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }
}



}












