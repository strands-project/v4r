/*
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (c) 2014, Johann Prankl, johann.prankl@josephinum.at
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
 * @author Johann Prankl
 *
 */

/**
 * $Id: IsctRayPlane.cc 51 2014-03-30 12:41:22Z hannes $
 */


#include "IsctRayPlane.hh"
#include "invPose.hpp"



namespace kp 
{


/************************************************************************************
 * Constructor/Destructor
 */
IsctRayPlane::IsctRayPlane(const Parameter &p)
 : param(p), pose(Eigen::Matrix4f::Identity()), R(Eigen::Matrix3f::Identity()), inv_R(Eigen::Matrix3f::Identity()), t(Eigen::Vector3f::Zero()), inv_t(Eigen::Vector3f::Zero())
{ 
}

IsctRayPlane::~IsctRayPlane()
{
}





/***************************************************************************************/


/**
 * Intersect
 * @param p a plane point (e.g. [0 0 0])
 * @param n plane normal (e.g. [0 0 1])
 * @param imPt image point which defines the ray from the focal point
 * @param plPt intersection point with the plane
 */
bool IsctRayPlane::intersect(const Eigen::Vector3f &p, const Eigen::Vector3f &n, const Eigen::Vector2f &im_pt, Eigen::Vector3f &isct_pt)
{
  Eigen::Vector3f ray, p_cam, n_cam, isct;

  p_cam = R*p + t; 
  n_cam = R*n;

  if (!dist_coeffs.empty())
  {
    std::vector<cv::Point2f> vec_ud_pt;
    cv::Mat mat_pt(1,1, CV_32FC2 );

    mat_pt.at<cv::Point2f>(0,0) = cv::Point2f(im_pt[0], im_pt[1]);
    cv::undistortPoints(mat_pt, vec_ud_pt, intrinsic, dist_coeffs);

    ray = Eigen::Vector3f(vec_ud_pt[0].x, vec_ud_pt[0].y, 1.);
  }
  else
  {
    double *C = intrinsic.ptr<double>(0);
    
    ray[0] = (im_pt[0] - C[2]) / C[0];
    ray[1] = (im_pt[1] - C[5]) / C[4];
    ray[2] = 1.;
  }

  ray.normalize();

  if (intersectPlaneLine(p_cam, n_cam, ray, isct))
  {
    isct_pt = inv_R*isct + inv_t;
    return true;
  }

  return false;
}

/**
 * Intersect
 * @param p a plane point (e.g. [0 0 0])
 * @param n plane normal (e.g. [0 0 1])
 * @param imPts image points which defines the ray from the focal point
 * @param plPts intersection points with the plane
 */
void IsctRayPlane::intersect(const Eigen::Vector3f &p, const Eigen::Vector3f &n, const std::vector<Eigen::Vector2f> &im_pts, std::vector<Eigen::Vector3f> &isct_pts)
{
  isct_pts.clear();
  isct_pts.reserve(im_pts.size());
  Eigen::Vector3f ray, p_cam, n_cam, isct;
  std::vector<cv::Point2f> ud_pts;

  p_cam = R*p + t;
  n_cam = R*n;

  if (!dist_coeffs.empty())
  {
    std::vector<cv::Point2f> fl_im_pts(im_pts.size());

    for (unsigned i=0; i<im_pts.size(); i++)
      fl_im_pts[i] = cv::Point2f(im_pts[i][0],im_pts[i][1]);
    
    cv::undistortPoints(cv::Mat(fl_im_pts), ud_pts, intrinsic, dist_coeffs);
  }
  else
  {
    double *C = intrinsic.ptr<double>(0);
    ud_pts.resize(im_pts.size());

    for (unsigned i=0; i<im_pts.size(); i++)
    {
      ud_pts[i] = cv::Point2f( (im_pts[i][0]-C[2])/C[0], (im_pts[i][1]-C[5])/C[4] );
    }
  }

  for (unsigned i=0; i<ud_pts.size(); i++)
  {
    ray = Eigen::Vector3f(ud_pts[i].x, ud_pts[i].y, 1.);

    ray.normalize();

    if (intersectPlaneLine(p_cam, n_cam, ray, isct))
    {
      isct_pts.push_back( inv_R*isct + inv_t );
    }
  }
}

/**
 * set coordinate system of the plane (to camera)
 */
void IsctRayPlane::setCameraPose(const Eigen::Matrix4f &_pose)
{
  Eigen::Matrix4f inv_pose;

  pose = _pose;
  R = pose.topLeftCorner<3,3>();
  t = pose.block<3,1>(0,3);

  invPose(pose, inv_pose);
  
  inv_R = inv_pose.topLeftCorner<3,3>();
  inv_t = inv_pose.block<3,1>(0,3);
}

/**
 * setCameraParameter
 */
void IsctRayPlane::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  dist_coeffs = cv::Mat_<double>();

  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;

  if (!_dist_coeffs.empty())
  {
    dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols; i++)
      dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }
}



}












