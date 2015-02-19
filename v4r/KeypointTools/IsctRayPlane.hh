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
 * $Id: IsctRayPlane.hh 51 2014-03-30 12:41:22Z hannes $
 */

#ifndef JVIS_ISCT_RAY_PLANE_HH
#define JVIS_ISCT_RAY_PLANE_HH

#include <stdio.h>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>


namespace kp
{

class IsctRayPlane
{
public:
  class Parameter
  {
  public:
    Parameter() {};
  };

private:
  Parameter param;

  cv::Mat_<double> intrinsic;
  cv::Mat_<double> dist_coeffs;

  Eigen::Matrix4f pose;        // coordinate system (defines the camera pose)
  Eigen::Matrix3f R, inv_R;
  Eigen::Vector3f t, inv_t;



public:

  IsctRayPlane(const Parameter &p=Parameter());
  ~IsctRayPlane();

  bool intersect(const Eigen::Vector3f &p, const Eigen::Vector3f &n, const Eigen::Vector2f &im_pt, Eigen::Vector3f &isct_pt);
  void intersect(const Eigen::Vector3f &p, const Eigen::Vector3f &n, const std::vector<Eigen::Vector2f> &im_pts, std::vector<Eigen::Vector3f> &isct_pts);

  void setCameraPose(const Eigen::Matrix4f &_pose);
  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);

  inline bool intersectPlaneLine(const Eigen::Vector3f &p, const Eigen::Vector3f &n, const Eigen::Vector3f &r, Eigen::Vector3f &isct);
};





/*************************** INLINE METHODES **************************/
/**
 * compute plane ray intersection
 * the plane normal n and the ray r must be normalised
 */
inline bool IsctRayPlane::intersectPlaneLine(const Eigen::Vector3f &p, const Eigen::Vector3f &n, const Eigen::Vector3f &r, Eigen::Vector3f &isct)
{
  float tmp = n.dot(r);

  if (fabs(tmp) > std::numeric_limits<float>::epsilon())
  {
    isct = (n.dot(p) / tmp) * r;
    return true;
  }

  return false;
}


} //--END--

#endif

