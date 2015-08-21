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

#ifndef KP_CONVERT_POSE_HPP
#define KP_CONVERT_POSE_HPP

#include <Eigen/Dense>
#include <v4r/common/rotation.h>
#include <opencv2/core/core.hpp>

namespace v4r
{

inline void convertPose(const Eigen::Matrix4f &pose, Eigen::Matrix<double, 6, 1> &rt)
{
  Eigen::Matrix3d R = pose.topLeftCorner<3,3>().cast<double>();
  RotationMatrixToAngleAxis(&R(0,0), &rt[0]);
  rt.tail<3>() = pose.block<3,1>(0,3).cast<double>();
}

inline void convertPose(const Eigen::Matrix4f &pose, Eigen::VectorXd &rt)
{
  rt.resize(6); 
  Eigen::Matrix3d R = pose.topLeftCorner<3,3>().cast<double>();
  RotationMatrixToAngleAxis(&R(0,0), &rt[0]);
  rt.tail<3>() = pose.block<3,1>(0,3).cast<double>();
}

inline void convertPose(const Eigen::Matrix<double, 6, 1> &rt, Eigen::Matrix4f &pose)
{
  Eigen::Matrix3d R;
  pose.setIdentity();
  AngleAxisToRotationMatrix(&rt[0], &R(0,0));
  pose.topLeftCorner<3,3>() = R.cast<float>();
  pose.block<3,1>(0,3) = rt.tail<3>().cast<float>();
}

/**
 * convertPose
 */
inline void convertPose(const cv::Mat_<double> &R, const cv::Mat_<double> &t, Eigen::Matrix4f &pose)
{
  pose.setIdentity();

  pose(0,0) = R(0,0); pose(0,1) = R(0,1); pose(0,2) = R(0,2);
  pose(1,0) = R(1,0); pose(1,1) = R(1,1); pose(1,2) = R(1,2);
  pose(2,0) = R(2,0); pose(2,1) = R(2,1); pose(2,2) = R(2,2);

  pose(0,3) = t(0,0);
  pose(1,3) = t(1,0);
  pose(2,3) = t(2,0);
}


} //--END--

#endif




