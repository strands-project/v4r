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

#ifndef KP_INV_POSE_HPP
#define KP_INV_POSE_HPP

#include <Eigen/Dense>
#include <v4r/common/rotation.h>

namespace v4r
{

inline void invPose(const Eigen::Matrix4f &pose, Eigen::Matrix4f &inv_pose)
{
  inv_pose.setIdentity();
  inv_pose.topLeftCorner<3,3>() = pose.topLeftCorner<3,3>().transpose();
  inv_pose.block<3, 1> (0, 3) = -1*( inv_pose.topLeftCorner<3,3>()*pose.block<3, 1>(0,3) );
}

inline void invPose(const Eigen::Matrix4d &pose, Eigen::Matrix4d &inv_pose)
{
  inv_pose.setIdentity();
  inv_pose.topLeftCorner<3,3>() = pose.topLeftCorner<3,3>().transpose();
  inv_pose.block<3, 1> (0, 3) = -1*( inv_pose.topLeftCorner<3,3>()*pose.block<3, 1>(0,3) );
}

/**
 * invPose6
 */
template<typename T1,typename T2, typename T3, typename T4>
inline void invPose6(const T1 r[3], const T2 t[3], T3 inv_r[3], T4 inv_t[3])
{
  inv_r[0] = T1(-1)*r[0];
  inv_r[1] = T1(-1)*r[1];
  inv_r[2] = T1(-1)*r[2];

  v4r::AngleAxisRotatePoint(inv_r, t, inv_t);
  
  inv_t[0] = T4(-1)*inv_t[0];
  inv_t[1] = T4(-1)*inv_t[1];
  inv_t[2] = T4(-1)*inv_t[2];
}


} //--END--

#endif




