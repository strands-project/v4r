/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_INV_POSE_HPP
#define KP_INV_POSE_HPP

#include <Eigen/Dense>

namespace kp
{

inline void invPose(const Eigen::Matrix4f &pose, Eigen::Matrix4f &inv_pose)
{
  inv_pose.setIdentity();
  inv_pose.topLeftCorner<3,3>() = pose.topLeftCorner<3,3>().transpose();
  inv_pose.block<3, 1> (0, 3) = -1*( inv_pose.topLeftCorner<3,3>()*pose.block<3, 1>(0,3) );
}


} //--END--

#endif




