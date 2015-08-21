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

#include <v4r/keypoints/PartRotation1D.h>

#include <stdexcept>



namespace v4r 
{


using namespace std;

/************************************************************************************
 * Constructor/Destructor
 */
PartRotation1D::PartRotation1D()
 : Part(ROTATION_1D), angle(0), rt(Eigen::Matrix<double, 6, 1>::Zero())
{ 
  //rt[0] = pi_2;
}

PartRotation1D::~PartRotation1D()
{
}

/***************************************************************************************/

/**
 * setParameter
 * @param _param rotation angle [rad]
 */
void PartRotation1D::setParameter(const Eigen::VectorXd &_param) 
{
  if (_param.size()==1)
    angle = _param[0];
  else throw std::runtime_error("[PartRotation1D::setParameter] Invalid number of parameter!");
}

/**
 * getParameter
 */
Eigen::VectorXd PartRotation1D::getParameter()
{
  return Eigen::Map<Eigen::VectorXd>(&angle,1);
}

/**
 * setBaseTransform
 */
void PartRotation1D::setBaseTransform(const Eigen::Matrix4f &_pose)
{
  Eigen::Matrix3d R = _pose.topLeftCorner<3, 3>().cast<double>();

  rt.tail<3>() = _pose.block<3,1>(0, 3).cast<double>(); 
  RotationMatrixToAngleAxis(&R(0,0), &rt(0));
}

/**
 * setBaseTransform
 */
void PartRotation1D::setBaseTransform(const Eigen::Matrix<double, 6, 1> &_pose)
{
  rt = _pose;
}

/**
 * getBaseTransform
 */
void PartRotation1D::getBaseTransform(Eigen::Matrix4f &_pose)
{
  Eigen::Matrix3d R;
  _pose.setIdentity();
  AngleAxisToRotationMatrix(&rt(0), &R(0,0));
  _pose.topLeftCorner<3,3>() = R.cast<float>();
  _pose.block<3,1>(0,3) = rt.tail<3>().cast<float>();
}


/**
 * updatePose
 */
void PartRotation1D::updatePose(const Eigen::Matrix4f &_pose)
{
  Eigen::Matrix4f inv_T, T(Eigen::Matrix4f::Identity());
  Eigen::Matrix3d R;
  Eigen::Vector3d rx(0.,0.,0.);//rx(1.,0.,0);

  AngleAxisToRotationMatrix(&rt(0), &R(0,0));
  T.topLeftCorner<3,3>() = R.cast<float>();
  T.block<3,1>(0,3) = rt.tail<3>().cast<float>();
  invPose(T,inv_T);

  //rx *= angle;
  rx[0] = angle;
  AngleAxisToRotationMatrix(&rx(0), &R(0,0));
  pose.setIdentity();
  pose.topLeftCorner<3,3>() = R.cast<float>();

  pose = _pose * T * pose * inv_T;
}
 
/**
 * getDeltaPose
 */
void PartRotation1D::getDeltaPose(Eigen::Matrix4f &delta_pose)
{
  Eigen::Matrix4f inv_T, T(Eigen::Matrix4f::Identity());
  Eigen::Matrix3d R;
  Eigen::Vector3d rx(0.,0.,0.);//rx(1.,0.,0);

  AngleAxisToRotationMatrix(&rt(0), &R(0,0));
  T.topLeftCorner<3,3>() = R.cast<float>();
  T.block<3,1>(0,3) = rt.tail<3>().cast<float>();
  invPose(T,inv_T);

  //rx *= angle;
  rx[0] = angle;
  AngleAxisToRotationMatrix(&rx(0), &R(0,0));
  delta_pose.setIdentity();
  delta_pose.topLeftCorner<3,3>() = R.cast<float>();

  delta_pose = T*delta_pose*inv_T;
}


}












