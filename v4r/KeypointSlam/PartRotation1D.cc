/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#include "PartRotation1D.hh"

#include <stdexcept>



namespace kp 
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












