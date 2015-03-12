/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 * 
 * TODO!!!
 */

#include "PartPlanarMotion3D.hh"

#include <stdexcept>



namespace kp 
{


using namespace std;

/************************************************************************************
 * Constructor/Destructor
 */
PartPlanarMotion3D::PartPlanarMotion3D()
 : Part(MOTION_PLANAR_3D), param(Eigen::Matrix<double,3,1>::Zero()), rt(Eigen::Matrix<double, 6, 1>::Zero())
{ 
}

PartPlanarMotion3D::~PartPlanarMotion3D()
{
}

/***************************************************************************************/

/**
 * setParameter
 * @param _param rotation angle [rad]
 */
void PartPlanarMotion3D::setParameter(const Eigen::VectorXd &_param) 
{
  /*if (_param.size()==1)
    angle = _param[0];
  else throw std::runtime_error("[PartPlanarMotion3D::setParameter] Invalid number of parameter!");
*/
}

/**
 * getParameter
 */
Eigen::VectorXd PartPlanarMotion3D::getParameter()
{
  //return Eigen::Map<Eigen::VectorXd>(&angle,1);
  return Eigen::VectorXd();
}

/**
 * setBaseTransform
 */
void PartPlanarMotion3D::setBaseTransform(const Eigen::Matrix4f &_pose)
{
  Eigen::Matrix3d R = _pose.topLeftCorner<3, 3>().cast<double>();

  rt.tail<3>() = _pose.block<3,1>(0, 3).cast<double>(); 
  RotationMatrixToAngleAxis(&R(0,0), &rt(0));
}

/**
 * setBaseTransform
 */
void PartPlanarMotion3D::setBaseTransform(const Eigen::Matrix<double, 6, 1> &_pose)
{
  rt = _pose;
}

/**
 * getBaseTransform
 */
void PartPlanarMotion3D::getBaseTransform(Eigen::Matrix4f &_pose)
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
void PartPlanarMotion3D::updatePose(const Eigen::Matrix4f &_pose)
{
/*  Eigen::Matrix4f inv_T, T(Eigen::Matrix4f::Identity());
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

  pose = _pose * T * pose * inv_T;*/
}
 
/**
 * getDeltaPose
 */
void PartPlanarMotion3D::getDeltaPose(Eigen::Matrix4f &delta_pose)
{
/*  Eigen::Matrix4f inv_T, T(Eigen::Matrix4f::Identity());
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

  delta_pose = T*delta_pose*inv_T;*/
}


}












