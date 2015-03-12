/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#include "PartMotion6D.hh"

#include <stdexcept>



namespace kp 
{


using namespace std;

/************************************************************************************
 * Constructor/Destructor
 */
PartMotion6D::PartMotion6D()
 : Part(MOTION_6D), rt(Eigen::Matrix<double, 6, 1>::Zero())
{ 
  //rt[0] = pi_2;
}

PartMotion6D::~PartMotion6D()
{
}

/***************************************************************************************/

/**
 * initParameter
 */
void PartMotion6D::initParameter()
{
  rt = Eigen::Matrix<double, 6, 1>::Zero();
  //rt[0] = pi_2;
}

/**
 * setParameter
 * @param _param rotation angle [rad]
 */
void PartMotion6D::setParameter(const Eigen::VectorXd &_param)
{
  if (_param.size()==6)
    rt = _param;
  else throw std::runtime_error("[PartMotion6D::setParameter] Invalid number of parameter!");
}

/**
 * getParameter
 */
Eigen::VectorXd PartMotion6D::getParameter()
{
  return rt;
}

/**
 * updatePose
 */
void PartMotion6D::updatePose(const Eigen::Matrix4f &_pose)
{
  Eigen::Matrix3d R;

  AngleAxisToRotationMatrix(&rt(0), &R(0,0));
  pose.topLeftCorner<3,3>() = R.cast<float>();
  pose.block<3,1>(0,3) = rt.tail<3>().cast<float>();

  pose = _pose * pose;
}
 
/**
 * getDeltaPose
 */
void PartMotion6D::getDeltaPose(Eigen::Matrix4f &delta_pose)
{
  Eigen::Matrix3d R;
  AngleAxisToRotationMatrix(&rt(0), &R(0,0));
  delta_pose.topLeftCorner<3,3>() = R.cast<float>();
  delta_pose.block<3,1>(0,3) = rt.tail<3>().cast<float>();
}


}












