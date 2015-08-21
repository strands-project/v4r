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

#include <v4r/keypoints/PartMotion6D.h>
#include <stdexcept>

namespace v4r
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
