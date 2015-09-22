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

#ifndef KP_PART_MOTION_6D_HH
#define KP_PART_MOTION_6D_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <vector>
#include <Eigen/Dense>
#include <v4r/keypoints/Part.h>
#include <v4r/common/rotation.h>

namespace v4r 
{

/**
 * free moving part
 */
class V4R_EXPORTS PartMotion6D : public Part
{
public:
  Eigen::Matrix<double, 6, 1> rt; // 6D motion [angle axis, translation] (parameter)

  PartMotion6D();
  ~PartMotion6D();

  typedef SmartPtr< ::v4r::PartMotion6D> Ptr;
  typedef SmartPtr< ::v4r::PartMotion6D const> ConstPtr;

  virtual void initParameter();
  virtual void setParameter(const Eigen::VectorXd &_param);
  virtual Eigen::VectorXd getParameter();
  virtual void updatePose(const Eigen::Matrix4f &_pose);
  virtual void getDeltaPose(Eigen::Matrix4f &delta_pose);
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

