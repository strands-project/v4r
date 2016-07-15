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

#ifndef KP_PART_HH
#define KP_PART_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <v4r/common/impl/SmartPtr.hpp>
#include <v4r/core/macros.h>
#include <v4r/keypoints/impl/triple.hpp>

namespace v4r
{

class V4R_EXPORTS Part
{
public:
  enum Type
  {
    STATIC,
    ROTATION_1D,
    TRANSLATION_1D,
    MOTION_6D,
    MOTION_PLANAR_3D,
    MAX_TYPE,
    UNDEF = MAX_TYPE
  };

public:
  typedef SmartPtr< ::v4r::Part> Ptr;
  typedef SmartPtr< ::v4r::Part const> ConstPtr;

  Type type;
  int idx;                    // index to ObjectModel::parts

  bool is_hyp;                // is a part hypothesis

  static double pi_2;

  Eigen::Matrix4f pose;             // global pose

  std::vector< std::pair<int, int> > features; // <view_idx,feature_idx>
  std::vector< std::vector< triple<int, cv::Point2f, Eigen::Vector3f> > > projs;//size of features <cam, proj, current_3d_location>

  std::vector<int> subparts;           //sub parts


  Part();
  Part(Type _type);

  virtual ~Part();

  virtual void setBaseTransform(const Eigen::Matrix4f &_pose) {(void)_pose;}
  virtual void setBaseTransform(const Eigen::Matrix<double, 6, 1> &_pose) {(void)_pose;}
  virtual void getBaseTransform(Eigen::Matrix4f &_pose) {(void)_pose;}

  virtual Eigen::VectorXd getParameter() {return Eigen::VectorXd();}
  virtual void setParameter(const Eigen::VectorXd &_param) {(void)_param;}
  virtual void updatePose(const Eigen::Matrix4f &_pose);
  virtual void initParameter() {}

  inline Eigen::Matrix4f &getPose() { return pose; }
  inline const Eigen::Matrix4f &getPose() const { return pose; }
 
  virtual void getDeltaPose(Eigen::Matrix4f &delta_pose) {delta_pose.setIdentity();}
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

