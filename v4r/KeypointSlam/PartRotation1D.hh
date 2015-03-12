/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_PART_ROTATION_1D_HH
#define KP_PART_ROTATION_1D_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <vector>
#include <Eigen/Dense>
#include <stdexcept>
#include "Part.hh"
#include "v4r/KeypointTools/rotation.h"
#include "v4r/KeypointTools/invPose.hpp"


namespace kp 
{

/**
 * Rotational part 
 */
class PartRotation1D : public Part
{
public:
  double angle;                   // x-axis rotation [rad] (parameter)
  Eigen::Matrix<double, 6, 1> rt; // rotation axis of the part [angle axis, translation]

  PartRotation1D();
  ~PartRotation1D();

  typedef SmartPtr< ::kp::PartRotation1D> Ptr;
  typedef SmartPtr< ::kp::PartRotation1D const> ConstPtr;

  virtual void setParameter(const Eigen::VectorXd &_param);
  virtual Eigen::VectorXd getParameter();
  virtual void setBaseTransform(const Eigen::Matrix4f &_pose);
  virtual void setBaseTransform(const Eigen::Matrix<double, 6, 1> &_pose);
  virtual void getBaseTransform(Eigen::Matrix4f &_pose);
  virtual void updatePose(const Eigen::Matrix4f &_pose);
  virtual void initParameter() {angle = 0;}
  virtual void getDeltaPose(Eigen::Matrix4f &delta_pose);
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

