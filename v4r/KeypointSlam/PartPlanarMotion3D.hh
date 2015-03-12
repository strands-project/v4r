/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_PART_PLANAR_MOTION_3D_HH
#define KP_PART_PLANAR_MOTION_3D_HH

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
class PartPlanarMotion3D : public Part
{
public:
  Eigen::Matrix<double,3,1> param;  // rotation 1D, in plane translation 2d
  Eigen::Matrix<double, 6, 1> rt;   // rotation axis of the part [angle axis, translation]

  PartPlanarMotion3D();
  ~PartPlanarMotion3D();

  typedef SmartPtr< ::kp::PartPlanarMotion3D> Ptr;
  typedef SmartPtr< ::kp::PartPlanarMotion3D const> ConstPtr;

  virtual void setParameter(const Eigen::VectorXd &_param);
  virtual Eigen::VectorXd getParameter();
  virtual void setBaseTransform(const Eigen::Matrix4f &_pose);
  virtual void setBaseTransform(const Eigen::Matrix<double, 6, 1> &_pose);
  virtual void getBaseTransform(Eigen::Matrix4f &_pose);
  virtual void updatePose(const Eigen::Matrix4f &_pose);
  virtual void initParameter() { param.setZero(); }
  virtual void getDeltaPose(Eigen::Matrix4f &delta_pose);
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

