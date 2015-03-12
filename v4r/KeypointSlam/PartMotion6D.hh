/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_PART_MOTION_6D_HH
#define KP_PART_MOTION_6D_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <vector>
#include <Eigen/Dense>
#include "Part.hh"
#include "v4r/KeypointTools/rotation.h"

namespace kp 
{

/**
 * free moving part
 */
class PartMotion6D : public Part
{
public:
  Eigen::Matrix<double, 6, 1> rt; // 6D motion [angle axis, translation] (parameter)

  PartMotion6D();
  ~PartMotion6D();

  typedef SmartPtr< ::kp::PartMotion6D> Ptr;
  typedef SmartPtr< ::kp::PartMotion6D const> ConstPtr;

  virtual void initParameter();
  virtual void setParameter(const Eigen::VectorXd &_param);
  virtual Eigen::VectorXd getParameter();
  virtual void updatePose(const Eigen::Matrix4f &_pose);
  virtual void getDeltaPose(Eigen::Matrix4f &delta_pose);
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

