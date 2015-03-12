/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_PART_HH
#define KP_PART_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "v4r/KeypointTools/triple.hpp"

namespace kp 
{

class Part
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
  typedef SmartPtr< ::kp::Part> Ptr;
  typedef SmartPtr< ::kp::Part const> ConstPtr;

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

  virtual void setBaseTransform(const Eigen::Matrix4f &_pose) {}
  virtual void setBaseTransform(const Eigen::Matrix<double, 6, 1> &_pose) {}
  virtual void getBaseTransform(Eigen::Matrix4f &_pose) {}

  virtual Eigen::VectorXd getParameter() {return Eigen::VectorXd();}
  virtual void setParameter(const Eigen::VectorXd &_param) {}
  virtual void updatePose(const Eigen::Matrix4f &_pose);
  virtual void initParameter() {}

  inline Eigen::Matrix4f &getPose() { return pose; }
  inline const Eigen::Matrix4f &getPose() const { return pose; }
 
  virtual void getDeltaPose(Eigen::Matrix4f &delta_pose) {delta_pose.setIdentity();}
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

