/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_FEATURE_SELECTION_HH
#define KP_FEATURE_SELECTION_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Dense>
#include <stdexcept>
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "v4r/KeypointTools/ClusteringRNN.hh"

namespace kp 
{

class FeatureSelection
{
public:
  class Parameter
  {
  public:
    float thr_image_px;
    float thr_desc;
    Parameter(float _thr_image_px=1., float _thr_desc=0.55)
    : thr_image_px(_thr_image_px), thr_desc(_thr_desc) {}
  };

private:
  Parameter param;

  ClusteringRNN rnn;

  DataMatrix2Df descs;
  DataMatrix2Df pts;

public:
  cv::Mat dbg;

  FeatureSelection(const Parameter &p=Parameter());
  ~FeatureSelection();

  void compute(std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors);

  typedef SmartPtr< ::kp::FeatureSelection> Ptr;
  typedef SmartPtr< ::kp::FeatureSelection const> ConstPtr;
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

