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
#include <v4r/common/impl/SmartPtr.hpp>
#include <v4r/common/ClusteringRNN.h>

namespace v4r 
{

class V4R_EXPORTS FeatureSelection
{
public:
  class Parameter
  {
  public:
    float thr_image_px;
    float thr_desc;
    Parameter(float _thr_image_px=1.5, float _thr_desc=0.55)
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

  typedef SmartPtr< ::v4r::FeatureSelection> Ptr;
  typedef SmartPtr< ::v4r::FeatureSelection const> ConstPtr;
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

