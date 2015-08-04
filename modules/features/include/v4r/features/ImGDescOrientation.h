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

#ifndef V4R_IM_GDESC_ORIENTATION_HH
#define V4R_IM_GDESC_ORIENTATION_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include <stdexcept>
#include <v4r/core/macros.h>
#include <v4r/common/impl/SmartPtr.hpp>

namespace v4r 
{

class V4R_EXPORTS ImGDescOrientation
{
public:
  class Parameter
  {
  public:
    bool smooth;
    float sigma; //1.6
    Parameter(bool _smooth=true, float _sigma=1.6)
    : smooth(_smooth), sigma(_sigma) {}
  };

private:
  Parameter param;

  cv::Mat_<unsigned char> im_smooth;
  cv::Mat_<short> im_dx, im_dy;
  cv::Mat_<float> lt_gauss;

  std::vector<float> hist;

  void ComputeGradients(const cv::Mat_<unsigned char> &im);
  void ComputeAngle(const cv::Mat_<float> &weight, float &angle);
  void ComputeLTGaussCirc(const cv::Mat_<unsigned char> &im);


public:

  ImGDescOrientation(const Parameter &p=Parameter());
  ~ImGDescOrientation();

  void compute(const cv::Mat_<unsigned char> &im, float &angle);
  void compute(const cv::Mat_<unsigned char> &im, const cv::Mat_<float> &weight, float &angle);

  typedef SmartPtr< ::v4r::ImGDescOrientation> Ptr;
  typedef SmartPtr< ::v4r::ImGDescOrientation const> ConstPtr;
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

