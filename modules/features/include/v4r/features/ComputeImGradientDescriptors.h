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

#ifndef V4R_COMPUTE_GRADIENT_DESCRIPTORS_HH
#define V4R_COMPUTE_GRADIENT_DESCRIPTORS_HH

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ImGradientDescriptor.h"

namespace v4r 
{

class V4R_EXPORTS ComputeImGradientDescriptors
{
public:
  class Parameter
  {
  public:
    int win_size;
    ImGradientDescriptor::Parameter ghParam;
    Parameter(int _win_size=34, 
      const ImGradientDescriptor::Parameter &_ghParam=ImGradientDescriptor::Parameter())
    : win_size(_win_size), ghParam(_ghParam) {}
  };

private:
  Parameter param;

  int h_win;

public:
 

  ComputeImGradientDescriptors(const Parameter &p=Parameter());
  ~ComputeImGradientDescriptors();

  void compute(const cv::Mat_<unsigned char> &image, const std::vector<cv::Point2f> &pts, 
        cv::Mat &descriptors);
  void compute(const cv::Mat_<unsigned char> &image, const std::vector<cv::KeyPoint> &keys, 
        cv::Mat &descriptors);
  //void compute(const cv::Mat_<unsigned char> &image, const std::vector<AffKeypoint> &keys, 
  //      cv::Mat &descriptors);




  typedef SmartPtr< ::v4r::ComputeImGradientDescriptors> Ptr;
  typedef SmartPtr< ::v4r::ComputeImGradientDescriptors const> ConstPtr;

};


/*************************** INLINE METHODES **************************/


} //--END--

#endif

