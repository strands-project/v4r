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

#ifndef KP_FEATURE_DETECTOR_HH
#define KP_FEATURE_DETECTOR_HH

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <v4r/core/macros.h>
#include <v4r/common/impl/SmartPtr.hpp>


namespace v4r 
{

class V4R_EXPORTS FeatureDetector
{
public:
  enum Type
  {
    K_MSER,
    K_HARRIS,
    KD_CVSURF,
    KD_CVSIFT,
    KD_SIFTGPU,
    D_FREAK,
    KD_ORB,
    KD_FAST_IMGD,
    KD_PSURF,
    KD_MSER_IMGD,
    KD_HARRIS_IMGD,
    KD_PSURF_FREAK,
    KD_PSURF_IMGD,
    KD_FAST_PSURF,
    KD_FAST_SIFTGPU,
    KD_CVSURF_FREAK,
    KD_CVSURF_IMGD,
    KD_FAST_CVSURF,
    KD_SIFTGPU_IMGD,
    MAX_TYPE,
    UNDEF = MAX_TYPE
  };

private:
  Type type;

public:
  FeatureDetector(Type _type=UNDEF) : type(_type) {}
  virtual ~FeatureDetector() {}

  virtual void detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors) { 
    std::cout<<"[FeatureDetector::detect] Not implemented!]"<<std::endl; };

  virtual void detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys) {
    std::cout<<"[FeatureDetector::detect] Not implemented!]"<<std::endl; }

   virtual void extract(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors) {
    std::cout<<"[FeatureDetector::extract] Not implemented!]"<<std::endl; }
 

  Type getType() {return type;}

  typedef SmartPtr< ::v4r::FeatureDetector> Ptr;
  typedef SmartPtr< ::v4r::FeatureDetector const> ConstPtr;
};

}

#endif

