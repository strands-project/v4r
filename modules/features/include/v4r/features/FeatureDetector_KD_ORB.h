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

#ifndef KP_FEATURE_DETECTOR_ORB_HH
#define KP_FEATURE_DETECTOR_ORB_HH

#include "FeatureDetector.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>


namespace v4r 
{

class V4R_EXPORTS FeatureDetector_KD_ORB : public FeatureDetector
{
public:
  class Parameter
  {
  public:
    int nfeatures;
    float scaleFactor;
    int nlevels;
    int patchSize;
    Parameter(int _nfeatures=1000, float _scaleFactor=1.44, int _nlevels=2, int _patchSize=17)
    : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels), patchSize(_patchSize) {}
  };

private:
  Parameter param;
  cv::Mat_<unsigned char> im_gray;  

  cv::Ptr<cv::ORB> orb;

public:
  FeatureDetector_KD_ORB(const Parameter &_p=Parameter());
  ~FeatureDetector_KD_ORB();

  virtual void detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors); 
  virtual void detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys); 
  virtual void extract(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors); 


  typedef SmartPtr< ::v4r::FeatureDetector_KD_ORB> Ptr;
  typedef SmartPtr< ::v4r::FeatureDetector_KD_ORB const> ConstPtr;
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

