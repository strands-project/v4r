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

#ifndef KP_FEATURE_DETECTOR_HARRIS_HH
#define KP_FEATURE_DETECTOR_HARRIS_HH

#include <opencv2/features2d/features2d.hpp>
#include "FeatureDetector.h"
#include "ComputeImGDescOrientations.h"

namespace v4r 
{

class V4R_EXPORTS FeatureDetector_K_HARRIS : public FeatureDetector
{
public:
  class Parameter
  {
  public:
    int winSize;        // e.g. 32*32 window + 2px border = 34
    int maxCorners; 
    double qualityLevel; // 0.0001
    double minDistance;
    bool computeDesc;
    bool uprightDesc;
    ComputeImGDescOrientations::Parameter goParam;

    Parameter(int _winSize=34, int _maxCorners=5000, const double &_qualityLevel=0.0002, 
      const double &_minDistance=1.,  bool _computeDesc=true, bool _uprightDesc=false,
      const ComputeImGDescOrientations::Parameter &_goParam=ComputeImGDescOrientations::Parameter())
    : winSize(_winSize), maxCorners(_maxCorners), qualityLevel(_qualityLevel), 
      minDistance(_minDistance), computeDesc(_computeDesc), uprightDesc(_uprightDesc),
      goParam(_goParam) {}
  };

private:
  Parameter param;

  cv::Mat_<unsigned char> im_gray;  

  std::vector<cv::Point2f> pts;

  ComputeImGDescOrientations::Ptr imGOri;

public:
  FeatureDetector_K_HARRIS(const Parameter &_p=Parameter());
  ~FeatureDetector_K_HARRIS();

  virtual void detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys); 

  typedef SmartPtr< ::v4r::FeatureDetector_K_HARRIS> Ptr;
  typedef SmartPtr< ::v4r::FeatureDetector_K_HARRIS const> ConstPtr;
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

