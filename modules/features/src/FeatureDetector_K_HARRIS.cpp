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

#include "v4r/features/FeatureDetector_K_HARRIS.h"


namespace v4r 
{

using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
FeatureDetector_K_HARRIS::FeatureDetector_K_HARRIS(const Parameter &_p)
 : FeatureDetector(K_HARRIS), param(_p)
{ 
  imGOri.reset(new ComputeImGDescOrientations(param.goParam));
}

FeatureDetector_K_HARRIS::~FeatureDetector_K_HARRIS()
{
}

/***************************************************************************************/

/**
 * detect
 */
void FeatureDetector_K_HARRIS::detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;  

  cv::goodFeaturesToTrack(im_gray, pts, param.maxCorners, param.qualityLevel, param.minDistance, cv::Mat(), 3, true, 0.01 );

  keys.resize(pts.size());
  for (unsigned i=0; i<keys.size(); i++)
  {
    keys[i] = cv::KeyPoint(pts[i], param.winSize-2);
  }

  imGOri->compute(im_gray, keys);
}



}












