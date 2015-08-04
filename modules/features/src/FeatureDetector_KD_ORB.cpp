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

#include <v4r/features/FeatureDetector_KD_ORB.h>

namespace v4r
{


using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
FeatureDetector_KD_ORB::FeatureDetector_KD_ORB(const Parameter &_p)
 : FeatureDetector(KD_ORB), param(_p)
{ 
  //orb = new cv::ORB(10000, 1.2, 6, 13, 0, 2, cv::ORB::HARRIS_SCORE, 13); //31
  //orb = new cv::ORB(1000, 1.44, 2, 17, 0, 2, cv::ORB::HARRIS_SCORE, 17);
  orb = new cv::ORB(param.nfeatures, param.scaleFactor, param.nlevels, param.patchSize, 0, 2, cv::ORB::HARRIS_SCORE, param.patchSize);
}

FeatureDetector_KD_ORB::~FeatureDetector_KD_ORB()
{
}

/***************************************************************************************/

/**
 * detect
 * descriptors is a cv::Mat_<unsigned char>
 */
void FeatureDetector_KD_ORB::detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;  

  (*orb)(im_gray, cv::Mat(), keys, descriptors);
}

/**
 * detect
 */
void FeatureDetector_KD_ORB::detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;  

  (*orb)(im_gray, cv::Mat(), keys);
}

/**
 * detect
 */
void FeatureDetector_KD_ORB::extract(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;  

  (*orb)(im_gray, cv::Mat(), keys, descriptors, true);
}


}
