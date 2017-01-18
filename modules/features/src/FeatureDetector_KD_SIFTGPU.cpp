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

#include <v4r/features/FeatureDetector_KD_SIFTGPU.h>




namespace v4r
{


using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
FeatureDetector_KD_SIFTGPU::FeatureDetector_KD_SIFTGPU(const Parameter &_p)
 : FeatureDetector(KD_SIFTGPU), 
   param( PSiftGPU::Parameter(_p.distmax,_p.ratiomax,_p.mutual_best_match,_p.computeRootSIFT) )
{ 
  sift = new PSiftGPU(param);
}

FeatureDetector_KD_SIFTGPU::~FeatureDetector_KD_SIFTGPU()
{
}

/***************************************************************************************/

/**
 * detect keypoints and extract descriptors
 */
void FeatureDetector_KD_SIFTGPU::detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;  

  sift->detect(im_gray, keys, descriptors);
}

/**
 * detect keypoints
 */
void FeatureDetector_KD_SIFTGPU::detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;  

  sift->detect(im_gray, keys);
}

/**
 * detect keypoints and extract descriptors
 */
void FeatureDetector_KD_SIFTGPU::extract(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;  

  sift->compute(im_gray, keys, descriptors);
}



}












