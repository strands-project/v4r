/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#include "FeatureDetector_KD_FAST_IMGD.hh"



namespace kp 
{


using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
FeatureDetector_KD_FAST_IMGD::FeatureDetector_KD_FAST_IMGD(const Parameter &_p)
 : FeatureDetector(KD_FAST_IMGD), param(_p)
{ 
  //orb = new cv::ORB(10000, 1.2, 6, 13, 0, 2, cv::ORB::HARRIS_SCORE, 13); //31
  //orb = new cv::ORB(1000, 1.44, 2, 17, 0, 2, cv::ORB::HARRIS_SCORE, 17);
  orb = new cv::ORB(param.nfeatures, param.scaleFactor, param.nlevels, param.patchSize, 0, 2, cv::ORB::HARRIS_SCORE, param.patchSize);

  imGDesc.reset(new ComputeImGradientDescriptors(param.gdParam));
}

FeatureDetector_KD_FAST_IMGD::~FeatureDetector_KD_FAST_IMGD()
{
}

/***************************************************************************************/

/**
 * detect
 */
void FeatureDetector_KD_FAST_IMGD::detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;  

  (*orb)(im_gray, cv::Mat(), keys);

  imGDesc->compute(im_gray, keys, descriptors);
}

/**
 * detect
 */
void FeatureDetector_KD_FAST_IMGD::detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;  

  (*orb)(im_gray, cv::Mat(), keys);
}

/**
 * detect
 */
void FeatureDetector_KD_FAST_IMGD::extract(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;  

  imGDesc->compute(im_gray, keys, descriptors);
}



}












