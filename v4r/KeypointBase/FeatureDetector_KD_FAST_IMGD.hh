/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_KEYPOINT_DETECTOR_ORB_IMGDESC_HH
#define KP_KEYPOINT_DETECTOR_ORB_IMGDESC_HH

#include <opencv2/features2d/features2d.hpp>
#include "FeatureDetector.hh"
#include "ComputeImGradientDescriptors.hh"



namespace kp 
{

class FeatureDetector_KD_FAST_IMGD : public FeatureDetector
{
public:
  class Parameter
  {
  public:
    int nfeatures;
    float scaleFactor;
    int nlevels;
    int patchSize;
    ComputeImGradientDescriptors::Parameter gdParam;

    Parameter(int _nfeatures=1000, float _scaleFactor=1.44, 
      int _nlevels=2, int _patchSize=17,
      const ComputeImGradientDescriptors::Parameter &_gdParam=ComputeImGradientDescriptors::Parameter()) 
    : nfeatures(_nfeatures), scaleFactor(_scaleFactor), 
      nlevels(_nlevels), patchSize(_patchSize),
      gdParam(_gdParam) {}
  };

private:
  Parameter param;

  cv::Mat_<unsigned char> im_gray;  

  std::vector<cv::Point2f> pts;

  cv::Ptr<cv::ORB> orb;
  ComputeImGradientDescriptors::Ptr imGDesc;

public:
  FeatureDetector_KD_FAST_IMGD(const Parameter &_p=Parameter());
  ~FeatureDetector_KD_FAST_IMGD();

  virtual void detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors); 
  virtual void detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys); 
  virtual void extract(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors); 


  typedef SmartPtr< ::kp::FeatureDetector_KD_FAST_IMGD> Ptr;
  typedef SmartPtr< ::kp::FeatureDetector_KD_FAST_IMGD const> ConstPtr;
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

