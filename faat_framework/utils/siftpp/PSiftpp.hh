/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef P_SIFTPP_WRAPPER_HH
#define P_SIFTPP_WRAPPER_HH

#include <limits.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include "sift.hpp"


namespace P 
{


class PSiftpp : public cv::FeatureDetector, public cv::DescriptorExtractor
{
public:
  class Parameter
  {
  public:
    int first_level;
    int octaves;
    int levels;
    float threshold;       // Keypoint strength threhsold
    float edge_threshold;  // On-edge threshold
    float magnif;          // Keypoint magnification
    bool computeRootSIFT;  // L1 norm and square root => euc dist = hellinger dist 
    Parameter(int _first_level=-1, int _octaves=-1, int _levels=3, float _threshold=-1, 
        float _edge_threshold=10., float _magnif=3., bool _computeRootSIFT=false) 
      : first_level(_first_level), octaves(_octaves), levels(_levels), threshold(_threshold), 
        edge_threshold(_edge_threshold), magnif(_magnif), computeRootSIFT(_computeRootSIFT) 
    {
      if (threshold<0) threshold = 0.04f / levels / 2.0f ;
    }
  };

private:
  Parameter param;

  int width, height;
  int O, S, omin;
  float sigman, sigma0;

  cv::Mat_<unsigned char> im_gray;
  std::vector<float> im_float;

  cv::Ptr<VL::Sift> sift;

  void initSift(int w, int h);
  void TransformToRootSIFT(cv::Mat& descriptors) const;
  void convertGrayToFloat(const cv::Mat_<unsigned char> &im_gray, std::vector<float> &im_float) const;


protected:
  virtual void detectImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, 
            const cv::Mat& mask=cv::Mat() ) const;
  virtual void computeImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, 
             cv::Mat& descriptors) const;

  
public:

  PSiftpp(const Parameter &p=Parameter());
  ~PSiftpp();

  /** detect keypoints and compute descriptors **/
  void detect( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const cv::Mat& mask=cv::Mat() );

  /** detect keypoints **/
  void detect( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask=cv::Mat() );

  /** compute descriptors **/
  void compute( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors );
  virtual int descriptorSize() const {return 128;}
  virtual int descriptorType() const {return CV_32F;}

  virtual bool isMaskSupported() const {return false;}
};


/************************** INLINE METHODES ******************************/


}

#endif

