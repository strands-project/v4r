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

#ifndef KP_SIFTGPU_WRAPPER_HH
#define KP_SIFTGPU_WRAPPER_HH

#include <limits.h>
#include <GL/glut.h>
#include <dlfcn.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <SiftGPU/SiftGPU.h>
#include <v4r/common/impl/DataMatrix2D.hpp>


namespace v4r
{


#if CV_MAJOR_VERSION < 3
class V4R_EXPORTS PSiftGPU : public cv::FeatureDetector, public cv::DescriptorExtractor, public cv::DescriptorMatcher
#else
class V4R_EXPORTS PSiftGPU : public cv::FeatureDetector, public cv::DescriptorMatcher
#endif
{
public:
  class Parameter
  {
  public:
    float distmax;         // absolute descriptor distance (e.g. = 0.6)
    float ratiomax;        // compare best match with second best (e.g. =0.8)
    int mutual_best_match; // compare forward/backward matches (1)
    bool computeRootSIFT;  // L1 norm and square root => euc dist = hellinger dist 
    Parameter(float d=FLT_MAX, float r=1., int m=0, bool _computeRootSIFT=false) 
      : distmax(d), ratiomax(r), mutual_best_match(m), computeRootSIFT(_computeRootSIFT) {}
  };

private:
  cv::Ptr<SiftGPU> sift;
  cv::Ptr<SiftMatchGPU> matcher;
  int gpuMemSize;

//  vector<cv::Mat> trainDescriptors;

  void TransformToRootSIFT(cv::Mat& descriptors) const;
  void TransformToRootSIFT(DataMatrix2Df &descriptors) const;


  inline float Distance128(float d1[128], float d2[128]);
  inline float sqr(const float &a) { return a*a; }


protected:
  virtual void detectImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask=cv::Mat() ) const;
  virtual void computeImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, 
             cv::Mat& descriptors) const;
  virtual void knnMatchImpl( const cv::Mat& queryDescriptors, std::vector<std::vector<cv::DMatch> >& matches,
             int k, const std::vector<cv::Mat>& masks=std::vector<cv::Mat>(), bool compactResult=false );
  virtual void radiusMatchImpl( const cv::Mat& queryDescriptors, std::vector<std::vector<cv::DMatch> >& matches, 
             float maxDistance, const std::vector<cv::Mat>& masks=std::vector<cv::Mat>(), bool compactResult=false );

#if CV_MAJOR_VERSION >= 3
  virtual void knnMatchImpl( cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch> >& matches, int k,
      cv::InputArrayOfArrays masks=cv::noArray(), bool compactResult=false )
  {
      cv::Mat queryDesc;
      queryDesc.setTo( queryDescriptors);
      knnMatchImpl( queryDesc, matches, k, masks, compactResult );
  }

  virtual void radiusMatchImpl( cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch> >& matches, float maxDistance,
      cv::InputArrayOfArrays masks=cv::noArray(), bool compactResult=false )
  {
      cv::Mat queryDesc;
      queryDesc.setTo( queryDescriptors);
      radiusMatchImpl( queryDesc, matches, maxDistance, masks, compactResult );
  }
#endif

  
public:
  Parameter param;

  PSiftGPU(Parameter p=Parameter(), cv::Ptr<SiftGPU> _sift=cv::Ptr<SiftGPU>(), cv::Ptr<SiftMatchGPU> _matcher=cv::Ptr<SiftMatchGPU>(), int memSize = 4096);
  ~PSiftGPU();

  /**
   * detects keypoints and the descriptors
   */
  void detect( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, DataMatrix2Df &descriptors, const cv::Mat& mask=cv::Mat() );

  /**
   * compute descriptor
   */
  void compute(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, DataMatrix2Df &descriptors);
 
  /** detect keypoints **/
  void detect( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask=cv::Mat() );

  /** compute descriptors **/
  void compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors );
  virtual int descriptorSize() const {return 128;}
  virtual int descriptorType() const {return CV_32F;}

  /** match descriptors - for what ever reason:-) **/
  void match( const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors, std::vector<cv::DMatch>& matches, const cv::Mat& mask=cv::Mat() );
  //  virtual void add( const vector<cv::Mat>& descriptors );
  //  virtual void clear();

  virtual bool isMaskSupported() const {return false;}
  virtual cv::Ptr<cv::DescriptorMatcher> clone( bool emptyTrainData=false ) const; 
};


/************************** INLINE METHODES ******************************/

inline float PSiftGPU::Distance128(float d1[128], float d2[128])
{
  float sqrDist=0.;

  for (unsigned i=0; i<128; i++)
    sqrDist += sqr(d1[i]-d2[i]);

  return sqrt(sqrDist);
}


}

#endif

