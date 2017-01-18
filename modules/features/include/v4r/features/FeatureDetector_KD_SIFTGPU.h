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

#ifndef KP_FEATURE_DETECTOR_SIFTGPU_HH
#define KP_FEATURE_DETECTOR_SIFTGPU_HH

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <boost/shared_ptr.hpp>

#include <v4r/features/FeatureDetector.h>
#include <v4r/features/PSiftGPU.h>


namespace v4r
{

class V4R_EXPORTS FeatureDetector_KD_SIFTGPU : public FeatureDetector
{
public:
  class Parameter
  {
  public:
    float distmax;         // absolute descriptor distance (e.g. = 0.6)
    float ratiomax;        // compare best match with second best (e.g. =0.8)
    int mutual_best_match; // compare forward/backward matches (1)
    bool computeRootSIFT;  // L1 norm and square root => euc dist = hellinger dist 
    Parameter(float d=FLT_MAX, float r=1., int m=0, 
      bool _computeRootSIFT=true)
    : distmax(d), ratiomax(r), mutual_best_match(m), 
      computeRootSIFT(_computeRootSIFT) {}
  };

private:
  PSiftGPU::Parameter param;
  cv::Mat_<unsigned char> im_gray;  

  cv::Ptr<v4r::PSiftGPU> sift;

public:
  FeatureDetector_KD_SIFTGPU(const Parameter &_p=Parameter());
  ~FeatureDetector_KD_SIFTGPU();

  virtual void detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors); 
  virtual void detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys); 
  virtual void extract(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors); 

  typedef boost::shared_ptr< ::v4r::FeatureDetector_KD_SIFTGPU> Ptr;
  typedef boost::shared_ptr< ::v4r::FeatureDetector_KD_SIFTGPU const> ConstPtr;
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

