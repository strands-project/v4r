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

#ifndef KP_KEYPOINT_POSE_DETECTOR_RT_HH
#define KP_KEYPOINT_POSE_DETECTOR_RT_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <v4r/common/impl/SmartPtr.hpp>
#include <v4r/features/FeatureDetectorHeaders.h>
#include <v4r/keypoints/RigidTransformationRANSAC.h>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/keypoints/impl/Object.hpp>


namespace v4r
{


/**
 * KeypointPoseDetectorRT
 */
class KeypointPoseDetectorRT
{
public:
  class Parameter
  {
  public:
    float nnr;
    bool compute_global_pose;
    RigidTransformationRANSAC::Parameter rt_param; // 0.01 (slam: 0.03)
    Parameter(float _nnr=.9, bool _compute_global_pose=true,
      const RigidTransformationRANSAC::Parameter &_rt_param=RigidTransformationRANSAC::Parameter(0.01))
    : nnr(_nnr), compute_global_pose(_compute_global_pose),
      rt_param(_rt_param) {}
  };

private:
  Parameter param;

  cv::Mat_<unsigned char> im_gray;
  cv::Mat descs;
  std::vector<cv::KeyPoint> keys;
  std::vector< std::vector<cv::DMatch> > matches;
  std::vector< Eigen::Vector3f > query_pts;
  std::vector< Eigen::Vector3f > model_pts;
  std::vector< int> inliers;

  ObjectView::Ptr model;

  //cv::Ptr<cv::BFMatcher> matcher;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  RigidTransformationRANSAC::Ptr rt;
  
  v4r::FeatureDetector::Ptr detector;
  v4r::FeatureDetector::Ptr descEstimator;




public:
  cv::Mat dbg;

  KeypointPoseDetectorRT(const Parameter &p=Parameter(), 
    const v4r::FeatureDetector::Ptr &_detector=v4r::FeatureDetector::Ptr(),
    const v4r::FeatureDetector::Ptr &_descEstimator=new v4r::FeatureDetector_KD_FAST_IMGD(v4r::FeatureDetector_KD_FAST_IMGD::Parameter(1000, 1.44, 2, 17)));
  ~KeypointPoseDetectorRT();

  double detect(const cv::Mat &image, const v4r::DataMatrix2D<Eigen::Vector3f> &cloud, Eigen::Matrix4f &pose);

  void setModel(const ObjectView::Ptr &_model);


  typedef SmartPtr< ::v4r::KeypointPoseDetectorRT> Ptr;
  typedef SmartPtr< ::v4r::KeypointPoseDetectorRT const> ConstPtr;
};


/***************************** inline methods *******************************/






} //--END--

#endif

