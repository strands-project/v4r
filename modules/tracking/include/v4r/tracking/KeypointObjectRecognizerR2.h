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

#ifndef KP_OBJECT_RECOGNIZER_R2_HH
#define KP_OBJECT_RECOGNIZER_R2_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Dense>
#include <v4r/common/impl/SmartPtr.hpp>
#include <v4r/keypoints/CodebookMatcher.h>
#include <v4r/features/FeatureDetectorHeaders.h>
#include <v4r/keypoints/impl/Object.hpp>
#include <v4r/core/macros.h>


namespace v4r
{


/**
 * KeypointObjectRecognizerR2
 */
class V4R_EXPORTS KeypointObjectRecognizerR2
{
public:
  class Parameter
  {
  public:
    double inl_dist;
    double eta_ransac;               // eta for pose ransac
    unsigned max_rand_trials;         // max. number of trials for pose ransac
    int pnp_method;            // cv::ITERATIVE, cv::P3P
    int nb_ransac_points;
    int use_n_views;
    CodebookMatcher::Parameter cb_param;
    Parameter(double _inl_dist=3, double _eta_ransac=0.01, unsigned _max_rand_trials=5000,
      int _pnp_method=cv::P3P, int _nb_ransac_points=4, int _use_n_views=3,
      const CodebookMatcher::Parameter &_cb_param=CodebookMatcher::Parameter())
    : inl_dist(_inl_dist), eta_ransac(_eta_ransac), max_rand_trials(_max_rand_trials),
      pnp_method(_pnp_method), nb_ransac_points(_nb_ransac_points), use_n_views(_use_n_views),
      cb_param(_cb_param){}
  };

private:
  Parameter param;

  float sqr_inl_dist;

  cv::Mat_<double> dist_coeffs;
  cv::Mat_<double> intrinsic;
  
  cv::Mat_<unsigned char> im_gray;
  std::vector< cv::Point2f > im_points;
  std::vector< int > inliers;

  cv::Mat descs;
  std::vector<cv::KeyPoint> keys;
  std::vector< std::vector< cv::DMatch > > matches;
  std::vector< cv::Point2f > query_pts;
  std::vector< cv::Point3f > model_pts;

  Object::Ptr model;

  CodebookMatcher::Ptr cbMatcher;
  v4r::FeatureDetector::Ptr detector;
  v4r::FeatureDetector::Ptr descEstimator;

  void ransacSolvePnP(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, Eigen::Matrix4f &pose, std::vector<int> &inliers);
  void getRandIdx(int size, int num, std::vector<int> &idx);
  unsigned countInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4f &pose);
  void getInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4f &pose, std::vector<int> &inliers);

  inline void cvToEigen(const cv::Mat_<double> &R, const cv::Mat_<double> &t, Eigen::Matrix4f &pose); 
  inline bool contains(const std::vector<int> &idx, int num);



public:
  cv::Mat dbg;

  KeypointObjectRecognizerR2(const Parameter &p=Parameter(),
                             const v4r::FeatureDetector::Ptr &_detector=v4r::FeatureDetector::Ptr(),
                             const v4r::FeatureDetector::Ptr &_descEstimator=new v4r::FeatureDetector_KD_FAST_IMGD(v4r::FeatureDetector_KD_FAST_IMGD::Parameter(10000, 1.44, 2, 17)));
  ~KeypointObjectRecognizerR2();

  double detect(const cv::Mat &image, Eigen::Matrix4f &pose, int &view_idx);

  void setModel(const Object::Ptr &_model);

  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);

  typedef SmartPtr< ::v4r::KeypointObjectRecognizerR2> Ptr;
  typedef SmartPtr< ::v4r::KeypointObjectRecognizerR2 const> ConstPtr;
};


/***************************** inline methods *******************************/
/**
 * cvToEigen
 */
inline void KeypointObjectRecognizerR2::cvToEigen(const cv::Mat_<double> &R, const cv::Mat_<double> &t, Eigen::Matrix4f &pose)
{
  pose.setIdentity();

  pose(0,0) = R(0,0); pose(0,1) = R(0,1); pose(0,2) = R(0,2);
  pose(1,0) = R(1,0); pose(1,1) = R(1,1); pose(1,2) = R(1,2);
  pose(2,0) = R(2,0); pose(2,1) = R(2,1); pose(2,2) = R(2,2);

  pose(0,3) = t(0,0);
  pose(1,3) = t(1,0);
  pose(2,3) = t(2,0);
}

inline bool KeypointObjectRecognizerR2::contains(const std::vector<int> &idx, int num)
{
  for (unsigned i=0; i<idx.size(); i++)
    if (idx[i]==num)
      return true;
  return false;
}




} //--END--

#endif

