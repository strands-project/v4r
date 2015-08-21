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

#ifndef KP_LK_POSE_TRACKER_RT_HH
#define KP_LK_POSE_TRACKER_RT_HH

#include <stdio.h>
#include <iostream>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Dense>
#include "v4r/common/impl/SmartPtr.hpp"
#include <v4r/keypoints/RigidTransformationRANSAC.h>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/keypoints/impl/Object.hpp>


namespace v4r
{


/**
 * LKPoseTrackerRT
 */
class LKPoseTrackerRT
{
public:
  class Parameter
  {
  public:
    bool compute_global_pose;
    cv::Size win_size;
    int max_level;
    cv::TermCriteria termcrit;
    float max_error;                     //100
    RigidTransformationRANSAC::Parameter rt_param; // 0.01 (slam: 0.03)
    Parameter(bool _compute_global_pose=true, const cv::Size &_win_size=cv::Size(21,21), int _max_level=2,
      const cv::TermCriteria &_termcrit=cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03), 
      float _max_error=100, 
      const RigidTransformationRANSAC::Parameter &_rt_param=RigidTransformationRANSAC::Parameter(0.01))
    : compute_global_pose(_compute_global_pose), win_size(_win_size), max_level(_max_level),
      termcrit(_termcrit),
      max_error(_max_error),
      rt_param(_rt_param) {}
  };

private:
  Parameter param;

  cv::Mat_<unsigned char> im_gray, im_last;
  std::vector< cv::Point2f > im_points0, im_points1;
  std::vector< int > inliers;
  Eigen::Matrix4f last_pose;

  bool have_im_last;

  std::vector<unsigned char> status;
  std::vector<float> error;

  cv::Mat_<double> dist_coeffs;
  cv::Mat_<double> intrinsic;


  ObjectView::Ptr model;

  RigidTransformationRANSAC::Ptr rt;



public:
  cv::Mat dbg;

  LKPoseTrackerRT(const Parameter &p=Parameter());
  ~LKPoseTrackerRT();

  double detectIncremental(const cv::Mat &im, const DataMatrix2D<Eigen::Vector3f> &cloud, Eigen::Matrix4f &pose);
  void setLastFrame(const cv::Mat &image, const Eigen::Matrix4f &pose);


  void setModel(const ObjectView::Ptr &_model);
  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);

  void getProjections(std::vector< std::pair<int,cv::Point2f> > &im_pts);

  typedef SmartPtr< ::v4r::LKPoseTrackerRT> Ptr;
  typedef SmartPtr< ::v4r::LKPoseTrackerRT const> ConstPtr;
};


/***************************** inline methods *******************************/



} //--END--

#endif

