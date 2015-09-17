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

#include <v4r/reconstruction/LKPoseTrackerRT.h>
#include "opencv2/video/tracking.hpp"
#include <v4r/reconstruction/impl/projectPointToImage.hpp>


namespace v4r
{

using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
LKPoseTrackerRT::LKPoseTrackerRT(const Parameter &p)
 : param(p), last_pose(Eigen::Matrix4f::Identity()), have_im_last(false)
{ 
  rt.reset(new RigidTransformationRANSAC(param.rt_param));
}

LKPoseTrackerRT::~LKPoseTrackerRT()
{
}






/******************************* PUBLIC ***************************************/

/**
 * detect
 */
double LKPoseTrackerRT::detectIncremental(const cv::Mat &image, const DataMatrix2D<Eigen::Vector3f> &cloud, Eigen::Matrix4f &pose)
{
  if (model.get()==0)
    throw std::runtime_error("[LKPoseTrackerRT::detect] No model available!");
  if (intrinsic.empty())
    throw std::runtime_error("[LKPoseTrackerRT::detect] Intrinsic camera parameter not set!");

  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;

  if (!have_im_last || im_last.empty()) {
    //last_pose = pose;
    //im_gray.copyTo(im_last);
    return 0.;
  }

  have_im_last = false;

  ObjectView &m = *model;

  cv::Mat_<double> R(3,3), rvec, tvec;
  std::vector<Eigen::Vector3f> points;
  std::vector<Eigen::Vector3f> model_pts;
  std::vector<Eigen::Vector3f> query_pts;
  std::vector<int> lk_inliers, rt_inliers;

  if (param.compute_global_pose) 
    m.getPoints(points);
  else points = m.cam_points;

  inliers.clear();

  Eigen::Matrix3f pose_R = last_pose.topLeftCorner<3, 3>();
  Eigen::Vector3f pose_t = last_pose.block<3,1>(0, 3);
  Eigen::Vector3f pt3;
  bool have_dist = !dist_coeffs.empty();

  for (unsigned i=0; i<points.size(); i++)
  {
    pt3 = pose_R*points[i] + pose_t;

    if (have_dist)
      projectPointToImage(&pt3[0], intrinsic.ptr<double>(), dist_coeffs.ptr<double>(), &im_points0[i].x);
    else projectPointToImage(&pt3[0], intrinsic.ptr<double>(), &im_points0[i].x);
  }

  cv::calcOpticalFlowPyrLK(im_last, im_gray, im_points0, im_points1, status, error, param.win_size, param.max_level, param.termcrit, 0, 0.001 );

  for (unsigned i=0; i<im_points0.size(); i++)
  {
    const cv::Point2f &im_pt = im_points1[i];

    if (status[i]!=0 && error[i]<param.max_error &&
        im_pt.x>=0 && im_pt.y>=0 && im_pt.x+.5<cloud.cols && im_pt.y+.5<cloud.rows)
    {
      const Eigen::Vector3f &pt3 = cloud(int(im_pt.y+.5),int(im_pt.x+.5));
      if (!isnan(pt3[0]))
      { 
        lk_inliers.push_back(i);
        model_pts.push_back(points[i]);
        query_pts.push_back(pt3);
        if (!dbg.empty()) cv::line(dbg,im_points0[i], im_points1[i],CV_RGB(0,0,0));
      }
    }
  }

  if (int(query_pts.size())<4) {
    //last_pose = pose;
    //im_gray.copyTo(im_last);
    return 0.;
  }


  rt->compute(model_pts, query_pts, pose, rt_inliers);

  //last_pose = pose;
  //im_gray.copyTo(im_last);

  if (int(rt_inliers.size())<4) 
    return 0.;

  for (unsigned i=0; i<rt_inliers.size(); i++) {
    inliers.push_back(lk_inliers[rt_inliers[i]]);
    if (!dbg.empty()) cv::circle(dbg,im_points1[inliers.back()],2,CV_RGB(255,255,0));
  }

  return double(inliers.size())/double(model->points.size());
}


/**
 * setLastFrame
 */
void LKPoseTrackerRT::setLastFrame(const cv::Mat &image, const Eigen::Matrix4f &pose)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_last, CV_RGB2GRAY );
  else image.copyTo(im_last);

  last_pose = pose;
  have_im_last = true;
}



/**
 * getProjections
 * @param im_pts <model_point_index, projection>
 */
void LKPoseTrackerRT::getProjections(std::vector< std::pair<int,cv::Point2f> > &im_pts)
{
  im_pts.clear();
  if (model.get()==0 || im_points1.size()!=model->keys.size() || inliers.size()>im_points1.size())
    return;

  for (unsigned i=0; i<inliers.size(); i++)
    im_pts.push_back(make_pair(inliers[i],im_points1[inliers[i]]));
}

/**
 * setModel
 */
void LKPoseTrackerRT::setModel(const ObjectView::Ptr &_model) 
{  
  if (_model->points.size()!=_model->keys.size())
    throw std::runtime_error("[LKPoseTrackerRT::setModel] Invalid model!");

  model=_model; 
  ObjectView &view = *model;

  im_points0.resize(view.keys.size());
  im_points1.resize(view.keys.size());
}

/**
 * setCameraParameter
 */
void LKPoseTrackerRT::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  dist_coeffs = cv::Mat_<double>();
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;
  if (!_dist_coeffs.empty())
  {
    dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols*_dist_coeffs.rows; i++)
      dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }
}



}












