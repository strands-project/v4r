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

#include <v4r/reconstruction/ProjLKPoseTrackerRT.h>
#include "opencv2/video/tracking.hpp"
#include <v4r/reconstruction/impl/projectPointToImage.hpp>

namespace v4r
{

using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
ProjLKPoseTrackerRT::ProjLKPoseTrackerRT(const Parameter &p)
 : param(p)
{ 
  plk.reset(new RefineProjectedPointLocationLK(p.plk_param) );
  rt.reset(new RigidTransformationRANSAC(param.rt_param));

}

ProjLKPoseTrackerRT::~ProjLKPoseTrackerRT()
{
}




/******************************* PUBLIC ***************************************/

/**
 * detect
 */
double ProjLKPoseTrackerRT::detect(const cv::Mat &image, const DataMatrix2D<Eigen::Vector3f> &cloud, Eigen::Matrix4f &pose)
{
  if (model.get()==0)
    throw std::runtime_error("[ProjLKPoseTrackerRT::detect] No model available!");
  if (src_intrinsic.empty()||tgt_intrinsic.empty())
    throw std::runtime_error("[ProjLKPoseTrackerRT::detect] Intrinsic camera parameter not set!");


  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;

  ObjectView &m = *model;

  cv::Mat_<double> R(3,3), rvec, tvec;
  std::vector<Eigen::Vector3f> points, normals;
  std::vector<int> lk_inliers, rt_inliers;
  model_pts.clear();
  query_pts.clear();

  if (param.compute_global_pose)
  {
    m.getPoints(points);
    m.getNormals(normals);
  }
  else
  {
    points = m.cam_points;
    m.getNormals(normals);
    Eigen::Matrix3f R = model->getCamera().topLeftCorner<3,3>();
    for (unsigned i=0; i<normals.size(); i++)
      normals[i] = R*normals[i];
  }

  inliers.clear();

  // tracking
  plk->setTargetImage(im_gray,pose);

  plk->refineImagePoints(points, normals, im_points, converged);

  for (unsigned z=0; z<im_points.size(); z++)
  {
    if (converged[z]==1)
    {
      const cv::Point2f &im_pt = im_points[z];
      const Eigen::Vector3f &pt = cloud(int(im_pt.y+.5),int(im_pt.x+.5));
      if (!isnan(pt[0]))
      {
        lk_inliers.push_back(z);
        query_pts.push_back(pt);
        model_pts.push_back(points[z]);
        if (!dbg.empty()) cv::line(dbg,model->keys[z].pt, im_points[z],CV_RGB(0,0,0));
      }
    }
  }


  if (int(query_pts.size())<4) return 0.;

  rt->compute(model_pts, query_pts, pose, rt_inliers);

  for (unsigned i=0; i<rt_inliers.size(); i++) {
    inliers.push_back(lk_inliers[rt_inliers[i]]);
    if (!dbg.empty()) cv::circle(dbg,im_points[inliers.back()],2,CV_RGB(255,255,0));
  }


  return double(inliers.size())/double(model->points.size());
}

/**
 * getProjections
 * @param im_pts <model_point_index, projection>
 */
void ProjLKPoseTrackerRT::getProjections(std::vector< std::pair<int,cv::Point2f> > &im_pts)
{
  im_pts.clear();
  if (model.get()==0 || im_points.size()!=model->points.size() || inliers.size()>im_points.size())
    return;

  for (unsigned i=0; i<inliers.size(); i++)
    im_pts.push_back(make_pair(inliers[i],im_points[inliers[i]]));
}

/**
 * setModel
 */
void ProjLKPoseTrackerRT::setModel(const ObjectView::Ptr &_model, const Eigen::Matrix4f &_pose) 
{  
  model=_model; 
  im_points.resize(model->keys.size());

  if (param.compute_global_pose)
    plk->setSourceImage(model->image,_pose);
  else plk->setSourceImage(model->image,Eigen::Matrix4f::Identity());
}


/**
 * setSourceCameraParameter
 */
void ProjLKPoseTrackerRT::setSourceCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  src_dist_coeffs = cv::Mat_<double>();
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(src_intrinsic, CV_64F);
  else src_intrinsic = _intrinsic;
  if (!_dist_coeffs.empty())
  {
    src_dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols*_dist_coeffs.rows; i++)
      src_dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }
  plk->setSourceCameraParameter(src_intrinsic,src_dist_coeffs);
}

/**
 * setTargetCameraParameter
 */
void ProjLKPoseTrackerRT::setTargetCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  tgt_dist_coeffs = cv::Mat_<double>();
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(tgt_intrinsic, CV_64F);
  else tgt_intrinsic = _intrinsic;
  if (!_dist_coeffs.empty())
  {
    tgt_dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols*_dist_coeffs.rows; i++)
      tgt_dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }
  plk->setTargetCameraParameter(tgt_intrinsic,tgt_dist_coeffs);
}



}












