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

#include <v4r/reconstruction/ProjLKPoseTrackerR2.h>
#include "opencv2/video/tracking.hpp"
#include <v4r/reconstruction/impl/projectPointToImage.hpp>

namespace v4r
{

using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
ProjLKPoseTrackerR2::ProjLKPoseTrackerR2(const Parameter &p)
 : param(p)
{ 
  plk.reset(new RefineProjectedPointLocationLK(p.plk_param) );

  sqr_inl_dist = param.inl_dist*param.inl_dist;
}

ProjLKPoseTrackerR2::~ProjLKPoseTrackerR2()
{
}

/**
 * getRandIdx
 */
void ProjLKPoseTrackerR2::getRandIdx(int size, int num, std::vector<int> &idx)
{
  int temp;
  idx.clear();
  for (int i=0; i<num; i++)
  {
    do{
      temp = rand()%size;
    }while(contains(idx,temp));
    idx.push_back(temp);
  }
}

/**
 * countInliers
 */
unsigned ProjLKPoseTrackerR2::countInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4f &pose)
{
  unsigned cnt=0;

  Eigen::Vector2f im_pt;
  Eigen::Vector3f pt3;
  bool have_dist = !tgt_dist_coeffs.empty();
  
  Eigen::Matrix3f R = pose.topLeftCorner<3, 3>();
  Eigen::Vector3f t = pose.block<3,1>(0, 3);

  for (unsigned i=0; i<points.size(); i++)
  {
    pt3 = R*Eigen::Map<const Eigen::Vector3f>(&points[i].x) + t;

    if (have_dist)
      projectPointToImage(&pt3[0], tgt_intrinsic.ptr<double>(), tgt_dist_coeffs.ptr<double>(), &im_pt[0]);
    else projectPointToImage(&pt3[0], tgt_intrinsic.ptr<double>(), &im_pt[0]);

    if ((im_pt - Eigen::Map<const Eigen::Vector2f>(&im_points[i].x)).squaredNorm() < sqr_inl_dist)
    {
      cnt++;
    }
  }

  return cnt;
}

/**
 * getInliers
 */
void ProjLKPoseTrackerR2::getInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4f &pose, std::vector<int> &inliers)
{
  Eigen::Vector2f im_pt;
  Eigen::Vector3f pt3;
  bool have_dist = !tgt_dist_coeffs.empty();
  
  Eigen::Matrix3f R = pose.topLeftCorner<3, 3>();
  Eigen::Vector3f t = pose.block<3,1>(0, 3);

  inliers.clear();

  for (unsigned i=0; i<points.size(); i++)
  {
    pt3 = R*Eigen::Map<const Eigen::Vector3f>(&points[i].x) + t;

    if (have_dist)
      projectPointToImage(&pt3[0], tgt_intrinsic.ptr<double>(), tgt_dist_coeffs.ptr<double>(), &im_pt[0]);
    else projectPointToImage(&pt3[0], tgt_intrinsic.ptr<double>(), &im_pt[0]);

    if ((im_pt - Eigen::Map<const Eigen::Vector2f>(&im_points[i].x)).squaredNorm() < sqr_inl_dist)
    {
      inliers.push_back(i);
    }
  }
}


/**
 * ransacSolvePnP
 */
void ProjLKPoseTrackerR2::ransacSolvePnP(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, Eigen::Matrix4f &pose, std::vector<int> &inliers)
{
  int k=0;
  float sig=param.nb_ransac_points, sv_sig=0.;
  float eps = sig/(float)points.size();
  std::vector<int> indices;
  std::vector<cv::Point3f> model_pts(param.nb_ransac_points);
  std::vector<cv::Point2f> query_pts(param.nb_ransac_points);
  cv::Mat_<double> R(3,3), rvec, tvec, sv_rvec, sv_tvec;

  while (pow(1. - pow(eps,param.nb_ransac_points), k) >= param.eta_ransac && k < (int)param.max_rand_trials)
  {
    getRandIdx(points.size(), param.nb_ransac_points, indices);

    for (unsigned i=0; i<indices.size(); i++)
    {
      model_pts[i] = points[indices[i]];
      query_pts[i] = im_points[indices[i]];
    }

    cv::solvePnP(cv::Mat(model_pts), cv::Mat(query_pts), tgt_intrinsic, tgt_dist_coeffs, rvec, tvec, false, param.pnp_method);

    cv::Rodrigues(rvec, R);
    cvToEigen(R, tvec, pose);  
  
    sig = countInliers(points, im_points, pose);

    if (sig > sv_sig)
    {
      sv_sig = sig;
      rvec.copyTo(sv_rvec);
      tvec.copyTo(sv_tvec);

      eps = sv_sig / (float)points.size();
    }

    k++;
  }

  if (sv_sig<4) return;

  cv::Rodrigues(sv_rvec, R);
  cvToEigen(R, sv_tvec, pose);
  getInliers(points, im_points, pose, inliers);

  model_pts.resize(inliers.size());
  query_pts.resize(inliers.size());

  for (unsigned i=0; i<inliers.size(); i++)
  {
    model_pts[i] = points[inliers[i]];
    query_pts[i] = im_points[inliers[i]];
  }

  cv::solvePnP(cv::Mat(model_pts), cv::Mat(query_pts), tgt_intrinsic, tgt_dist_coeffs, sv_rvec, sv_tvec, true, cv::ITERATIVE );

  cv::Rodrigues(sv_rvec, R);
  cvToEigen(R, sv_tvec, pose);


  //if (!dbg.empty()) cout<<"Num ransac trials: "<<k<<endl;
}




/******************************* PUBLIC ***************************************/

/**
 * detect
 */
double ProjLKPoseTrackerR2::detect(const cv::Mat &image, Eigen::Matrix4f &pose)
{
  if (model.get()==0)
    throw std::runtime_error("[ProjLKPoseTrackerR2::detect] No model available!");
  if (src_intrinsic.empty()||tgt_intrinsic.empty())
    throw std::runtime_error("[ProjLKPoseTrackerR2::detect] Intrinsic camera parameter not set!");


  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;

  ObjectView &m = *model;

  cv::Mat_<double> R(3,3), rvec, tvec;
  std::vector<cv::Point3f> model_pts;
  std::vector<cv::Point2f> query_pts;
  std::vector<Eigen::Vector3f> points, normals;
  std::vector<int> lk_inliers, pnp_inliers;

  m.getPoints(points);
  m.getNormals(normals);

  inliers.clear();

  // tracking
  plk->setTargetImage(im_gray,pose);

  plk->refineImagePoints(points, normals, im_points, converged);

  for (unsigned z=0; z<im_points.size(); z++)
  {
    if (converged[z]==1)
    {
      lk_inliers.push_back(z);
      const Eigen::Vector3f &pt = points[z];
      query_pts.push_back(im_points[z]);
      model_pts.push_back(cv::Point3f(pt[0],pt[1],pt[2]));
      if (!dbg.empty()) cv::line(dbg,model->keys[z].pt, im_points[z],CV_RGB(0,0,0));
    }
  }

  if (int(query_pts.size())<4) return 0.;

  ransacSolvePnP(model_pts, query_pts, pose, pnp_inliers);

  if (int(pnp_inliers.size())<4) return 0.;

  for (unsigned i=0; i<pnp_inliers.size(); i++) {
    inliers.push_back(lk_inliers[pnp_inliers[i]]);
    if (!dbg.empty()) cv::circle(dbg,im_points[inliers.back()],2,CV_RGB(255,255,0));
  }

  return double(inliers.size())/double(model->points.size());
}

/**
 * getProjections
 * @param im_pts <model_point_index, projection>
 */
void ProjLKPoseTrackerR2::getProjections(std::vector< std::pair<int,cv::Point2f> > &im_pts)
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
void ProjLKPoseTrackerR2::setModel(const ObjectView::Ptr &_model, const Eigen::Matrix4f &_pose) 
{  
  model=_model; 

  im_points.resize(model->keys.size());
  plk->setSourceImage(model->image,_pose);
}


/**
 * setSourceCameraParameter
 */
void ProjLKPoseTrackerR2::setSourceCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
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
void ProjLKPoseTrackerR2::setTargetCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
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












