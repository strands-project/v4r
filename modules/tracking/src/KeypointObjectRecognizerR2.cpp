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

#include <v4r/tracking/KeypointObjectRecognizerR2.h>
#include <v4r/reconstruction/impl/projectPointToImage.hpp>

namespace v4r
{

using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
KeypointObjectRecognizerR2::KeypointObjectRecognizerR2(const Parameter &p,
                                                       const v4r::FeatureDetector::Ptr &_detector,
                                                       const v4r::FeatureDetector::Ptr &_descEstimator)
 : param(p), detector(_detector), descEstimator(_descEstimator)
{ 
  sqr_inl_dist = param.inl_dist*param.inl_dist;
  if (detector.get()==0) detector = descEstimator;
  cbMatcher.reset(new CodebookMatcher(param.cb_param));
}

KeypointObjectRecognizerR2::~KeypointObjectRecognizerR2()
{
}

/**
 * getRandIdx
 */
void KeypointObjectRecognizerR2::getRandIdx(int size, int num, std::vector<int> &idx)
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
unsigned KeypointObjectRecognizerR2::countInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4f &pose)
{
  unsigned cnt=0;

  Eigen::Vector2f im_pt;
  Eigen::Vector3f pt3;
  bool have_dist = !dist_coeffs.empty();
  
  Eigen::Matrix3f R = pose.topLeftCorner<3, 3>();
  Eigen::Vector3f t = pose.block<3,1>(0, 3);

  for (unsigned i=0; i<points.size(); i++)
  {
    pt3 = R*Eigen::Map<const Eigen::Vector3f>(&points[i].x) + t;

    if (have_dist)
      projectPointToImage(&pt3[0], intrinsic.ptr<double>(), dist_coeffs.ptr<double>(), &im_pt[0]);
    else projectPointToImage(&pt3[0], intrinsic.ptr<double>(), &im_pt[0]);

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
void KeypointObjectRecognizerR2::getInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4f &pose, std::vector<int> &inliers)
{
  Eigen::Vector2f im_pt;
  Eigen::Vector3f pt3;
  bool have_dist = !dist_coeffs.empty();
  
  Eigen::Matrix3f R = pose.topLeftCorner<3, 3>();
  Eigen::Vector3f t = pose.block<3,1>(0, 3);

  inliers.clear();

  for (unsigned i=0; i<points.size(); i++)
  {
    pt3 = R*Eigen::Map<const Eigen::Vector3f>(&points[i].x) + t;

    if (have_dist)
      projectPointToImage(&pt3[0], intrinsic.ptr<double>(), dist_coeffs.ptr<double>(), &im_pt[0]);
    else projectPointToImage(&pt3[0], intrinsic.ptr<double>(), &im_pt[0]);

    if ((im_pt - Eigen::Map<const Eigen::Vector2f>(&im_points[i].x)).squaredNorm() < sqr_inl_dist)
    {
      inliers.push_back(i);
    }
  }
}


/**
 * ransacSolvePnP
 */
void KeypointObjectRecognizerR2::ransacSolvePnP(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, Eigen::Matrix4f &pose, std::vector<int> &inliers)
{
  int k=0;
  float sig=param.nb_ransac_points, sv_sig=0.;
  float eps = sig/(float)points.size();
  std::vector<int> indices;
  std::vector<cv::Point3f> model_pts(param.nb_ransac_points);
  std::vector<cv::Point2f> query_pts(param.nb_ransac_points);
  cv::Mat_<double> R(3,3), rvec, tvec, sv_rvec, sv_tvec;
  inliers.clear();

  while (pow(1. - pow(eps,param.nb_ransac_points), k) >= param.eta_ransac && k < (int)param.max_rand_trials)
  {
    getRandIdx(points.size(), param.nb_ransac_points, indices);

    for (unsigned i=0; i<indices.size(); i++)
    {
      model_pts[i] = points[indices[i]];
      query_pts[i] = im_points[indices[i]];
    }

    cv::solvePnP(cv::Mat(model_pts), cv::Mat(query_pts), intrinsic, dist_coeffs, rvec, tvec, false, param.pnp_method);

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

  cv::solvePnP(cv::Mat(model_pts), cv::Mat(query_pts), intrinsic, dist_coeffs, sv_rvec, sv_tvec, true, cv::ITERATIVE );

  cv::Rodrigues(sv_rvec, R);
  cvToEigen(R, sv_tvec, pose);


  //if (!dbg.empty()) cout<<"Num ransac trials: "<<k<<endl;
}




/******************************* PUBLIC ***************************************/

/**
 * detect
 */
double KeypointObjectRecognizerR2::detect(const cv::Mat &image, Eigen::Matrix4f &pose, int &view_idx)
{
  if (model.get()==0)
    throw std::runtime_error("[KeypointObjectRecognizerR2::detect] No model available!");
  if (intrinsic.empty())
    throw std::runtime_error("[KeypointObjectRecognizerR2::detect] Intrinsic camera parameter not set!");

  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;

  Object &m = *model;

  // get matches
  detector->detect(im_gray, keys);
  descEstimator->extract(im_gray, keys, descs);

  cbMatcher->queryMatches(descs, matches);

  // select points
  std::vector<bool> use_views(cbMatcher->getViewRank().size(),false);
  std::vector<int> view_indices;
  std::vector<int> view_votes;
  int max = 0;
  view_idx= -1;


  for (unsigned i=0; i<cbMatcher->getViewRank().size()&&i<(unsigned)param.use_n_views; i++)
    use_views[cbMatcher->getViewRank()[i].first] = true;

  model_pts.clear();
  query_pts.clear();

  for (unsigned i=0; i<matches.size(); i++)
  {
    const std::vector<cv::DMatch> &ms = matches[i];

    for (unsigned j=0; j<ms.size(); j++)
    {
      if (use_views[ms[j].imgIdx])
      {
        const cv::DMatch &ma0 = ms[j];
        const Eigen::Vector3d &pt = m.points[m.views[ma0.imgIdx]->points[ma0.trainIdx]].pt;
        model_pts.push_back(cv::Point3f(pt[0],pt[1],pt[2]));
        query_pts.push_back(keys[ma0.queryIdx].pt);
        view_indices.push_back(ma0.imgIdx);
        if (!dbg.empty()) cv::circle(dbg, query_pts.back(), 3, CV_RGB(255,255,255), -1);
      }
    }
  }

  if (int(query_pts.size())<4) return 0.;

  ransacSolvePnP(model_pts, query_pts, pose, inliers);

  // get best view
  view_votes.assign(use_views.size(),0);

  for (unsigned i=0; i<inliers.size(); i++)
  {
    view_votes[view_indices[inliers[i]]]++;
  }

  for (unsigned i=0; i<view_votes.size(); i++)
  {
    if (view_votes[i]>max)
    {
      view_idx = (int)i;
      max = view_votes[i];
    }
  }

  // debug draw
  if (!dbg.empty()) {
    for (unsigned i=0; i<inliers.size(); i++)
      cv::circle(dbg, query_pts[inliers[i]], 3, CV_RGB(0,255,0), -1);
  }

  int norm = (inliers.size()>200?inliers.size():200);    // that's a hard coded valued for normalization

  return double(inliers.size())/double(norm);
}


/**
 * setModel
 */
void KeypointObjectRecognizerR2::setModel(const Object::Ptr &_model)
{  
  model=_model; 

  if (model->haveCodebook())
  {
    cbMatcher->setCodebook(model->cb_centers, model->cb_entries);
  }
  else
  {
    // create codebook
    cbMatcher->clear();

    for (unsigned i=0; i<model->views.size(); i++)
      cbMatcher->addView(model->views[i]->descs, i);

    cbMatcher->createCodebook();
  }
}


/**
 * setSourceCameraParameter
 */
void KeypointObjectRecognizerR2::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
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












