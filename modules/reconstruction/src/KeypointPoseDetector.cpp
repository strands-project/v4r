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

#include <v4r/reconstruction/KeypointPoseDetector.h>
#include <v4r/common/impl/ScopeTime.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace v4r
{

using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
KeypointPoseDetector::KeypointPoseDetector(const Parameter &p,
    const v4r::FeatureDetector::Ptr &_detector,
    const v4r::FeatureDetector::Ptr &_descEstimator)
 : param(p), detector(_detector), descEstimator(_descEstimator)
{ 
  matcher = new cv::BFMatcher(cv::NORM_L2);
  //matcher = new cv::FlannBasedMatcher();

  if (detector.get()==0) detector = descEstimator;
}

KeypointPoseDetector::~KeypointPoseDetector()
{
}






/******************************* PUBLIC ***************************************/

/**
 * detect
 */
double KeypointPoseDetector::detect(const cv::Mat &image, Eigen::Matrix4f &pose)
{
  if (model.get()==0)
    throw std::runtime_error("[KeypointPoseDetector::detect] No model available!");
  if (intrinsic.empty())
    throw std::runtime_error("[KeypointPoseDetector::detect] Intrinsic camera parameter not set!");


  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;


  cv::Mat_<double> R(3,3), rvec, tvec;
  std::vector<cv::Point3f> model_pts2;
  std::vector<cv::Point2f> query_pts2;
  std::vector<int> ma_inliers;

  //{ v4r::ScopeTime t("detect keypoints");
  detector->detect(im_gray, keys);
  descEstimator->extract(im_gray, keys, descs);
  //}
  
  //matcher->knnMatch( descs, model->descs, matches, 2 );
  matcher->knnMatch( descs, matches, 2 );

  query_pts.clear();
  model_pts.clear();

  for (unsigned z=0; z<matches.size(); z++)
  {
    if (matches[z].size()>1)
    {
      cv::DMatch &ma0 = matches[z][0];
      if (ma0.distance/matches[z][1].distance < param.nnr)
      {
        const Eigen::Vector3d &pt = model->getPt(ma0.trainIdx).pt;
        query_pts.push_back(keys[ma0.queryIdx].pt);
        model_pts.push_back(cv::Point3f(pt[0],pt[1],pt[2]));
        ma_inliers.push_back(ma0.trainIdx);
      }
    }
  }

  if (model_pts.size()<4) return 0.;

  cv::solvePnPRansac(cv::Mat(model_pts), cv::Mat(query_pts), intrinsic, dist_coeffs, rvec, tvec, false, param.iterationsCount, param.reprojectionError, param.minInliersCount, inliers, cv::P3P );

  if (inliers.size()<4) return 0.;

  // refine pose
  model_pts2.resize(inliers.size());
  query_pts2.resize(inliers.size());

  for (unsigned i=0; i<inliers.size(); i++)
  {
    model_pts2[i] = model_pts[inliers[i]];
    query_pts2[i] = query_pts[inliers[i]];
    if (!dbg.empty()) cv::line(dbg,model->keys[ma_inliers[inliers[i]]].pt, query_pts2[i],CV_RGB(255,0,0));
    if (!dbg.empty()) cv::circle(dbg,query_pts2[i],2,CV_RGB(0,255,0));
  }
  cv::solvePnP(cv::Mat(model_pts2), cv::Mat(query_pts2), intrinsic, dist_coeffs, rvec, tvec, true, cv::ITERATIVE );


  cv::Rodrigues(rvec, R);
  cvToEigen(R, tvec, pose);  

  return double(inliers.size())/double(model->points.size());
}

/**
 * setModel
 */
void KeypointPoseDetector::setModel(const ObjectView::Ptr &_model) 
{
  model=_model;

  matcher->clear();
  matcher->add(std::vector<cv::Mat>(1,model->descs));
  matcher->train();
}


/**
 * setCameraParameter
 */
void KeypointPoseDetector::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  dist_coeffs = cv::Mat_<double>();

  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic=_intrinsic;

  if (!_dist_coeffs.empty())
  {
    dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols*_dist_coeffs.rows; i++)
      dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }
}


}












