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

#include <v4r/reconstruction/KeypointPoseDetectorRT.h>
#include <v4r/common/impl/ScopeTime.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace v4r
{

using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
KeypointPoseDetectorRT::KeypointPoseDetectorRT(const Parameter &p,
    const v4r::FeatureDetector::Ptr &_detector,
    const v4r::FeatureDetector::Ptr &_descEstimator)
 : param(p), detector(_detector), descEstimator(_descEstimator)
{ 
  matcher = new cv::BFMatcher(cv::NORM_L2);
  //matcher = new cv::FlannBasedMatcher();

  rt.reset(new RigidTransformationRANSAC(param.rt_param));

  if (detector.get()==0) detector = descEstimator;
}

KeypointPoseDetectorRT::~KeypointPoseDetectorRT()
{
}






/******************************* PUBLIC ***************************************/

/**
 * detect
 */
double KeypointPoseDetectorRT::detect(const cv::Mat &image, const DataMatrix2D<Eigen::Vector3f> &cloud, Eigen::Matrix4f &pose)
{
  if (model.get()==0)
    throw std::runtime_error("[KeypointPoseDetectorRT::detect] No model available!");
  if (cloud.rows != image.rows || cloud.cols != image.cols)
    throw std::runtime_error("[KeypointPoseDetectorRT::detect] Invalid image/ cloud!");


  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;


  //{ v4r::ScopeTime t("detect keypoints");
  detector->detect(im_gray, keys);
  descEstimator->extract(im_gray, keys, descs);
  //}
  
  //matcher->knnMatch( descs, model->descs, matches, 2 );
  matcher->knnMatch( descs, matches, 2 );

  std::vector<int> ma_inliers;

  query_pts.clear();
  model_pts.clear();

  for (unsigned z=0; z<matches.size(); z++)
  {
    if (matches[z].size()>1)
    {
      cv::DMatch &ma0 = matches[z][0];
      if (ma0.distance/matches[z][1].distance < param.nnr)
      {
        const cv::Point2f &im_pt = keys[ma0.queryIdx].pt;
        const Eigen::Vector3f &pt = cloud(int(im_pt.y+.5),int(im_pt.x+.5));
        if (!isnan(pt[0]))
        {
          query_pts.push_back(pt);      
          if (param.compute_global_pose)
            model_pts.push_back(model->getPt(ma0.trainIdx).pt.cast<float>());
          else model_pts.push_back(model->cam_points[ma0.trainIdx]);
          ma_inliers.push_back(ma0.trainIdx);
        }
      }
    }
  }

  if (model_pts.size()<4) return 0.;

  rt->compute(model_pts, query_pts, pose, inliers);

  return double(inliers.size())/double(model->points.size());
}

/**
 * setModel
 */
void KeypointPoseDetectorRT::setModel(const ObjectView::Ptr &_model) 
{
  model=_model;

  matcher->clear();
  matcher->add(std::vector<cv::Mat>(1,model->descs));
  matcher->train();
}



}












