/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#include "LKHomographyTracker.hh"
#include "opencv2/video/tracking.hpp"

namespace kp
{

using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
LKHomographyTracker::LKHomographyTracker(const Parameter &p)
 : param(p)
{ 
  imr.reset(new ImageTransformRANSAC(param.imr_param));
}

LKHomographyTracker::~LKHomographyTracker()
{
}






/******************************* PUBLIC ***************************************/

/**
 * detect
 */
double LKHomographyTracker::detect(const cv::Mat &image, Eigen::Matrix3f &transform)
{
  if (model.get()==0)
    throw std::runtime_error("[LKHomographyTracker::detect] No model available!");

  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;

  if (im_points0.size()<4) return 0.;

  cv::calcOpticalFlowPyrLK(model->image, im_gray, im_points0, im_points1, status, error, param.win_size, param.max_level, param.termcrit, cv::OPTFLOW_USE_INITIAL_FLOW, 0.001 );

  inliers.clear();

  if (param.outlier_rejection_method==0)
  {
    for (unsigned i=0; i<im_points0.size(); i++)
    {
      if (status[i]!=0 && error[i]<param.max_error)
        inliers.push_back(i);
      else im_points1[i] = im_points0[i];
    }
  } 
  else 
  {
    pts0.clear();
    pts1.clear();
    std::vector<int> lt, inls, map(im_points0.size(),0);

    // convert
    for (unsigned i=0; i<im_points0.size(); i++)
    {
      if (status[i]!=0 && error[i]<param.max_error)
      {
        pts0.push_back(Eigen::Map<Eigen::Vector2f>(&im_points0[i].x));
        pts1.push_back(Eigen::Map<Eigen::Vector2f>(&im_points1[i].x));
        lt.push_back(int(i));
      }
    }

    if (lt.size()<4) return 0.;

    // compute homography
    if (param.outlier_rejection_method==1)
    {
      imr->ransacAffine(pts0,pts1,transform, inls);
    }
    else if (param.outlier_rejection_method==2)
    {
      imr->ransacHomography(pts0,pts1,transform, inls);
    }

    inliers.resize(inls.size());

    for (unsigned i=0; i<inls.size(); i++)
    {
      inliers[i] = lt[inls[i]];
      map[inliers[i]] = 1;
    }

    if (!dbg.empty()) { //-- dbg draw --
      for (unsigned i=0; i<map.size(); i++)
        if (map[i]==0) cv::line(dbg,im_points0[i],im_points1[i],CV_RGB(255,0,0));
    }

    // predict
    for (unsigned i=0; i<map.size(); i++)
    {
      if (map[i] == 1) continue;

      Eigen::Map<Eigen::Vector2f>(&im_points1[i].x) = transform.topLeftCorner<2,2>()*Eigen::Map<Eigen::Vector2f>(&im_points0[i].x) + transform.block<2,1>(0,2);
    }

    if (!dbg.empty()) { //-- dbg draw --
      for (unsigned i=0; i<map.size(); i++)
        if (map[i]==0) cv::line(dbg,im_points0[i],im_points1[i],CV_RGB(0,0,255));
    }
  }

  if (!dbg.empty()) { //-- dbg draw --
    for (unsigned i=0; i<inliers.size(); i++)
      cv::line(dbg,im_points0[inliers[i]],im_points1[inliers[i]],CV_RGB(0,255,0));
    cout<<"inliers="<<inliers.size()<<"/"<<model->keys.size()<<endl;
  }

  return double(inliers.size())/double(model->keys.size());
}

/**
 * getProjections
 * @param im_pts <model_point_index, projection>
 */
void LKHomographyTracker::getProjections(std::vector< std::pair<int,cv::Point2f> > &im_pts)
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
void LKHomographyTracker::setModel(const ObjectView::Ptr &_model) 
{  
  model=_model; 
  ObjectView &view = *model;

cout<<"view.keys.size()="<<view.keys.size()<<endl;

  im_points0.resize(view.keys.size());

  for (unsigned i=0; i<view.keys.size(); i++)
    im_points0[i] = view.keys[i].pt;

  im_points1 = im_points0;

  pts0.reserve(im_points0.size());
  pts1.reserve(im_points1.size());
}




}












