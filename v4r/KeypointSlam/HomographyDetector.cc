/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#include "HomographyDetector.hh"

namespace kp
{

using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
HomographyDetector::HomographyDetector(const Parameter &p, 
    const kp::FeatureDetector::Ptr &_detector,
    const kp::FeatureDetector::Ptr &_descEstimator)
 : param(p), detector(_detector), descEstimator(_descEstimator)
{ 
  imr.reset(new ImageTransformRANSAC(param.imr_param));

  matcher = new cv::BFMatcher(cv::NORM_L2);
  if (detector.get()==0) detector = descEstimator;
}

HomographyDetector::~HomographyDetector()
{
}






/******************************* PUBLIC ***************************************/

/**
 * detect
 */
double HomographyDetector::detect(const cv::Mat &image, Eigen::Matrix3f &transform)
{
  lt.clear();
  inliers.clear();
  inls.clear();

  if (model.get()==0)
    throw std::runtime_error("[HomographyDetector::detect] No model available!");

  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;

  if (model->keys.size()<4) return 0.;

  //{ kp::ScopeTime t("detect keypoints");
  detector->detect(im_gray, keys);
  descEstimator->extract(im_gray, keys, descs);
  //}

  //matcher->knnMatch( descs, model->descs, matches, 2 );
  matcher->knnMatch( descs, matches, 2 );


  if (matches.size()<4) return 0.;

  pts0.clear();
  pts1.clear();

  // convert data
  for (unsigned z=0; z<matches.size(); z++)
  {
    if (matches[z].size()>1)
    {
      cv::DMatch &ma0 = matches[z][0];
      if (ma0.distance/matches[z][1].distance < param.nnr)
      {
        pts0.push_back(Eigen::Map<Eigen::Vector2f>(&model->keys[ma0.trainIdx].pt.x));
        pts1.push_back(Eigen::Map<Eigen::Vector2f>(&keys[ma0.queryIdx].pt.x));
        lt.push_back(ma0.trainIdx);
      }
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
  }

  if (!dbg.empty()) { //-- dbg draw --
    for (unsigned i=0; i<inliers.size(); i++) {
      const Eigen::Vector2f &pt = pts1[inls[i]];
      cv::line(dbg,model->keys[inliers[i]].pt,cv::Point2f(pt[0],pt[1]),CV_RGB(0,255,0));
    }
    cout<<"inliers="<<inliers.size()<<"/"<<model->keys.size()<<endl;
  }

  return double(inliers.size())/double(model->keys.size());
}

/**
 * getProjections
 * @param im_pts <model_point_index, projection>
 */
void HomographyDetector::getProjections(std::vector< std::pair<int,cv::Point2f> > &im_pts)
{
  im_pts.clear();
  if (model.get()==0)
    return;

  for (unsigned i=0; i<inliers.size(); i++) {
    const Eigen::Vector2f &pt = pts1[inls[i]];
    im_pts.push_back(make_pair(inliers[i],cv::Point2f(pt[0],pt[1])));
  }
}

/**
 * setModel
 */
void HomographyDetector::setModel(const ObjectView::Ptr &_model) 
{  
  model=_model; 
  ObjectView &view = *model;

  pts0.reserve(view.keys.size());
  pts1.reserve(view.keys.size());

  matcher->clear();
  matcher->add(std::vector<cv::Mat>(1,model->descs));
  matcher->train();
}




}












