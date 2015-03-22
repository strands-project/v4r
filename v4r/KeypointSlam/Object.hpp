/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_OBJECT_HH
#define KP_OBJECT_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Dense>
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "v4r/KeypointTools/triple.hpp"


namespace kp
{

class Object;

/***********************************************************************
 * GlobalPoint
 */
class GlobalPoint
{
public:
  int cnt;                   // view instance counter
  Eigen::Vector3d pt;        // 3d point in global coordinates
  Eigen::Vector3d n;         // normal
  GlobalPoint() : cnt(0) {}
  GlobalPoint(const Eigen::Vector3d &_pt, const Eigen::Vector3d &_n) : cnt(1), pt(_pt), n(_n) {}
};

/***********************************************************************
 * ObjectView 
 */
class ObjectView
{
public:

  typedef SmartPtr< ::kp::ObjectView> Ptr;
  typedef SmartPtr< ::kp::ObjectView const> ConstPtr;

  int idx;
  int camera_id;
  Eigen::Vector3f center;
  cv::Mat_<unsigned char> image;               // ... this camera view
  cv::Mat descs;
  std::vector<cv::KeyPoint> keys;
  std::vector<unsigned> points;
  std::vector<Eigen::Vector3f> viewrays;
  std::vector<Eigen::Vector3f> cam_points;    // 3d points in camera coordinates
  std::vector< std::vector< triple<int, cv::Point2f, Eigen::Vector3f> > > projs;
  std::vector< int > part_indices;
  std::vector< ObjectView::Ptr > related_views_;

  Object *object;

  ObjectView(Object *_o, int _camera_id=-1) : idx(-1), camera_id(_camera_id), center(Eigen::Vector3f::Zero()), object(_o) {}

  /** access a global 3d point **/
  inline GlobalPoint& getPt(unsigned idx);

  /** access a global 3d point **/
  inline const GlobalPoint& getPt(unsigned idx) const;

  /** add a global 3d point **/
  inline void addPt(const Eigen::Vector3f &pt, const Eigen::Vector3f &n=Eigen::Vector3f(std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()));

  /** add a point and increment the global counter **/
  inline void addPt(unsigned glob_idx); 

  /** decrement the global point counter **/
  inline bool decPt(unsigned idx);

  /** get global 3d points **/
  void getPoints(std::vector<Eigen::Vector3f> &pts);
  
  /** get global normals **/
  void getNormals(std::vector<Eigen::Vector3f> &normals);

  /** get camera pose **/
  inline Eigen::Matrix4f &getCamera();

  /** get camera pose **/
  inline const Eigen::Matrix4f &getCamera() const;

  /* clear */
  void clear();

  /* add a keypoint */
  void add(const cv::KeyPoint &key, float *d, int dsize, const Eigen::Vector3f &pt3, const Eigen::Vector3f &n, const Eigen::Vector3f &vr, const Eigen::Vector3f &cam_pt=Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN()), const int &part_idx=-1);

  /* add a keypoint */
  void add(const cv::KeyPoint &key, float *d, int dsize, unsigned glob_idx, const Eigen::Vector3f &vr, const Eigen::Vector3f &cam_pt=Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN()), const int &part_idx=-1);


  /* delete a complete entry */
  void del(unsigned idx);

  /* compute center */
  void computeCenter();

  /* copy to (projs are not copied!) */
  void copyTo(ObjectView &view);
};





/*************************************************************************** 
 * Object 
 */
class Object
{
public:
  std::string id;
  std::vector< Eigen::Matrix4f > cameras;                       // cameras for a static scene
  std::vector< std::vector<double> > camera_parameter;          // camera parameter [fx fy cx cy k1 k2 k3 p1 p2]

  std::vector<ObjectView::Ptr> views;

  std::vector<GlobalPoint> points;  // global point array

  cv::Mat cb_centers;
  std::vector< std::vector< std::pair<int,int> > > cb_entries;

  Object() {};
  virtual ~Object() {}

  /** check if a codebook is available **/
  inline bool haveCodebook() {return !cb_centers.empty(); }

  /** add a new global 3d point **/
  inline unsigned addPt(const Eigen::Vector3f &pt, const Eigen::Vector3f &n=Eigen::Vector3f(std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN())) {
    points.push_back(GlobalPoint(pt.cast<double>(), n.cast<double>()));
    return points.size()-1;
  }
  
  /** increment a global 3d point **/
  inline unsigned incPt(unsigned idx) {
    points[idx].cnt++;
    return idx;
  }

  /** decrement a global 3d point **/
  inline bool decPt(unsigned idx) {
    if (points[idx].cnt > 0) {
      points[idx].cnt--;
      return true;
    }
    return false;
  }

  /** access a global 3d point **/
  inline GlobalPoint &getPt(unsigned idx) { return points[idx]; }

  /** access a global 3d point **/
  inline const GlobalPoint &getPt(unsigned idx) const { return points[idx]; }

  /* clear */
  void clear() {
    cameras.clear();
    camera_parameter.clear();
    views.clear();
    points.clear();
  }

  /* add camera parameter 
     ATTENTION: Either each camera has its own parameter or all have the same
     -> cameras.size()==camera_parameter.size() or camera_parameter.size()==1 */
  void addCameraParameter(const cv::Mat_<double> &_intrinsic, const cv::Mat_<double> &_dist_coeffs) {
    camera_parameter.push_back(std::vector<double>());
    std::vector<double> &param = camera_parameter.back();
    param.clear();
    if (!_dist_coeffs.empty()) {
      param.resize(9);
      param[4] = _dist_coeffs(0,0);
      param[5] = _dist_coeffs(0,1);
      param[6] = _dist_coeffs(0,5);
      param[7] = _dist_coeffs(0,2);
      param[8] = _dist_coeffs(0,3);
    } else param.resize(4);
    param[0] = _intrinsic(0,0);
    param[1] = _intrinsic(1,1);
    param[2] = _intrinsic(0,2);
    param[3] = _intrinsic(1,2);
  }

  /* add a new object view */
  ObjectView &addObjectView(const Eigen::Matrix4f &_pose, const cv::Mat_<unsigned char> &im=cv::Mat_<unsigned char>()) {
    views.push_back( ObjectView::Ptr(new ObjectView(this, cameras.size())) );
    cameras.push_back(_pose);
    if (im.empty()) views.back()->image=cv::Mat_<unsigned char>();
    else im.copyTo(views.back()->image);
    return *views.back();
  }

  void addObjectView(kp::ObjectView::Ptr & view, const Eigen::Matrix4f &_pose) {
    view->object = this;
    view->camera_id = cameras.size();
    views.push_back( view );
    cameras.push_back(_pose);
    /*if (im.empty()) views.back()->image=cv::Mat_<unsigned char>();
    else im.copyTo(views.back()->image);
    return *views.back();*/
  }

  /* init projections */
  void initProjections(ObjectView &view) {
    if (view.keys.size()!=view.points.size()) std::runtime_error("[ObjectView::init] keys.size()!=points.size()!");
    view.projs.clear();
    view.projs.resize(view.keys.size());
    for (unsigned i=0; i<view.keys.size(); i++) {
      view.projs[i].push_back(triple<int,cv::Point2f,Eigen::Vector3f>(view.camera_id,view.keys[i].pt,view.cam_points[i]));
    }
  }

  /* add projections */
  void addProjections(ObjectView &view, const std::vector< std::pair<int,cv::Point2f> > &im_pts, const Eigen::Matrix4f &pose) {
    if (im_pts.size()>view.projs.size()) std::runtime_error("[ObjectView::add] Invalid number of points! Did you call init?");
    const Eigen::Vector3f NaNs(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN());
    for (unsigned i=0; i<im_pts.size(); i++)
      view.projs[im_pts[i].first].push_back(triple<int,cv::Point2f,Eigen::Vector3f>(cameras.size(),im_pts[i].second,NaNs));
    cameras.push_back(pose);
  }

  /* add projections */
  void addProjections(ObjectView &view, const std::vector< std::pair<int,cv::Point2f> > &im_pts, int camera_idx) {
    if (im_pts.size()>view.projs.size()) std::runtime_error("[ObjectView::add] Invalid number of points! Did you call init?");
    const Eigen::Vector3f NaNs(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN());
    for (unsigned i=0; i<im_pts.size(); i++)
      view.projs[im_pts[i].first].push_back(triple<int,cv::Point2f,Eigen::Vector3f>(camera_idx,im_pts[i].second,NaNs));
  }

  /* add projections */
  void addProjections(ObjectView &view, const std::vector< std::pair<int,cv::Point2f> > &im_pts, const std::vector<Eigen::Vector3f> &pts3, const Eigen::Matrix4f &pose) {
    if (im_pts.size()>view.projs.size() || im_pts.size()!=pts3.size()) std::runtime_error("[ObjectView::add] Invalid number of points! Did you call init?");
    const Eigen::Vector3f NaNs(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN());
    for (unsigned i=0; i<im_pts.size(); i++)
      view.projs[im_pts[i].first].push_back(triple<int,cv::Point2f,Eigen::Vector3f>(cameras.size(),im_pts[i].second,pts3[i]));
    cameras.push_back(pose);
  }

  /* add projections */
  void addProjections(ObjectView &view, const std::vector< std::pair<int,cv::Point2f> > &im_pts, const std::vector<Eigen::Vector3f> &pts3, int camera_idx) {
    if (im_pts.size()>view.projs.size() || im_pts.size()!=pts3.size()) std::runtime_error("[ObjectView::add] Invalid number of points! Did you call init?");
    const Eigen::Vector3f NaNs(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN());
    for (unsigned i=0; i<im_pts.size(); i++)
      view.projs[im_pts[i].first].push_back(triple<int,cv::Point2f,Eigen::Vector3f>(camera_idx,im_pts[i].second,pts3[i]));
  }


  typedef SmartPtr< ::kp::Object> Ptr;
  typedef SmartPtr< ::kp::Object const> ConstPtr;
};



/******************** impl ObjectView **************************/

/** access a global 3d point **/
inline GlobalPoint& ObjectView::getPt(unsigned idx) { 
  return object->getPt(points[idx]); 
}

/** access a global 3d point **/
inline const GlobalPoint& ObjectView::getPt(unsigned idx) const { 
  return object->getPt(points[idx]); 
}

/** add a new global 3d point **/
inline void ObjectView::addPt(const Eigen::Vector3f &pt, const Eigen::Vector3f &n) {
  points.push_back(object->addPt(pt,n));
}

/** add a point and increment the global counter **/
inline void ObjectView::addPt(unsigned glob_idx) {
  object->incPt(glob_idx);
  points.push_back(glob_idx);
}

/** decrement the global point counter */
inline bool ObjectView::decPt(unsigned idx)
{
  return object->decPt(points[idx]);
}

/** get camera pose **/
inline Eigen::Matrix4f &ObjectView::getCamera() 
{ 
  return object->cameras[camera_id]; 
}

/** get camera pose **/
inline const Eigen::Matrix4f &ObjectView::getCamera() const 
{ 
  return object->cameras[camera_id]; 
}


} //--END--

#endif

