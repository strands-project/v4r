/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_ARTICULATED_OBJECT_HH
#define KP_ARTICULATED_OBJECT_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Dense>
#include "PartMotion6D.hh"
#include "v4r/KeypointTools/SmartPtr.hpp"
#include "v4r/KeypointSlam/Object.hpp"
#include "v4r/KeypointTools/convertPose.hpp"


namespace kp
{

/**
 * feature pairs to compute the pnp pose transformation
 */
class FeatureGroup
{
public:
  int part_idx;
  int view_idx;
  std::vector<cv::Point2f> im_points;
  std::vector<Eigen::Vector3f> points;
  std::vector<Eigen::Vector3f> normals;
  std::vector< int > part_feature_indices;
  std::vector< int > view_feature_indices;
  FeatureGroup() {}
  /** clear **/
  inline void clear() { 
    im_points.clear();
    points.clear();
    normals.clear();
    part_feature_indices.clear();
    view_feature_indices.clear();
  }
  /** add data **/
  inline void push_back(const cv::Point2f &im_pt, const Eigen::Vector3f &pt3, const Eigen::Vector3f &n, int _part_feature_idx, int _view_feature_idx) {
    points.push_back(pt3);
    normals.push_back(n);
    part_feature_indices.push_back(_part_feature_idx);
    view_feature_indices.push_back(_view_feature_idx);
  }
  /** resize **/
  inline void resize(int z) {
    im_points.resize(z);
    points.resize(z);
    normals.resize(z);
    part_feature_indices.resize(z);
    view_feature_indices.resize(z);
  }
  /** filter **/
  inline int filter(const std::vector<int> &valid_indices)
  {
    int z=0;
    for (unsigned i=0; i<valid_indices.size(); i++)
    {
      if (valid_indices[i]==1)
      {
        im_points[z] = im_points[i];
        points[z] = points[i];
        normals[z] = normals[i];
        part_feature_indices[z] = part_feature_indices[i];
        view_feature_indices[z] = view_feature_indices[i];
        z++;
      }
    }
    resize(z);
    return z;
  }
};



/*************************************************************************** 
 * ArticulatedObject 
 */
class ArticulatedObject : public Object, public PartMotion6D
{
public:
  std::vector< std::vector< Eigen::VectorXd > > part_parameter; // parameter for articulated scenes (objects)

  std::vector<Part::Ptr> parts;         // the first part is the object itself

  ArticulatedObject() {};

  /* clear */
  void clearArticulatedObject();

  /* add a new object view */
  ObjectView &addArticulatedView(const Eigen::Matrix4f &_pose, const cv::Mat_<unsigned char> &im=cv::Mat_<unsigned char>(), const std::vector<Eigen::VectorXd> &_part_parameter=std::vector<Eigen::VectorXd>());

  /* add projections */
  void addArticulatedProjections(ObjectView &view, const std::vector< std::pair<int,cv::Point2f> > &im_pts, const Eigen::Matrix4f &pose, const std::vector<Eigen::VectorXd> &_part_parameter=std::vector<Eigen::VectorXd>());

  /* get parameters of the articulated parts */
  void getParameters(std::vector<Eigen::VectorXd> &_params);

  /* set parameters of the articulated parts */
  void setParameters(const std::vector<Eigen::VectorXd> &_params);

  /* updatePoseRecursive */
  void updatePoseRecursive(const Eigen::Matrix4f &_pose, Part &part, std::vector<Part::Ptr> &parts);

  /* updatePoseRecursive */
  void updatePoseRecursive(const Eigen::Matrix4f &_pose=Eigen::Matrix4f::Identity());

  /* getKinematicChain */
  void getChainRecursive(const Part &part, const std::vector<Part::Ptr> &parts, int idx, std::vector< std::vector<int> > &kinematics);


  /* getKinematicChain */
  void getKinematicChain(std::vector< std::vector<int> > &kinematics);

  /* getFeatures */
  void getFeatures(int part_idx, int view_idx, FeatureGroup &features);

  /** getDescriptors */
  void getDescriptors(const FeatureGroup &features, cv::Mat &descs);

  /** addCamera **/
  int addCamera(const std::vector<Eigen::VectorXd> &_part_parameter);

  typedef SmartPtr< ::kp::ArticulatedObject> Ptr;
  typedef SmartPtr< ::kp::ArticulatedObject const> ConstPtr;
};





} //--END--

#endif

