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

#ifndef KP_ARTICULATED_OBJECT_HH
#define KP_ARTICULATED_OBJECT_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Dense>
#include <v4r/keypoints/PartMotion6D.h>
#include <v4r/common/impl/SmartPtr.hpp>
#include <v4r/keypoints/impl/Object.hpp>
#include <v4r/common/convertPose.h>


namespace v4r
{

/**
 * feature pairs to compute the pnp pose transformation
 */
class V4R_EXPORTS FeatureGroup
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
class V4R_EXPORTS ArticulatedObject : public Object, public PartMotion6D
{
public:
  std::string version;
  std::vector< std::vector< Eigen::VectorXd > > part_parameter; // parameter for articulated scenes (objects)

  std::vector<Part::Ptr> parts;         // the first part is the object itself

  ArticulatedObject() : version(std::string("1.0")) {};

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

  typedef SmartPtr< ::v4r::ArticulatedObject> Ptr;
  typedef SmartPtr< ::v4r::ArticulatedObject const> ConstPtr;
};





} //--END--

#endif

