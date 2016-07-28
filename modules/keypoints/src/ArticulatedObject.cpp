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

#include <v4r/keypoints/ArticulatedObject.h>

namespace v4r
{

/* clear */
void ArticulatedObject::clearArticulatedObject() 
{
  clear();
  part_parameter.clear();
  parts.clear();
}

/* add a new object view */
ObjectView &ArticulatedObject::addArticulatedView(const Eigen::Matrix4f &_pose, const cv::Mat_<unsigned char> &im, const std::vector<Eigen::VectorXd> &_part_parameter) 
{
  part_parameter.push_back(_part_parameter);
  return addObjectView(_pose, im);
}

/* add projections */
void ArticulatedObject::addArticulatedProjections(ObjectView &view, const std::vector< std::pair<int,cv::Point2f> > &im_pts, const Eigen::Matrix4f &pose, const std::vector<Eigen::VectorXd> &_part_parameter) 
{
  addProjections(view, im_pts, pose);
  part_parameter.push_back(_part_parameter);
}

/* get parameters of the articulated parts */
void ArticulatedObject::getParameters(std::vector<Eigen::VectorXd> &_params) 
{
  _params.resize(parts.size());

  for (unsigned i=0; i<parts.size(); i++)
    _params[i] = parts[i]->getParameter();
}

/* set parameters of the articulated parts */
void ArticulatedObject::setParameters(const std::vector<Eigen::VectorXd> &_params) 
{ 
  if (parts.size() != _params.size())
    throw std::runtime_error("[ArticulatedObjectModel::setParameters] Invalid number of parameters!");

  for (unsigned i=0; i<parts.size(); i++)
    parts[i]->setParameter(_params[i]);
}

/* updatePoseRecursive */
void ArticulatedObject::updatePoseRecursive(const Eigen::Matrix4f &_pose, Part &part, std::vector<Part::Ptr> &_parts)
{
  part.updatePose(_pose);
  for (unsigned i=0; i<part.subparts.size(); i++) {
    updatePoseRecursive(part.pose, *_parts[part.subparts[i]], _parts);
  } 
} 

/* updatePoseRecursive */
void ArticulatedObject::updatePoseRecursive(const Eigen::Matrix4f &_pose) 
{
  updatePose(_pose);
  for (unsigned i=0; i<subparts.size(); i++) {
    updatePoseRecursive(pose, *parts[subparts[i]], parts);
  }
}

/* getKinematicChain */
void ArticulatedObject::getChainRecursive(const Part &part, const std::vector<Part::Ptr> &parts, int _idx, std::vector< std::vector<int> > &kinematics)
{
  kinematics[part.idx] = kinematics[_idx];
  kinematics[part.idx].push_back(part.idx);

  for (unsigned i=0; i<part.subparts.size(); i++)
    getChainRecursive(*parts[part.subparts[i]], parts, part.idx, kinematics);
}

/* getKinematicChain */
void ArticulatedObject::getKinematicChain(std::vector< std::vector<int> > &kinematics) 
{
  kinematics.clear();
  kinematics.resize(parts.size());
  kinematics[this->idx].push_back(this->idx);

  for (unsigned i=0; i<subparts.size(); i++)
    getChainRecursive(*parts[subparts[i]], parts, this->idx, kinematics);
}

/* getFeatures */
void ArticulatedObject::getFeatures(int part_idx, int view_idx, FeatureGroup &features)
{
  features.part_idx = part_idx;
  features.view_idx = view_idx;

  features.clear();
  ObjectView &view = *views[view_idx];
  Part &part = *parts[part_idx];

  for (unsigned i=0; i<part.features.size(); i++)
  {
    std::pair<int,int> &f = part.features[i];
    if (f.first == view_idx) 
      features.push_back(view.keys[f.second].pt,view.getPt(f.second).pt.cast<float>(), view.getPt(f.second).n.cast<float>(), i, f.second);      
  }
}

/**
 * getDescriptors
 */
void ArticulatedObject::getDescriptors(const FeatureGroup &features, cv::Mat &descs)
{
  descs = cv::Mat();
  if (features.points.size()==0)
    return;

  cv::Mat &dst = views[features.view_idx]->descs;

  descs = cv::Mat_<float>(features.view_feature_indices.size(), dst.cols);
  
  for (unsigned i=0; i<features.view_feature_indices.size(); i++)
  {
    std::memcpy(descs.ptr<float>(i,0), dst.ptr<float>(features.view_feature_indices[i],0), dst.cols*sizeof(float));
  }
}

/**
 * addCamera
 */
int ArticulatedObject::addCamera(const std::vector<Eigen::VectorXd> &_part_parameter)
{
  if (_part_parameter.size()==0)
    return -1;

  Eigen::Matrix4f _pose;

  convertPose(_part_parameter[0],_pose);
  part_parameter.push_back(_part_parameter);
  cameras.push_back(_pose);

  return part_parameter.size()-1;  
}


} //--END--
