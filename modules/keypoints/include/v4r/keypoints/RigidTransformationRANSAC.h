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

#ifndef KP_RIGID_TRANSFORMATION_RANSAC_HH
#define KP_RIGID_TRANSFORMATION_RANSAC_HH

#include <vector>
#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include <v4r/core/macros.h>
#include <v4r/common/impl/SmartPtr.hpp>


namespace v4r
{

/**
 * RigidTransformationRANSAC
 */
class V4R_EXPORTS RigidTransformationRANSAC
{
public:
  class V4R_EXPORTS Parameter
  {
  public:
    double inl_dist;
    double eta_ransac;               // eta for pose ransac
    unsigned max_rand_trials;         // max. number of trials for pose ransac

    Parameter(double _inl_dist=0.01, double _eta_ransac=0.01, unsigned _max_rand_trials=10000)
     : inl_dist(_inl_dist), eta_ransac(_eta_ransac), max_rand_trials(_max_rand_trials) {}
  };

private:
  Eigen::Matrix4f invPose;

  void DemeanPoints (
        const std::vector<Eigen::Vector3f > &inPts, 
        const std::vector<int> &indices,
        const Eigen::Vector3f &centroid,
        Eigen::MatrixXf &outPts);

  void ComputeCentroid(
        const std::vector<Eigen::Vector3f > &pts, 
        const std::vector<int> &indices,
        Eigen::Vector3f &centroid);

  void Ransac(
        const std::vector<Eigen::Vector3f > &srcPts,
        const std::vector<Eigen::Vector3f > &tgtPts,
        Eigen::Matrix4f &transform,
        std::vector<int> &inliers);

  void GetDistances(
        const std::vector<Eigen::Vector3f > &srcPts,
        const std::vector<Eigen::Vector3f > &tgtPts,
        const Eigen::Matrix4f &transform,
        std::vector<float> &dists);

  void GetInliers(std::vector<float> &dists, std::vector<int> &inliers);
  unsigned CountInliers(std::vector<float> &dists);
  void GetRandIdx(int size, int num, std::vector<int> &idx);


  inline bool Contains(const std::vector<int> &idx, int num);
  inline void InvPose(const Eigen::Matrix4f &pose, Eigen::Matrix4f &invPose);



public:
  Parameter param;

  RigidTransformationRANSAC(Parameter p=Parameter());
  ~RigidTransformationRANSAC();

  void estimateRigidTransformationSVD(
        const std::vector<Eigen::Vector3f > &srcPts,
        const std::vector<int> &srcIndices,
        const std::vector<Eigen::Vector3f > &tgtPts,
        const std::vector<int> &tgtIndices,
        Eigen::Matrix4f &transform);

  void compute(
        const std::vector<Eigen::Vector3f > &srcPts,
        const std::vector<Eigen::Vector3f > &tgtPts,
        Eigen::Matrix4f &transform,
        std::vector<int> &inliers);

  typedef SmartPtr< ::v4r::RigidTransformationRANSAC> Ptr;
  typedef SmartPtr< ::v4r::RigidTransformationRANSAC const> ConstPtr;
};




/*********************** INLINE METHODES **************************/
inline bool RigidTransformationRANSAC::Contains(const std::vector<int> &idx, int num)
{
  for (unsigned i=0; i<idx.size(); i++)
    if (idx[i]==num)
      return true;
  return false;
}

inline void RigidTransformationRANSAC::InvPose(const Eigen::Matrix4f &pose, Eigen::Matrix4f &invPose)
{ 
  invPose.setIdentity();
  invPose.block<3, 3> (0, 0) = pose.block<3, 3> (0, 0).transpose();
  invPose.block<3, 1> (0, 3) = -1*(invPose.block<3, 3> (0, 0)*pose.block<3, 1> (0, 3));
}


}

#endif

