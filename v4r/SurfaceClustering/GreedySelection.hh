/**
 *  Copyright (C) 2012  
 *    Andreas Richtsfeld, Johann Prankl, Thomas Mörwald
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1170 Vienn, Austria
 *    ari(at)acin.tuwien.ac.at
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
 *  along with this program.  If not, see http://www.gnu.org/licenses/
 */


#ifndef SURFACE_GREEDY_SELECTION_HH
#define SURFACE_GREEDY_SELECTION_HH

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>

#include "v4r/SurfaceUtils/SurfaceModel.hpp"
#include "PPlane.h"


namespace surface 
{
  
static const double feps = 1e-6;

class PointSurfaceProb
{
public:
  int idxPoint;  // it's the index of the point in a surface container
  int idxSurface;
  float prob;

  PointSurfaceProb(){}
  PointSurfaceProb(int pt, int sf, float _prob) 
   : idxPoint(pt), idxSurface(sf), prob(_prob) {}
};


class GreedySelection
{
public:
  class Parameter
  {
  public:
    float kappa1;       ///< model cost factor    // 10 (for normalized e.g. 0.1)
    float kappa2;       ///< Error cost factor
    float sigma;        ///< Error reference value

    Parameter(float kap1=10, float kap2=.9, float _sigma=.6) 
     : kappa1(kap1), kappa2(kap2), sigma(_sigma) {}
  };

private:
  int width, height;
  float invSqrSigma;
  std::vector<std::vector<PointSurfaceProb> > ptsSurfaceProbs;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

  std::vector<int> idMap;
  std::vector<unsigned> mask;
  unsigned mcnt;
  std::vector<SurfaceModel::Ptr> tmpPlanes;

  std::vector<int> indices0;
  std::vector<float> probs;
  std::vector<SurfaceModel::Ptr> overlap;  

  void ComputePointProbs(pcl::PointCloud<pcl::PointXYZRGB> &cloud, 
                         std::vector<int> &indices, 
                         std::vector<float> &coeffs, 
                         std::vector<float> &probs);
  void IndexToPointSurfaceProbs(SurfaceModel &surf, int idx);
  float ComputeSavings(std::vector<float> &probs);
  void Init(std::vector<SurfaceModel::Ptr> &planes0);
  bool SelectPlane(std::vector<SurfaceModel::Ptr> &planes0, SurfaceModel &plane);



public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Parameter param;
  cv::Mat dbg;

  GreedySelection(Parameter p=Parameter());
  ~GreedySelection();

  /** Set input cloud **/
  void setInputCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud);
  
  /** Set parameter for greedy selection **/
  void setParameter(Parameter &p);
  
  /** Compute **/
  void compute(std::vector<SurfaceModel::Ptr> &planes0, std::vector<SurfaceModel::Ptr> &planes1);
};



/*********************** INLINE METHODES **************************/




}

#endif

