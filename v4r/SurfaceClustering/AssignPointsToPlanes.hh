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

#ifndef SURFACE_ASSING_POINTS_TO_PLANE_HH
#define SURFACE_ASSING_POINTS_TO_PLANE_HH

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include "v4r/SurfaceUtils/SurfaceModel.hpp"
#include "PPlane.h"

namespace surface 
{

template<typename T1,typename T2>
extern T1 Dot3(const T1 v1[3], const T2 v2[3]);
  
class PointSurface
{
public:
  int idxPoint;  // it's the index of the point in a surface container
  int idxSurface;

  PointSurface(){}
  PointSurface(int pt, int sf) 
   : idxPoint(pt), idxSurface(sf) {}
};


class AssignPointsToPlanes
{
public:
  class Parameter
  {
  public:
    int minPoints;
    float inlDist;

    Parameter(int _minPoints=10, float _inlDist=0.06) 
     : minPoints(_minPoints), inlDist(_inlDist) {}
  };

private:
  int width, height;
  std::vector<std::vector<int> > ptsSurface;
  std::vector<unsigned char> umask;
  std::vector<unsigned> mask;
  unsigned mcnt;
  

  std::vector<int> queue;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  pcl::PointCloud<pcl::Normal>::Ptr normals;

  void IndexToPointSurface(pcl::PointCloud<pcl::PointXYZRGB> &cloud, 
        SurfaceModel &surf, int idx);
  void FilterPointsNormalDist(pcl::PointCloud<pcl::Normal> &normals, 
        std::vector<SurfaceModel::Ptr > &planes); 
  void AccumulateNeighbours(pcl::PointCloud<pcl::PointXYZRGB> &cloud, 
        std::vector<SurfaceModel::Ptr> &planes);
  void ClusterNeighbours(pcl::PointCloud<pcl::PointXYZRGB> &cloud,
        std::vector<float> &coeffs, std::vector<int> &pts);
  void ComputeLSPlanes(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, 
        std::vector<SurfaceModel::Ptr> &planes);

  inline int GetIdx(short x, short y);
  inline short X(int idx);
  inline short Y(int idx);

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Parameter param;
  cv::Mat dbg;

  AssignPointsToPlanes(Parameter p=Parameter());
  ~AssignPointsToPlanes();
	
  /** Set parameter set **/
  void setParameter(Parameter p);

  /** Set pcl input cloud **/
  void setInputCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud);

  /** Set input normals corresponding to point cloud **/
  void setInputNormals(const pcl::PointCloud<pcl::Normal>::Ptr &_normals);

  /** Compute results **/
  void compute(std::vector<SurfaceModel::Ptr> &planes);
};



/*********************** INLINE METHODES **************************/

inline int AssignPointsToPlanes::GetIdx(short x, short y)
{
  return y*width+x;
}

inline short AssignPointsToPlanes::X(int idx)
{
  return idx%width;
}

inline short AssignPointsToPlanes::Y(int idx)
{
  return idx/width;
}



}

#endif

