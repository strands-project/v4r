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

#ifndef SURFACE_SUBSAMPLE_POINT_CLOUD2_HH
#define SURFACE_SUBSAMPLE_POINT_CLOUD2_HH

#include <iostream>
#include <stdexcept>
#include <omp.h>
#include <math.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/sac_model_plane.h>


namespace surface
{

/**
 * Subsample point cloud for pyramid calculations
 */
class SubsamplePointCloud2
{
public:
  class Parameter
  {
    public:
      float dist;                       ///< TODO Distance ???

      Parameter(float _dist=0.02)
       : dist(_dist) {}
  };

private:
  Parameter param;
  float sqrDist;

  int width, height;
  static float NaN;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr resCloud;

  void SubsampleMean(pcl::PointCloud<pcl::PointXYZRGB> &in, pcl::PointCloud<pcl::PointXYZRGB> &out);

  inline int GetIdx(short x, short y);
  inline bool IsNaN(const pcl::PointXYZRGB &pt);
  inline float Sqr(const float x);
  inline float SqrDistance(const pcl::PointXYZRGB &pt1, const pcl::PointXYZRGB &pt2);


public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SubsamplePointCloud2(Parameter p=Parameter());
  ~SubsamplePointCloud2();

  void setParameter(Parameter p);
  void setInputCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud);
  void compute();
  void getCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud);
};




/*********************** INLINE METHODES **************************/
inline int SubsamplePointCloud2::GetIdx(short x, short y)
{
  return y*width+x;
}

inline bool SubsamplePointCloud2::IsNaN(const pcl::PointXYZRGB &pt)
{
  if (pt.x!=pt.x)
    return true;
  return false;
}

inline float SubsamplePointCloud2::Sqr(const float x)
{       
  return x*x;
}       
        
inline float SubsamplePointCloud2::SqrDistance(const pcl::PointXYZRGB &pt1, const pcl::PointXYZRGB &pt2)
{       
  if (pt1.x==pt1.x && pt2.x==pt2.x)
    return Sqr(pt1.x-pt2.x)+Sqr(pt1.y-pt2.y)+Sqr(pt1.z-pt2.z);
        
  return FLT_MAX;
}       

}

#endif

