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

#ifndef SURFACE_MOS_PLANES_3D_HH
#define SURFACE_MOS_PLANES_3D_HH

#ifdef DEBUG
  #include "v4r/TomGine/tgTomGineThread.h"
  #include "v4r/PCLAddOns/PCLUtils.h"
#endif

//#define USE_NOISE_MODEL

#include <iostream>
#include <vector>
#include <set>
#include <iomanip>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include "v4r/PCLAddOns/NormalsEstimationNR.hh"
#include "v4r/SurfaceUtils/SurfaceModel.hpp"
#include "v4r/SurfaceUtils/Utils.hh"

#include "SubsamplePointCloud2.hh"
#include "GreedySelection.hh"
#include "AssignPointsToPlanes.hh"


namespace surface 
{

template<typename T1,typename T2>
extern T1 Dot3(const T1 v1[3], const T2 v2[3]);

template<typename T1,typename T2, typename T3>
extern void Mul3(const T1 v[3], T2 s, T3 r[3]);

template<typename T1,typename T2, typename T3>
extern void Add3(const T1 v1[3], const T2 v2[3], T3 r[3]);

#ifdef DEBUG
  double timespec_diff(struct timespec *x, struct timespec *y);
#endif


/**
 * MoSPlanes3D
 */
class MoSPlanes3D
{
public:
  class Parameter
  {
  public:
    int pyrLevels;                              ///< number of pyramid levels
    float nbDist;                               ///< Maximum distance to neighbors for normals clustering
    float thrAngleNormalClustering;             ///< Threshold of angle for normal clustering
    float inlDist;                              ///< Maximum inlier distance for assigning points to plane
    float sigma;                                ///< Maximum error to plane constraint
    int minPoints;                              ///< Minimum number of points to create plane (normals clustern + assign points2plane)
    pclA::NormalsEstimationNR::Parameter ne;    
    GreedySelection::Parameter mos;
    
    Parameter(int _pyrLevels=3, float _nbDist=0.02, float thrAngleNC=0.2, float _inlDist=0.01, 
       float _sigma=0.005, int _minPoints=16,
       pclA::NormalsEstimationNR::Parameter _ne=pclA::NormalsEstimationNR::Parameter(), 
       GreedySelection::Parameter _mos=GreedySelection::Parameter() ) 
     : pyrLevels(_pyrLevels), nbDist(_nbDist), thrAngleNormalClustering(thrAngleNC), inlDist(_inlDist), 
       sigma(_sigma), minPoints(_minPoints),
       ne(_ne), mos(_mos) {}
  };

private:
  Parameter param;
  int width, height;
  float cosThrAngleNC;
  float invSqrSigma;

  bool line_check;    // do line check
  int lc_neighbors;   // threshold for line check neighbors (3-8)
  
  std::vector<unsigned char> mask;
  std::vector<int> queue;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

  std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > clouds;
  std::vector< pcl::PointCloud<pcl::Normal>::Ptr > normals;
  boost::shared_ptr< std::vector<SurfaceModel::Ptr> > tmpPlanes, planes;

  SubsamplePointCloud2 resize;
  pclA::NormalsEstimationNR normalsEstimation;
  GreedySelection modsel;
  AssignPointsToPlanes ptsToPlane;

  void Init(pcl::PointCloud<pcl::PointXYZRGB> &cloud);
  void ClusterNormals(pcl::PointCloud<pcl::PointXYZRGB> &cloud, 
        pcl::PointCloud<pcl::Normal> &normals, 
        std::vector<SurfaceModel::Ptr> &planes, int level=0); 
  void ClusterNormals(unsigned idx, pcl::PointCloud<pcl::PointXYZRGB> &cloud, 
        pcl::PointCloud<pcl::Normal> &normals, std::vector<int> &pts, pcl::Normal &normal);
  void ComputeLSPlanes(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, 
        std::vector<SurfaceModel::Ptr> &planes);
  void ComputePointProbs(pcl::PointCloud<pcl::PointXYZRGB> &cloud, 
        std::vector<SurfaceModel::Ptr> &planes);
  void PseudoUpsample(pcl::PointCloud<pcl::PointXYZRGB> &cloud0, 
        pcl::PointCloud<pcl::PointXYZRGB> &cloud1, 
        std::vector<SurfaceModel::Ptr> &planes, float nbDist);
  void CCFilter(std::vector<SurfaceModel::Ptr> &planes);
  void LineCheck();

  inline int GetIdx(short x, short y);
  inline short X(int idx);
  inline short Y(int idx);
  inline int GetIdx(short x, short y, short width);
  inline short X(int idx, short width);
  inline short Y(int idx, short width);
  inline bool IsNaN(const pcl::PointXYZRGB &pt);
  inline bool Contains(std::vector<int> &indices, int idx);
  inline float SqrDistance(const pcl::PointXYZRGB &pt1, const pcl::PointXYZRGB &pt2);
  template<typename T1,typename T2, typename T3>
  inline void ProjectPoint2Image(T1 p[3], T2 C[9], T3 i[2]);
  inline float SqrDistanceZ(const pcl::PointXYZRGB &pt1, const pcl::PointXYZRGB &pt2);

#ifdef DEBUG
  void DrawNormals(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointCloud<pcl::Normal>::Ptr &normals);
  void DrawPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);
  void DrawPlanePoints(pcl::PointCloud<pcl::PointXYZRGB> &cloud, std::vector<SurfaceModel::Ptr> &planes);
  void DrawSurfaces(std::vector<SurfaceModel::Ptr> &surfaces);
#endif
  
  cv::Point ComputeMean(std::vector<int> &indices);

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

#ifdef DEBUG
   cv::Mat dbg;
   cv::Ptr<TomGine::tgTomGineThread> dbgWin;
#endif

  MoSPlanes3D(Parameter p=Parameter());
  ~MoSPlanes3D();

  /** Set parameters for plane estimation **/
  void setParameter(Parameter p);

  /** Set line-check and thresholds **/
  void setLineCheck(bool check, int neighbors);
 
  /** Set input cloud **/
  void setInputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud);

  /** Compute planes by surface normal grouping **/
  void compute();

  /** Returns the estimated surface patches **/
  void getSurfaceModels(std::vector<SurfaceModel::Ptr> &_planes);

  /** Returs normals of the full image **/
  void getNormals(pcl::PointCloud<pcl::Normal>::Ptr &_normals) {_normals = normals[0];}
  
  /** Returns the error values of the estimated points **/
  void getError(std::vector< std::vector<double> > &_error);
};




/*********************** INLINE METHODES **************************/
inline float MoSPlanes3D::SqrDistanceZ(const pcl::PointXYZRGB &pt1, const pcl::PointXYZRGB &pt2)
{
  return pow((pt1.z-pt2.z), 2);
}

template<typename T1,typename T2, typename T3>
inline void MoSPlanes3D::ProjectPoint2Image(T1 p[3], T2 C[9], T3 i[2])
{
  i[0] = C[0] * p[0]/p[2] + C[2];
  i[1] = C[4] * p[1]/p[2] + C[5];
}

inline int MoSPlanes3D::GetIdx(short x, short y)
{
  return y*width+x; 
}

inline short MoSPlanes3D::X(int idx)
{
  return idx%width;
}

inline short MoSPlanes3D::Y(int idx)
{
  return idx/width;
}

inline int MoSPlanes3D::GetIdx(short x, short y, short width)
{
  return y*width+x; 
}

inline short MoSPlanes3D::X(int idx, short width)
{
  return idx%width;
}

inline short MoSPlanes3D::Y(int idx, short width)
{
  return idx/width;
}

inline bool MoSPlanes3D::IsNaN(const pcl::PointXYZRGB &pt)
{
  if (pt.x!=pt.x)
    return true;
  return false;
}

inline bool MoSPlanes3D::Contains(std::vector<int> &indices, int idx)
{
  for (unsigned i=0; i<indices.size(); i++)
    if (indices[i] == idx)
      return true;
  return false;
}

inline float MoSPlanes3D::SqrDistance(const pcl::PointXYZRGB &pt1, const pcl::PointXYZRGB &pt2)
{
  if (pt1.x==pt1.x && pt2.x==pt2.x)
    return pow((pt1.x-pt2.x),2) + pow((pt1.y-pt2.y),2) + pow((pt1.z-pt2.z),2);

  return FLT_MAX;
}

template <class T>
inline std::string toString (const T& t, unsigned precision=2)
{
  std::stringstream ss;
  ss << std::setprecision(precision) << t;
  return ss.str();
}

}

#endif

