/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_CLUSTER_NORMALS_TO_PLANES_HH
#define KP_CLUSTER_NORMALS_TO_PLANES_HH

#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "v4r/common/keypoint/impl/DataMatrix2D.hpp"
#include "v4r/common/keypoint/impl/SmartPtr.hpp"
#include "v4r/common/keypoint/PlaneEstimationRANSAC.hh"

namespace kp
{



/**
 * ClusterNormalsToPlanes
 */
class ClusterNormalsToPlanes
{
public:

  /**
   * @brief The Parameter class
   */
  class Parameter
  {
  public:
    float thrAngle;             // Threshold of angle for normal clustering 
    float inlDist;              // Maximum inlier distance
    unsigned minPoints;              // Minimum number of points for a plane
    bool least_squares_refinement;
    bool smooth_clustering;
    float thrAngleSmooth;             // Threshold of angle for normal clustering
    float inlDistSmooth;              // Maximum inlier distance
    unsigned minPointsSmooth;
    
    Parameter(float thrAngleNC=30, float _inlDist=0.01, unsigned _minPoints=9, bool _least_squares_refinement=true, bool _smooth_clustering=false,
              float _thrAngleSmooth=30, float _inlDistSmooth=0.02, unsigned _minPointsSmooth=3)
    : thrAngle(thrAngleNC), inlDist(_inlDist), minPoints(_minPoints), least_squares_refinement(_least_squares_refinement), smooth_clustering(_smooth_clustering),
      thrAngleSmooth(_thrAngleSmooth), inlDistSmooth(_inlDistSmooth), minPointsSmooth(_minPointsSmooth){}
  };

  /**
   * @brief The Plane class
   */
  class Plane
  {
  public:
    bool is_plane;
    Eigen::Vector3f pt;
    Eigen::Vector3f normal;
    std::vector<int> indices;
    /** clear **/
    void clear() {
      pt.setZero();
      normal.setZero();
      indices.clear();
    }
    /** init **/
    inline void init(const Eigen::Vector3f &_pt, const Eigen::Vector3f &_n, int idx) {
      pt = _pt;
      normal = _n;
      indices.resize(1);
      indices[0] = idx;
    }
    /** add **/
    inline void add(const Eigen::Vector3f &_pt, const Eigen::Vector3f &_n, int idx) {
      pt *= float(indices.size());
      normal *= float(indices.size());
      pt += _pt;
      normal += _n;
      indices.push_back(idx);
      pt /= float(indices.size());
      normal /= float(indices.size());
    }
    /** size **/
    inline unsigned size() {return indices.size(); }
    /** Plane **/
    Plane(bool _is_plane=false) : is_plane(_is_plane) {}
    ~Plane() {}
    typedef SmartPtr< ::kp::ClusterNormalsToPlanes::Plane> Ptr;
    typedef SmartPtr< ::kp::ClusterNormalsToPlanes::Plane const> ConstPtr;
  };

private:
  Parameter param;
  float cos_rad_thr_angle, cos_rad_thr_angle_smooth;

  std::vector<bool> mask;
  std::vector<int> queue;

  PlaneEstimationRANSAC pest;
  
  // cluster normals
  void doClustering(const kp::DataMatrix2D<Eigen::Vector3f> &cloud, const kp::DataMatrix2D<Eigen::Vector3f> &normals, std::vector<Plane::Ptr> &planes);
  // cluster normals from point
  void clusterNormals(const kp::DataMatrix2D<Eigen::Vector3f> &cloud, const kp::DataMatrix2D<Eigen::Vector3f> &normals, int idx, Plane &plane);
  // do a smooth clustering
  void smoothClustering(const kp::DataMatrix2D<Eigen::Vector3f> &cloud, const kp::DataMatrix2D<Eigen::Vector3f> &normals, int idx, Plane &plane);
  // least square plane
  void computeLeastSquarePlanes(const kp::DataMatrix2D<Eigen::Vector3f> &cloud, std::vector<Plane::Ptr> &planes);
  // adds normals to each point of segmented patches

  inline bool isnan(const Eigen::Vector3f &pt);

public:

  ClusterNormalsToPlanes(const Parameter &_p=Parameter());
  ~ClusterNormalsToPlanes();

  /** Compute planes by surface normal grouping **/
  void compute(const kp::DataMatrix2D<Eigen::Vector3f> &_cloud, const kp::DataMatrix2D<Eigen::Vector3f> &_normals, std::vector<Plane::Ptr> &planes);

  /** Compute a plane starting from a seed point **/
  void compute(const kp::DataMatrix2D<Eigen::Vector3f> &cloud, const kp::DataMatrix2D<Eigen::Vector3f> &normals, int x, int y, Plane &plane);


  typedef SmartPtr< ::kp::ClusterNormalsToPlanes> Ptr;
  typedef SmartPtr< ::kp::ClusterNormalsToPlanes const> ConstPtr;
};


/**
 * @brief ClusterNormalsToPlanes::isnan
 * @param pt
 * @return
 */
inline bool ClusterNormalsToPlanes::isnan(const Eigen::Vector3f &pt)
{
  if (std::isnan(pt[0]) || std::isnan(pt[1]) || std::isnan(pt[2]))
    return true;
  return false;
}

}

#endif

