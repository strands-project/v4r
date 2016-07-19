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
#include <v4r/common/plane_model.h>
#include <v4r/core/macros.h>
#include <pcl/common/angles.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <flann/flann.h>

namespace v4r
{

/**
 * ClusterNormalsToPlanes
 */
template< typename PointT >
class V4R_EXPORTS ClusterNormalsToPlanesPCL
{
public:

  /**
   * @brief The Parameter class
   */
  class Parameter
  {
  public:
    double thrAngle;             /// @brief Threshold of angle for normal clustering
    double inlDist;              /// @brief Maximum inlier distance
    unsigned minPoints;              /// @brief Minimum number of points for a plane
    bool least_squares_refinement;
    bool smooth_clustering;
    double thrAngleSmooth;             /// @brief Threshold of angle for normal clustering
    double inlDistSmooth;              /// @brief Maximum inlier distance
    unsigned minPointsSmooth;
    int K_; // k in nearest neighor search when doing smooth clustering in unorganized point clouds
    int normal_computation_method_; /// @brief defines the method used for normal computation (only used when point cloud is downsampled / unorganized)
    
    Parameter(double thrAngleNC=30, double _inlDist=0.01, unsigned _minPoints=9, bool _least_squares_refinement=true, bool _smooth_clustering=false,
              double _thrAngleSmooth=30, double _inlDistSmooth=0.02, unsigned _minPointsSmooth=3, int K=5, int normal_computation_method = 2)
    : thrAngle(thrAngleNC), inlDist(_inlDist), minPoints(_minPoints), least_squares_refinement(_least_squares_refinement), smooth_clustering(_smooth_clustering),
      thrAngleSmooth(_thrAngleSmooth), inlDistSmooth(_inlDistSmooth), minPointsSmooth(_minPointsSmooth), K_(K), normal_computation_method_ (normal_computation_method) {}
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

    void clear() {
      pt.setZero();
      normal.setZero();
      indices.clear();
    }

    inline void init(const Eigen::Vector3f &_pt, const Eigen::Vector3f &_n, size_t idx) {
      pt = _pt;
      normal = _n;
      indices.resize(1);
      indices[0] = (int)idx;
    }

    inline void add(const Eigen::Vector3f &_pt, const Eigen::Vector3f &_n, size_t idx) {    // update average normal and point
      double w = indices.size() / double(indices.size() + 1);
      pt = pt  * w + _pt / double(indices.size() + 1); // this should be numerically more stable than the previous version (but more division / computationally expensive)
      normal =  normal * w + _n/double(indices.size() + 1);
      indices.push_back((int)idx);
    }

    inline unsigned size() {return indices.size(); }

    Plane(bool _is_plane=false) : is_plane(_is_plane) {}
    ~Plane() {}
    typedef boost::shared_ptr< ::v4r::ClusterNormalsToPlanesPCL<PointT>::Plane> Ptr;
    typedef boost::shared_ptr< ::v4r::ClusterNormalsToPlanesPCL<PointT>::Plane const> ConstPtr;
  };

private:
  typedef flann::L1<float> DistT;

  Parameter param;
  float cos_rad_thr_angle, cos_rad_thr_angle_smooth;

  std::vector<bool> mask_;
  std::vector<int> queue_;

  boost::shared_ptr< flann::Index<DistT> > flann_index; // for unorganized point clouds;

  
  // cluster normals
  void doClustering(const typename pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<pcl::Normal> &normals, std::vector<typename ClusterNormalsToPlanesPCL<PointT>::Plane::Ptr> &planes);
  // cluster normals from point
  void clusterNormals(const typename pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<pcl::Normal> &normals, size_t idx, Plane &plane);
  // cluster normals from point for an unorganized pointcloud
  void clusterNormalsUnorganized(const typename pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<pcl::Normal> &normals, size_t idx, Plane &plane);
  // do a smooth clustering
  void smoothClustering(const typename pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<pcl::Normal> &normals, size_t idx, Plane &plane);
  // adds normals to each point of segmented patches

public:

  ClusterNormalsToPlanesPCL(const Parameter &_p=Parameter())
      : param(_p)
  {
      cos_rad_thr_angle = cos( pcl::deg2rad(param.thrAngle) );
      cos_rad_thr_angle_smooth = cos( pcl::deg2rad(param.thrAngleSmooth) );
  }

  ~ClusterNormalsToPlanesPCL()
  {}

  /** Compute planes by surface normal grouping **/
  void compute(const typename pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<pcl::Normal> &normals, std::vector<PlaneModel<PointT> > &_planes);

  /** Compute a plane starting from a seed point **/
  void compute(const typename pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<pcl::Normal> &normals, int x, int y, PlaneModel<PointT> &pm);


  typedef boost::shared_ptr< ::v4r::ClusterNormalsToPlanesPCL<PointT> > Ptr;
  typedef boost::shared_ptr< ::v4r::ClusterNormalsToPlanesPCL<PointT> const> ConstPtr;
};
}

#endif

