#ifndef FAAT_PCL_COMMON_DATA_STR
#define FAAT_PCL_COMMON_DATA_STR

#include <pcl/common/common.h>
#include <pcl/io/io.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/PolygonMesh.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/voxel_grid.h>
#include <v4r/core/macros.h>

namespace v4r
{
  template<typename PointT>
  struct V4R_EXPORTS PlaneModel
  {
    pcl::ModelCoefficients coefficients_;
//    typename pcl::PointCloud<PointT>::Ptr plane_cloud_;
    pcl::PolygonMeshPtr convex_hull_;
    typename pcl::PointCloud<PointT>::Ptr cloud_;
    pcl::PointIndices inliers_;

    typename pcl::PointCloud<PointT>::Ptr
    projectPlaneCloud(float resolution=0.005f) const
    {
      typename pcl::PointCloud<PointT>::Ptr plane_cloud (new pcl::PointCloud<PointT>);
      Eigen::Vector4f model_coefficients;
      model_coefficients[0] = coefficients_.values[0];
      model_coefficients[1] = coefficients_.values[1];
      model_coefficients[2] = coefficients_.values[2];
      model_coefficients[3] = coefficients_.values[3];

      typename pcl::PointCloud<PointT>::Ptr projected(new pcl::PointCloud<PointT>);
//      pcl::copyPointCloud(*cloud_, inliers_, *projected);
      pcl::SampleConsensusModelPlane<PointT> sacmodel (cloud_);
      sacmodel.projectPoints (inliers_.indices, model_coefficients, *projected, false);

      pcl::VoxelGrid<PointT> vg;
      vg.setInputCloud (projected);
      float leaf_size_ = resolution;
      vg.setLeafSize (leaf_size_, leaf_size_, leaf_size_);
      vg.filter (*plane_cloud);
      return plane_cloud;
    }

    typename pcl::PointCloud<PointT>::Ptr
    getConvexHullCloud() {
        typename pcl::PointCloud<PointT>::Ptr convex_hull_cloud (new pcl::PointCloud<PointT>);

        pcl::ConvexHull<PointT> convex_hull;
        convex_hull.setInputCloud (projectPlaneCloud());
        convex_hull.setDimension (2);
        convex_hull.setComputeAreaVolume (false);
        pcl::PolygonMeshPtr mesh_out(new pcl::PolygonMesh);

        std::vector<pcl::Vertices> polygons;
        convex_hull.reconstruct (*convex_hull_cloud, polygons);
        convex_hull.reconstruct (*mesh_out);
        convex_hull_ = mesh_out;

        return convex_hull_cloud;
    }

    bool operator < (const PlaneModel& pm2) const
    {
        return (inliers_.indices.size() < pm2.inliers_.indices.size());
    }

    bool operator > (const PlaneModel& pm2) const
    {
        return (inliers_.indices.size() > pm2.inliers_.indices.size());
    }


    typedef boost::shared_ptr< PlaneModel> Ptr;
    typedef boost::shared_ptr< PlaneModel const> ConstPtr;

  };
}

#endif
