#ifndef V4R_PLANE_MODEL_H__
#define V4R_PLANE_MODEL_H__

#include <pcl/common/common.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/PolygonMesh.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/voxel_grid.h>
#include <v4r/core/macros.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace v4r
{
  template<typename PointT>
  class V4R_EXPORTS PlaneModel
  {
  private:
      mutable pcl::visualization::PCLVisualizer::Ptr vis_;
      mutable int vp1_, vp2_, vp3_, vp4_;

  public:
    Eigen::Vector4f coefficients_;
//    typename pcl::PointCloud<PointT>::Ptr plane_cloud_;
    pcl::PolygonMeshPtr convex_hull_;
    typename pcl::PointCloud<PointT>::Ptr cloud_;
    std::vector<int> inliers_;

    typename pcl::PointCloud<PointT>::Ptr
    projectPlaneCloud(float resolution=0.005f) const
    {
      typename pcl::PointCloud<PointT>::Ptr plane_cloud (new pcl::PointCloud<PointT>);
      typename pcl::PointCloud<PointT>::Ptr projected(new pcl::PointCloud<PointT>);
//      pcl::copyPointCloud(*cloud_, inliers_, *projected);
      pcl::SampleConsensusModelPlane<PointT> sacmodel (cloud_);
      sacmodel.projectPoints (inliers_, coefficients_, *projected, false);

      pcl::VoxelGrid<PointT> vg;
      vg.setInputCloud (projected);
      float leaf_size_ = resolution;
      vg.setLeafSize (leaf_size_, leaf_size_, leaf_size_);
      vg.filter (*plane_cloud);
      return plane_cloud;
    }

    typename pcl::PointCloud<PointT>::Ptr
    getConvexHullCloud()
    {
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

    bool operator < (const PlaneModel& pm2) const { return inliers_.size() < pm2.inliers_.size(); }
    bool operator > (const PlaneModel& pm2) const  { return inliers_.size() > pm2.inliers_.size(); }

    void
    visualize()
    {
        if(!vis_)
        {
            vis_.reset (new pcl::visualization::PCLVisualizer("plane visualization"));
            vis_->createViewPort(0,0,0.25,1,vp1_);
            vis_->createViewPort(0.25,0,0.50,1,vp2_);
            vis_->createViewPort(0.5,0,0.75,1,vp3_);
            vis_->createViewPort(0.75,0,1,1,vp4_);
        }
        vis_->removeAllPointClouds();
        vis_->removeAllShapes();
        vis_->addText("Input", 10, 10, 15, 1,1,1,"input", vp1_);
        vis_->addText("Convex hull points", 10, 10, 15, 1,1,1,"convex_hull_pts", vp2_);
        vis_->addText("plane", 10, 10, 15, 1,1,1,"plane", vp3_);
        vis_->addText("plane (inliers)", 10, 10, 15, 1,1,1,"plane(inliers)", vp4_);
        vis_->addPointCloud(cloud_, "cloud", vp1_);
        vis_->addPointCloud(getConvexHullCloud(), "convex_hull", vp2_);
        vis_->addPointCloud(projectPlaneCloud(), "projected plane cloud", vp3_);


        typename pcl::PointCloud<PointT>::Ptr projected(new pcl::PointCloud<PointT>);
        pcl::copyPointCloud(*cloud_, inliers_, *projected);
        vis_->addPointCloud(projected, "projected plane cloud2", vp4_);
        vis_->spin();
    }

    typedef boost::shared_ptr< PlaneModel> Ptr;
    typedef boost::shared_ptr< PlaneModel const> ConstPtr;

  };
}

#endif
