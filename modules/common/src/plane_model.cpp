#include <v4r/common/plane_model.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/impl/instantiate.hpp>

namespace v4r
{

//template<typename PointT>
//typename pcl::PointCloud<PointT>::Ptr
//PlaneModel<PointT>::getConvexHullCloud()
//{
//    typename pcl::PointCloud<PointT>::Ptr convex_hull_cloud (new pcl::PointCloud<PointT>);

//    pcl::ConvexHull<PointT> convex_hull;
//    convex_hull.setInputCloud (projectPlaneCloud());
//    convex_hull.setDimension (2);
//    convex_hull.setComputeAreaVolume (false);
//    pcl::PolygonMeshPtr mesh_out(new pcl::PolygonMesh);

//    std::vector<pcl::Vertices> polygons;
//    convex_hull.reconstruct (*convex_hull_cloud, polygons);
//    convex_hull.reconstruct (*mesh_out);
//    convex_hull_ = mesh_out;

//    return convex_hull_cloud;
//}

//template<typename PointT>
//typename pcl::PointCloud<PointT>::Ptr
//PlaneModel<PointT>::projectPlaneCloud (float resolution) const
//{
//  typename pcl::PointCloud<PointT>::Ptr plane_cloud (new pcl::PointCloud<PointT>);
//  typename pcl::PointCloud<PointT>::Ptr projected(new pcl::PointCloud<PointT>);
////      pcl::copyPointCloud(*cloud_, inliers_, *projected);
//  pcl::SampleConsensusModelPlane<PointT> sacmodel (cloud_);
//  sacmodel.projectPoints (inliers_, coefficients_, *projected, false);

//  pcl::VoxelGrid<PointT> vg;
//  vg.setInputCloud (projected);
//  float leaf_size_ = resolution;
//  vg.setLeafSize (leaf_size_, leaf_size_, leaf_size_);
//  vg.filter (*plane_cloud);
//  return plane_cloud;
//}


#define PCL_INSTANTIATE_PlaneModel(T) template class V4R_EXPORTS PlaneModel<T>;
PCL_INSTANTIATE(PlaneModel, PCL_XYZ_POINT_TYPES )


template<>
V4R_EXPORTS void
PlaneModel<pcl::PointXYZRGB>::visualize()
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
//    vis_->addPointCloud(getConvexHullCloud(), "convex_hull", vp2_);
//    vis_->addPointCloud(projectPlaneCloud(), "projected plane cloud", vp3_);


    typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr projected(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud_, inliers_, *projected);
    vis_->addPointCloud(projected, "projected plane cloud2", vp4_);
    vis_->spin();
}

}
