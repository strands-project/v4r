#ifndef V4R_PLANE_MODEL_H__
#define V4R_PLANE_MODEL_H__

#include <pcl/common/common.h>
#include <pcl/PolygonMesh.h>
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
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef boost::shared_ptr< PlaneModel> Ptr;
    typedef boost::shared_ptr< PlaneModel const> ConstPtr;

    Eigen::Vector4f coefficients_;
    pcl::PolygonMeshPtr convex_hull_;
    typename pcl::PointCloud<PointT>::ConstPtr cloud_;
    std::vector<int> inliers_;

//    typename pcl::PointCloud<PointT>::Ptr
//    projectPlaneCloud(float resolution=0.005f) const;

//    typename pcl::PointCloud<PointT>::Ptr
//    getConvexHullCloud();

    bool operator < (const PlaneModel& pm2) const { return inliers_.size() < pm2.inliers_.size(); }
    bool operator > (const PlaneModel& pm2) const  { return inliers_.size() > pm2.inliers_.size(); }

    void
    visualize();


};
}

#endif
