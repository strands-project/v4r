#include <v4r/recognition/boost_graph_extension.h>

namespace v4r
{

template<typename PointT>
View<PointT>::View ()
{
//    pScenePCl.reset ( new pcl::PointCloud<pcl::PointXYZRGB> );
    scene_f_.reset ( new pcl::PointCloud<PointT> );
    scene_normals_.reset ( new pcl::PointCloud<pcl::Normal> );
    kp_normals_.reset ( new pcl::PointCloud<pcl::Normal> );
////    pIndices_above_plane.reset ( new pcl::PointIndices );
//    pSiftSignatures_.reset ( new pcl::PointCloud<FeatureT> );
    has_been_hopped_ = false;
    cumulative_weight_to_new_vrtx_ = 0;
}

}
