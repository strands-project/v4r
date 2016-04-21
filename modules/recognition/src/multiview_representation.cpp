#include <v4r/recognition/multiview_representation.h>

namespace v4r
{

template<typename PointT>
View<PointT>::View ()
{
//    scene_f_.reset ( new pcl::PointCloud<PointT> );
    scene_normals_.reset ( new pcl::PointCloud<pcl::Normal> );
//    kp_normals_.reset ( new pcl::PointCloud<pcl::Normal> );
    absolute_pose_ = Eigen::Matrix4f::Identity();
    has_been_hopped_ = false;
    cumulative_weight_to_new_vrtx_ = 0;
}

template class V4R_EXPORTS View<pcl::PointXYZRGB>;
}

