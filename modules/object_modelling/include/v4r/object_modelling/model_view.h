#ifndef V4R_OBJECT_MODELLING_MODELVIEW_H__
#define V4R_OBJECT_MODELLING_MODELVIEW_H__

/*
 * Author: Thomas Faeulhammer
 * Date: 20th July 2015
 * License: MIT
 *
 * This class represents a single-view of the object model used for learning
 *
 * */
#include <pcl/common/common.h>

namespace v4r
{
    namespace object_modelling
    {
        class modelView
        {
        public:
            typedef pcl::PointXYZRGB PointT;
            typedef pcl::Histogram<128> FeatureT;

            pcl::PointCloud<PointT>::Ptr  cloud_;
            pcl::PointCloud<pcl::Normal>::Ptr  normal_;
            pcl::PointCloud<PointT>::Ptr  transferred_cluster_;

            pcl::PointCloud<FeatureT>::Ptr  sift_signatures_;
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr  supervoxel_cloud_;

            std::vector< size_t > obj_indices_eroded_to_original_;
            std::vector< size_t > obj_indices_2_to_filtered_;
            std::vector< size_t > scene_points_;
            std::vector< size_t > transferred_nn_points_;
            std::vector< size_t > transferred_object_indices_without_plane_;
            std::vector< size_t > initial_indices_good_to_unfiltered_;
            std::vector< size_t > obj_indices_3_to_original_;
            std::vector< size_t > sift_keypoint_indices_;
            Eigen::Matrix4f camera_pose_;
            Eigen::Matrix4f tracking_pose_;
            bool tracking_pose_set_ = false;
            bool camera_pose_set_ = false;

            size_t id_; // might be redundant

            bool is_pre_labelled_;

            modelView()
            {
                cloud_.reset(new pcl::PointCloud<PointT>());
                normal_.reset(new pcl::PointCloud<pcl::Normal>());
                transferred_cluster_.reset(new pcl::PointCloud<PointT>());
                supervoxel_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGBA>());
                sift_signatures_.reset (new pcl::PointCloud<FeatureT>());
                is_pre_labelled_ = false;
            }
        };
    }
}


#endif //V4R_OBJECT_MODELLING_MODELVIEW_H__
