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

            pcl::PointIndices obj_indices_eroded_to_original_;
            pcl::PointIndices obj_indices_2_to_filtered_;
            pcl::PointIndices scene_points_;
            pcl::PointIndices transferred_nn_points_;
            pcl::PointIndices transferred_object_indices_without_plane_;
            pcl::PointIndices initial_indices_good_to_unfiltered_;
            pcl::PointIndices obj_indices_3_to_original_;
            pcl::PointIndices sift_keypoint_indices_;
            Eigen::Matrix4f camera_pose_;

            size_t id; // might be redundant

            modelView()
            {
                cloud_.reset(new pcl::PointCloud<PointT>());
                normal_.reset(new pcl::PointCloud<pcl::Normal>());
                transferred_cluster_.reset(new pcl::PointCloud<PointT>());
                supervoxel_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGBA>());
                sift_signatures_.reset (new pcl::PointCloud<FeatureT>());
            }
        };
    }
}


#endif //V4R_OBJECT_MODELLING_MODELVIEW_H__
