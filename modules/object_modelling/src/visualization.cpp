#include <v4r/object_modelling/do_learning.h>
#include <pcl/common/transforms.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/pcl_visualization_utils.h>

#define NUM_SUBWINDOWS 7

namespace v4r
{
namespace object_modelling
{

void
DOL::createBigCloud()
{
    for (size_t view_id = 0; view_id < grph_.size(); view_id++)
    {
        pcl::PointCloud<PointT>::Ptr cloud_trans (new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr cloud_trans_filtered (new pcl::PointCloud<PointT>());
        pcl::transformPointCloud(*grph_[view_id].cloud_, *cloud_trans, grph_[view_id].camera_pose_);
        pcl::copyPointCloud(*cloud_trans, grph_[view_id].scene_points_, *cloud_trans_filtered);
        *big_cloud_ += *cloud_trans_filtered;

        pcl::PointCloud<PointT>::Ptr segmented_trans (new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*cloud_trans_filtered, grph_[view_id].obj_indices_2_to_filtered_, *segmented_trans);
        *big_cloud_segmented_ += *segmented_trans;
    }
}


void
DOL::visualize()
{
    createBigCloud();
    if (!vis_reconstructed_)
    {
        vis_reconstructed_.reset(new pcl::visualization::PCLVisualizer("segmented cloud"));
        vis_reconstructed_viewpoint_.resize( 2 );
        vis_reconstructed_->createViewPort(0,0,0.5,1,vis_reconstructed_viewpoint_[0]);
        vis_reconstructed_->createViewPort(0.5,0,1,1,vis_reconstructed_viewpoint_[1]);
    }
    vis_reconstructed_->removeAllPointClouds(vis_reconstructed_viewpoint_[0]);
    vis_reconstructed_->removeAllPointClouds(vis_reconstructed_viewpoint_[1]);
    vis_reconstructed_->addPointCloud(big_cloud_, "big", vis_reconstructed_viewpoint_[0]);
    vis_reconstructed_->addPointCloud(big_cloud_segmented_, "segmented", vis_reconstructed_viewpoint_[1]);
    vis_reconstructed_->spinOnce();

    if (!vis_) {
        vis_.reset(new pcl::visualization::PCLVisualizer());
    }
    else
    {
        for (size_t vp_id=0; vp_id < vis_viewpoint_.size(); vp_id++)
        {
            vis_->removeAllPointClouds( vis_viewpoint_[vp_id] );
        }
    }
    std::vector<std::string> subwindow_title;
    subwindow_title.push_back("original scene");
    subwindow_title.push_back("filtered scene");
    subwindow_title.push_back("supervoxelled scene");
    subwindow_title.push_back("after nearest neighbor search");
    subwindow_title.push_back("good points");
    subwindow_title.push_back("before 2D erosion");
    subwindow_title.push_back("after 2D erosion");

    vis_viewpoint_ = v4r::common::pcl_visualizer::visualization_framework (vis_, grph_.size(), NUM_SUBWINDOWS, subwindow_title);

    for (size_t view_id = 0; view_id < grph_.size(); view_id++)
    {
        pcl::PointCloud<PointT>::Ptr cloud_trans (new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr cloud_trans_filtered (new pcl::PointCloud<PointT>());
        pcl::transformPointCloud(*grph_[view_id].cloud_, *cloud_trans, grph_[view_id].camera_pose_);
        pcl::copyPointCloud(*cloud_trans, grph_[view_id].scene_points_, *cloud_trans_filtered);

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sv_trans (new pcl::PointCloud<pcl::PointXYZRGBA>());
        pcl::transformPointCloud(*grph_[view_id].supervoxel_cloud_, *sv_trans, grph_[view_id].camera_pose_);

        pcl::PointCloud<PointT>::Ptr segmented (new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr segmented_trans (new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*grph_[view_id].cloud_, grph_[view_id].transferred_nn_points_, *segmented);
        pcl::transformPointCloud(*segmented, *segmented_trans, grph_[view_id].camera_pose_);

        pcl::PointCloud<PointT>::Ptr segmented2_trans (new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*cloud_trans_filtered, grph_[view_id].initial_indices_good_to_unfiltered_, *segmented2_trans);

        pcl::PointCloud<PointT>::Ptr segmented3_trans (new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*cloud_trans_filtered, grph_[view_id].obj_indices_2_to_filtered_, *segmented3_trans);

        pcl::PointCloud<PointT>::Ptr segmented_eroded_trans (new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*cloud_trans, grph_[view_id].obj_indices_eroded_to_original_, *segmented_eroded_trans);

        std::stringstream cloud_name;
        cloud_name << "cloud_" << grph_[view_id].id_;
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler0(cloud_trans);
        vis_->addPointCloud(cloud_trans, rgb_handler0, cloud_name.str(), vis_viewpoint_[view_id * NUM_SUBWINDOWS + 0]);

        size_t subwindow_id=0;

        if (!grph_[view_id].is_pre_labelled_)
        {
            cloud_name << "_search_pts";
            pcl::PointCloud<PointT>::Ptr cloud_trans_tmp (new pcl::PointCloud<PointT>());
            pcl::transformPointCloud(*grph_[view_id].transferred_cluster_, *cloud_trans_tmp, grph_[view_id].camera_pose_);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> green_source (cloud_trans_tmp, 0, 255, 0);
            vis_->addPointCloud(cloud_trans_tmp, green_source, cloud_name.str(), vis_viewpoint_[view_id * NUM_SUBWINDOWS + subwindow_id++]);
        }
        else
        {
            cloud_name << "_search_pts";
            pcl::PointCloud<PointT>::Ptr cloud_trans_tmp (new pcl::PointCloud<PointT>());
            pcl::PointCloud<PointT>::Ptr obj_trans_tmp (new pcl::PointCloud<PointT>());
            pcl::transformPointCloud(*grph_[view_id].cloud_, *cloud_trans_tmp, grph_[view_id].camera_pose_);
            pcl::copyPointCloud(*cloud_trans_tmp, grph_[view_id].transferred_nn_points_, *obj_trans_tmp);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> red_source (obj_trans_tmp, 255, 0, 0);
            vis_->addPointCloud(obj_trans_tmp, red_source, cloud_name.str(), vis_viewpoint_[view_id * NUM_SUBWINDOWS + subwindow_id++]);
        }

        cloud_name << "_filtered";
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler1(cloud_trans_filtered);
        vis_->addPointCloud(cloud_trans_filtered, rgb_handler1, cloud_name.str(), vis_viewpoint_[view_id * NUM_SUBWINDOWS + subwindow_id++]);

        cloud_name << "_supervoxellized";
        vis_->addPointCloud(sv_trans, cloud_name.str(), vis_viewpoint_[view_id * NUM_SUBWINDOWS + subwindow_id++]);

        cloud_name << "_nearest_neighbor";
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler2(segmented_trans);
        vis_->addPointCloud(segmented_trans, rgb_handler2, cloud_name.str(), vis_viewpoint_[view_id * NUM_SUBWINDOWS + subwindow_id++]);

        cloud_name << "_good";
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler3(segmented2_trans);
        vis_->addPointCloud(segmented2_trans, rgb_handler3, cloud_name.str(), vis_viewpoint_[view_id * NUM_SUBWINDOWS + subwindow_id++]);

        cloud_name << "_region_grown";
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler4(segmented3_trans);
        vis_->addPointCloud(segmented3_trans, rgb_handler4, cloud_name.str(), vis_viewpoint_[view_id * NUM_SUBWINDOWS + subwindow_id++]);

        cloud_name << "_eroded";
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler5(segmented_eroded_trans);
        vis_->addPointCloud(segmented_eroded_trans, rgb_handler5, cloud_name.str(), vis_viewpoint_[view_id * NUM_SUBWINDOWS + subwindow_id++]);
    }
    vis_->spin();
}

}
}
