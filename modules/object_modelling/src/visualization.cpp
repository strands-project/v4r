#include <v4r/object_modelling/do_learning.h>
#include <pcl/common/transforms.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/pcl_visualization_utils.h>

#include <opencv2/opencv.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/io/filesystem.h>
#include <time.h>
#include <stdlib.h>

namespace v4r
{
namespace object_modelling
{

void
DOL::createBigCloud()
{
     std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > keyframes_used;
     std::vector< pcl::PointCloud<pcl::Normal>::Ptr > normals_used;
     std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > cameras_used;
     std::vector<pcl::PointCloud<IndexPoint> > object_indices_clouds;
     std::vector<std::vector<float> > weights;
     std::vector<std::vector<size_t> > indices_used;

     size_t num_frames = grph_.size();
     weights.resize(num_frames);
     indices_used.resize(num_frames);
     object_indices_clouds.resize(num_frames);
     keyframes_used.resize(num_frames);
     normals_used.resize(num_frames);
     cameras_used.resize(num_frames);

     // only used keyframes with have object points in them
     size_t kept_keyframes=0;
     for (size_t view_id = 0; view_id < grph_.size(); view_id++)
     {
         // scene reconstruction without noise model
         pcl::PointCloud<PointT>::Ptr cloud_trans (new pcl::PointCloud<PointT>());
         pcl::transformPointCloud(*grph_[view_id].cloud_, *cloud_trans, grph_[view_id].camera_pose_);
         *big_cloud_ += *cloud_trans;

         // object reconstruction without noise model
         pcl::PointCloud<PointT>::Ptr segmented_trans (new pcl::PointCloud<PointT>());
         pcl::copyPointCloud(*cloud_trans, grph_[view_id].obj_mask_step_.back(), *segmented_trans);
         *big_cloud_segmented_ += *segmented_trans;


         //using noise model
         if ( grph_[view_id].obj_mask_step_.back().size() )
         {
             keyframes_used[ kept_keyframes ] = grph_[view_id].cloud_;
             normals_used [ kept_keyframes ] = grph_[view_id].normal_;
             cameras_used [ kept_keyframes ] = grph_[view_id].camera_pose_;
             indices_used[ kept_keyframes ] = createIndicesFromMask( grph_[view_id].obj_mask_step_.back() );

             object_indices_clouds[ kept_keyframes ].points.resize( indices_used[ kept_keyframes ].size());

             for(size_t k=0; k < indices_used[ kept_keyframes ].size(); k++)
             {
                 object_indices_clouds[ kept_keyframes ].points[k].idx = (int)indices_used[ kept_keyframes ][k];
             }
             kept_keyframes++;
         }
     }
     weights.resize(kept_keyframes);
     indices_used.resize(kept_keyframes);
     object_indices_clouds.resize(kept_keyframes);
     keyframes_used.resize(kept_keyframes);
     normals_used.resize(kept_keyframes);
     cameras_used.resize(kept_keyframes);

     if ( kept_keyframes > 0)
     {
         //compute noise weights
         for(size_t i=0; i < kept_keyframes; i++)
         {
             v4r::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
             nm.setInputCloud(keyframes_used[i]);
             nm.setInputNormals(normals_used[i]);
             nm.setLateralSigma(0.001);
             nm.setMaxAngle(60.f);
             nm.setUseDepthEdges(true);
             nm.compute();
             nm.getWeights(weights[i]);
         }

         pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
         v4r::utils::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration (nm_int_param_);
         nmIntegration.setInputClouds(keyframes_used);
         nmIntegration.setWeights(weights);
         nmIntegration.setTransformations(cameras_used);
         nmIntegration.setInputNormals(normals_used);
         nmIntegration.setIndices( indices_used );
         nmIntegration.compute(octree_cloud);
         pcl::PointCloud<pcl::Normal>::Ptr octree_normals;
         nmIntegration.getOutputNormals(octree_normals);

         pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_with_normals_oriented (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
         pcl::concatenateFields(*octree_normals, *octree_cloud, *filtered_with_normals_oriented);
         pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBNormal> sor;
         sor.setInputCloud (filtered_with_normals_oriented);
         sor.setMeanK (50);
         sor.setStddevMulThresh (3.0);
         sor.filter (*big_cloud_segmented_refined_);
     }
}

void
DOL::visualize_clusters()
{
    srand (time(NULL));
    if (!vis_seg_)
        vis_seg_.reset(new pcl::visualization::PCLVisualizer("smooth clusters"));
//        size_t total_sub_windows = view.planes_.size();
//        size_t cols = static_cast<size_t>(total_sub_windows * 16.f / (16.f*9.f) + 0.5f); // optimized for 16:9
//        size_t rows = std::ceil(total_sub_windows/float(cols));
//        std::vector<int> viewports = v4r::common::pcl_visualizer::visualization_framework (*vis_seg_, cols, rows);

//    size_t max_clusters=0;
//    for (size_t view_id = 0; view_id < grph_.size(); view_id++)
//    {
//      if( grph_[view_id].planes_.size() > max_clusters )
//         max_clusters = grph_[view_id].planes_.size();
//    }
    //    std::vector<int> viewports = v4r::common::pcl_visualizer::visualization_framework (*vis_seg_, grph_.size(), max_clusters+1);

    std::vector<std::string> subwindow_title;
    subwindow_title.push_back("original");
    subwindow_title.push_back("filtered");
    subwindow_title.push_back("passed");
    std::vector<int> viewports = v4r::common::pcl_visualizer::visualization_framework (*vis_seg_, grph_.size(), 3, subwindow_title);

    vis_seg_->removeAllPointClouds();

    for (size_t view_id = 0; view_id < grph_.size(); view_id++)
    {
        std::stringstream name;
        name << "cloud_" << view_id;
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler(grph_[view_id].cloud_);
        vis_seg_->addPointCloud(grph_[view_id].cloud_, rgb_handler, name.str(), viewports[view_id*3 + 0]);

        name << "_";
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler2(grph_[view_id].cloud_);
        vis_seg_->addPointCloud(grph_[view_id].cloud_, rgb_handler2, name.str(), viewports[view_id*3 + 1]);

        name << "_";
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler3(grph_[view_id].cloud_);
        vis_seg_->addPointCloud(grph_[view_id].cloud_, rgb_handler3, name.str(), viewports[view_id*3 + 2]);

        for(size_t cluster_id=0; cluster_id<grph_[view_id].planes_.size(); cluster_id++)
        {
            name << "_segmented_" << cluster_id;
            pcl::PointCloud<PointT>::Ptr segmented (new pcl::PointCloud<PointT>());
            pcl::copyPointCloud(*grph_[view_id].cloud_, grph_[view_id].planes_[cluster_id].indices, *segmented);

            const float r=50+rand()%205;
            const float g=50+rand()%205;
            const float b=50+rand()%205;

            std::stringstream text;
            const double num_vis = static_cast<double>(grph_[view_id].planes_[cluster_id].visible_indices.size()) / grph_[view_id].planes_[cluster_id].within_chop_z_indices.size();
            const double num_obj = static_cast<double>(grph_[view_id].planes_[cluster_id].object_indices.size()) / grph_[view_id].planes_[cluster_id].within_chop_z_indices.size();
            text << "v:" << std::setprecision(2) << num_vis << " / o: " << std::setprecision(2) << num_obj;

            if(grph_[view_id].planes_[cluster_id].is_filtered)
            {
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> colored_src (segmented, r, g, b);
                vis_seg_->addPointCloud(segmented, colored_src, name.str(), viewports[view_id*3 + 1]);
                vis_seg_->addText(text.str(),40, 10*cluster_id,10, r/255, g/255, b/255, name.str(), viewports[view_id*3 + 1]);
            }
            else
            {
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> colored_src (segmented, r, g, b);
                vis_seg_->addPointCloud(segmented, colored_src, name.str(), viewports[view_id*3 + 2]);
                vis_seg_->addText(text.str(), 40, 10*cluster_id,10,r,g,b,name.str(), viewports[view_id*3 + 2]);
            }
        }

    }
    vis_seg_->spinOnce();
}


void
DOL::visualize()
{
    visualize_clusters();
    createBigCloud();
    if (!vis_reconstructed_)
    {
        vis_reconstructed_.reset(new pcl::visualization::PCLVisualizer("segmented cloud"));
        vis_reconstructed_viewpoint_.resize( 3 );
        vis_reconstructed_->createViewPort(0,0,0.5,0.5,vis_reconstructed_viewpoint_[0]);
        vis_reconstructed_->createViewPort(0.5,0,1,0.5,vis_reconstructed_viewpoint_[1]);
        vis_reconstructed_->createViewPort(0.5,0.5,1,1,vis_reconstructed_viewpoint_[2]);
    }
    vis_reconstructed_->removeAllPointClouds(vis_reconstructed_viewpoint_[0]);
    vis_reconstructed_->removeAllPointClouds(vis_reconstructed_viewpoint_[1]);
    vis_reconstructed_->removeAllPointClouds(vis_reconstructed_viewpoint_[2]);
    vis_reconstructed_->addPointCloud(big_cloud_, "big", vis_reconstructed_viewpoint_[0]);
    vis_reconstructed_->addPointCloud(big_cloud_segmented_, "segmented", vis_reconstructed_viewpoint_[1]);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_segmented_refined (new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::copyPointCloud(*big_cloud_segmented_refined_, *big_cloud_segmented_refined);
    vis_reconstructed_->addPointCloud(big_cloud_segmented_refined, "segmented_refined", vis_reconstructed_viewpoint_[2]);
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
    vis_->removeAllPointClouds();
    std::vector<std::string> subwindow_title;
    subwindow_title.push_back("scene + init. ind. / transf. cluster");
    subwindow_title.push_back("filtered scene");
    subwindow_title.push_back("supervoxelled scene");
    subwindow_title.push_back("after nearest neighbor search");
    subwindow_title.push_back("after removing points on (other) planes");
    subwindow_title.push_back("points enforced by supervoxel patch");
    subwindow_title.push_back("after growing points within smooth surfaces");
    subwindow_title.push_back("after 2D erosion");

    size_t num_subwindows = grph_.back().obj_mask_step_.size() + 3;
    vis_viewpoint_ = v4r::common::pcl_visualizer::visualization_framework (*vis_, grph_.size(), num_subwindows, subwindow_title);

    for (size_t view_id = 0; view_id < grph_.size(); view_id++)
    {
        size_t subwindow_id=0;

        pcl::PointCloud<PointT>::Ptr cloud_trans (new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr cloud_trans_filtered (new pcl::PointCloud<PointT>());
        pcl::transformPointCloud(*grph_[view_id].cloud_, *cloud_trans, grph_[view_id].camera_pose_);
        pcl::copyPointCloud(*cloud_trans, grph_[view_id].scene_points_, *cloud_trans_filtered);

        std::stringstream cloud_name;
        cloud_name << "cloud_" << grph_[view_id].id_;
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler(cloud_trans);
        vis_->addPointCloud(cloud_trans, rgb_handler, cloud_name.str(), vis_viewpoint_[view_id * num_subwindows + subwindow_id]);

        if (!grph_[view_id].is_pre_labelled_)
        {
            cloud_name << "_search_pts";
            pcl::PointCloud<PointT>::Ptr cloud_trans_tmp (new pcl::PointCloud<PointT>());
            pcl::transformPointCloud(*grph_[view_id].transferred_cluster_, *cloud_trans_tmp, grph_[view_id].camera_pose_);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> green_source (cloud_trans_tmp, 0, 255, 0);
            vis_->addPointCloud(cloud_trans_tmp, green_source, cloud_name.str(), vis_viewpoint_[view_id * num_subwindows + subwindow_id++]);
        }
        else
        {
            cloud_name << "_initial_indices";
            pcl::PointCloud<PointT>::Ptr obj_trans_tmp (new pcl::PointCloud<PointT>());
            pcl::copyPointCloud(*cloud_trans, grph_[view_id].obj_mask_step_[0], *obj_trans_tmp);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> red_source (obj_trans_tmp, 255, 0, 0);
            vis_->addPointCloud(obj_trans_tmp, red_source, cloud_name.str(), vis_viewpoint_[view_id * num_subwindows + subwindow_id++]);
        }

        cloud_name << "_filtered";
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler1(cloud_trans_filtered);
        vis_->addPointCloud(cloud_trans_filtered, rgb_handler1, cloud_name.str(), vis_viewpoint_[view_id * num_subwindows + subwindow_id++]);

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sv_trans (new pcl::PointCloud<pcl::PointXYZRGBA>());
        pcl::transformPointCloud(*grph_[view_id].supervoxel_cloud_, *sv_trans, grph_[view_id].camera_pose_);
        cloud_name << "_supervoxellized";
        vis_->addPointCloud(sv_trans, cloud_name.str(), vis_viewpoint_[view_id * num_subwindows + subwindow_id++]);

        for(size_t step_id=0; step_id<grph_[view_id].obj_mask_step_.size(); step_id++)
        {
            if (grph_[view_id].is_pre_labelled_ && step_id==0) // initial indices already shown in other subwindow
            {
                continue;
            }

            pcl::PointCloud<PointT>::Ptr segmented (new pcl::PointCloud<PointT>());
            pcl::PointCloud<PointT>::Ptr segmented_trans (new pcl::PointCloud<PointT>());
            pcl::copyPointCloud(*grph_[view_id].cloud_, grph_[view_id].obj_mask_step_[step_id], *segmented);
            pcl::transformPointCloud(*segmented, *segmented_trans, grph_[view_id].camera_pose_);
            cloud_name << "__step_" << step_id;
            pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler_tmp(segmented_trans);
            vis_->addPointCloud(segmented_trans, rgb_handler_tmp, cloud_name.str(), vis_viewpoint_[view_id * num_subwindows + subwindow_id++]);
        }
    }
    vis_->spin();
}


void
DOL::writeImagesToDisk(const std::string &path, bool crop)
{
    v4r::io::createDirIfNotExist(path);

    for (size_t view_id = 0; view_id < grph_.size(); view_id++)
    {
        std::stringstream filename;
        filename << path << "/" << view_id << ".jpg";
        cv::Mat_ < cv::Vec3b > colorImage;
        PCLOpenCV::ConvertPCLCloud2Image<PointT> (grph_[view_id].cloud_, colorImage);
        cv::imwrite( filename.str(), colorImage);

        std::stringstream filename_sv;
        filename_sv << path << "/" << view_id << "_sv.jpg";
        PCLOpenCV::ConvertPCLCloud2Image<pcl::PointXYZRGBA> (grph_[view_id].supervoxel_cloud_organized_, colorImage);
        cv::imwrite( filename_sv.str(), colorImage);

        std::stringstream filename_filtered;
        filename_filtered << path << "/" << view_id << "_filtered.jpg";
        pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*grph_[view_id].cloud_, grph_[view_id].scene_points_, *cloud_filtered);
        PCLOpenCV::ConvertUnorganizedPCLCloud2Image<PointT> (cloud_filtered, colorImage);
        cv::imwrite( filename_filtered.str(), colorImage);

        std::stringstream filename_search_points;
        filename_search_points << path << "/" << view_id << "_search_pts.jpg";
        pcl::PointCloud<PointT>::Ptr cloud_with_search_pts (new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr search_pts (new pcl::PointCloud<PointT>());
        *cloud_with_search_pts = *grph_[view_id].cloud_;

        if (!grph_[view_id].is_pre_labelled_)
        {
            pcl::copyPointCloud(*grph_[view_id].transferred_cluster_, *search_pts);
            for (size_t s_id=0; s_id < search_pts->points.size(); s_id++)
            {
                search_pts->points[s_id].r = 255;
                search_pts->points[s_id].g = 0;
                search_pts->points[s_id].b = 0;
                search_pts->points[s_id].z -= 0.01f;    // just to force the search point in the foreground for visualization
            }
        }
        else
        {
            pcl::copyPointCloud(*grph_[view_id].cloud_, grph_[view_id].obj_mask_step_[0], *search_pts);
            for (size_t s_id=0; s_id < search_pts->points.size(); s_id++)
            {
                search_pts->points[s_id].r = 0;
                search_pts->points[s_id].g = 255;
                search_pts->points[s_id].b = 0;
                search_pts->points[s_id].z -= 0.01f;    // just to force the search point in the foreground for visualization
            }
        }

        *cloud_with_search_pts += *search_pts;
        PCLOpenCV::ConvertUnorganizedPCLCloud2Image<PointT> (cloud_with_search_pts, colorImage);
        cv::imwrite( filename_search_points.str(), colorImage);


        for(size_t step_id=0; step_id<grph_[view_id].obj_mask_step_.size(); step_id++)
        {
            if (grph_[view_id].is_pre_labelled_ && step_id==0) // initial indices already shown in other subwindow
            {
                continue;
            }

            pcl::PointCloud<PointT>::Ptr segmented (new pcl::PointCloud<PointT>());
            pcl::copyPointCloud(*grph_[view_id].cloud_, grph_[view_id].obj_mask_step_[step_id], *segmented);

            std::stringstream filename_step;
            filename_step << path << "/" << view_id << "_step_" << step_id << ".jpg";
            PCLOpenCV::ConvertUnorganizedPCLCloud2Image<PointT> (segmented, colorImage, crop);
            cv::imwrite( filename_step.str(), colorImage);
        }
    }
}

}
}
