/*
 * do_learning.cpp
 *
 * incrementally learning of objects by
 * transfering object indices from initial cloud to the remaining clouds
 * using given camera poses
 *
 *  Created on: June, 2015
 *      Author: Thomas Faeulhammer
 */


#ifndef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#endif

#include "v4r/object_modelling/incremental_object_learning.h"

#include <stdlib.h>
#include <thread>
#include <iostream>

#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#include <v4r/common/convertCloud.h>
#include <v4r/common/convertNormals.h>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/registration/metrics.h>
#include <v4r/common/binary_algorithms.h>
#include <v4r/common/normals.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <v4r/common/zbuffering.h>

#include <boost/graph/kruskal_min_spanning_tree.hpp>

#ifdef HAVE_SIFTGPU
    #include <v4r/features/sift_local_estimator.h>
#else
    #include <v4r/features/opencv_sift_local_estimator.h>
#endif

namespace v4r
{
namespace object_modelling
{

std::vector<bool>
IOL::extractEuclideanClustersSmooth (
        const pcl::PointCloud<PointT>::ConstPtr &cloud,
        const pcl::PointCloud<pcl::Normal> &normals,
        const std::vector<bool> &initial_mask,
        const std::vector<bool> &bg_mask) const
{
    assert (cloud->points.size () == normals.points.size ());

    pcl::octree::OctreePointCloudSearch<PointT> octree(0.005f);
    octree.setInputCloud ( cloud );
    octree.addPointsFromInputCloud ();

    // Create a bool vector of processed point indices, and initialize it to false
    std::vector<bool> to_grow = initial_mask;
    std::vector<bool> in_cluster = initial_mask;

    bool stop = false;
    while(!stop)
    {
        stop = true;
        std::vector<bool> is_new_point (cloud->points.size (), false);  // do as long as there is no new point
        std::vector<int> nn_indices;
        std::vector<float> nn_distances;

        for (size_t i = 0; i < cloud->points.size (); i++)
        {
            if (!to_grow[i])
                continue;

            if (octree.radiusSearch (cloud->points[i], param_.radius_, nn_indices, nn_distances))
            {
                for (size_t j = 0; j < nn_indices.size (); j++) // is nn_indices[0] the same point?
                {
                    if( !in_cluster[ nn_indices[j] ] && !bg_mask[ nn_indices[j] ])  // if nearest neighbor is not already an object and is not a point to be neglected (background)
                    {
                        //check smoothness constraint
                        Eigen::Vector3f n1 = normals.points[i].getNormalVector3fMap();
                        Eigen::Vector3f n2 = normals.points[nn_indices[j]].getNormalVector3fMap();
                        n1.normalize();
                        n2.normalize();
                        float dot_p = n1.dot(n2);

                        if (dot_p >= param_.eps_angle_)
                        {
                            stop = false;
                            is_new_point[ nn_indices[j] ] = true;
                            in_cluster[ nn_indices[j] ] = true;
                        }
                    }
                }
            }
        }
        to_grow = is_new_point;
    }
    return in_cluster;
}

void
IOL::updatePointNormalsFromSuperVoxels(const pcl::PointCloud<PointT>::Ptr & cloud,
                                            pcl::PointCloud<pcl::Normal>::Ptr & normals,
                                            const std::vector<bool> &obj_mask,
                                            std::vector<bool> &obj_mask_out,
                                            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &supervoxel_cloud,
                                            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &supervoxel_cloud_organized)
{
    assert( cloud->points.size() == normals->points.size() &&
            cloud->points.size() == obj_mask.size());

    pcl::SupervoxelClustering<PointT> super (param_.voxel_resolution_, param_.seed_resolution_, false);
    super.setInputCloud (cloud);
    super.setColorImportance (0.f);
    super.setSpatialImportance (0.5f);
    super.setNormalImportance (2.f);
    super.setNormalCloud(normals);
    std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
    super.extract (supervoxel_clusters);
    super.refineSupervoxels(2, supervoxel_clusters);
    supervoxel_cloud = super.getColoredVoxelCloud();
    supervoxel_cloud_organized = super.getColoredCloud();
    const pcl::PointCloud<pcl::PointXYZL>::Ptr supervoxels_labels_cloud = super.getLabeledCloud();

    std::cout << "Found " << supervoxel_clusters.size () << " supervoxels." << std::endl;

    const size_t max_label = super.getMaxLabel();
//    const pcl::PointCloud<pcl::PointNormal>::Ptr sv_normal_cloud = super.makeSupervoxelNormalCloud (supervoxel_clusters);

//    //count for all labels how many pixels are in the initial indices
    std::vector<size_t> label_count (max_label+1, 0);

    for(size_t i = 0; i < supervoxels_labels_cloud->points.size(); i++)
    {
        const size_t label = static_cast<size_t>(supervoxels_labels_cloud->points[i].label);
        label_count[ label ]++;
    }

    obj_mask_out.resize(cloud->points.size());

    for(size_t i = 0; i < cloud->points.size(); i++)
    {
        const size_t label = supervoxels_labels_cloud->points[i].label;
        assert(label<label_count.size());

        if ( !obj_mask[i] || !pcl::isFinite(cloud->points[i]))
        {
            obj_mask_out[i] = false;
            continue;
        }

        if( obj_mask[i] && label==0)    // label 0 means the point could not be associated to any supervoxel (see supervoxelclustering doc) - we will pass this point therefore by definition
        {
            obj_mask_out[i] = true;
            continue;
        }

        std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr >::const_iterator it = supervoxel_clusters.find(label);
        if (it != supervoxel_clusters.end())
        {
//            // refine normals
            const Eigen::Vector3f sv_normal = it->second->normal_.getNormalVector3fMap();
            normals->points[i].getNormalVector3fMap() = sv_normal;

            const size_t tot_pts_in_supervoxel = it->second->voxels_->points.size();
            if( label_count[label]  > param_.ratio_supervoxel_ * tot_pts_in_supervoxel)
                obj_mask_out[i] = true;
            else
                obj_mask_out[i] = false;
        }
        else
        {
            std::cerr << "Cluster for label does not exist" << std::endl;
            obj_mask_out[i] = true;
        }
    }
}

void
IOL::nnSearch(const pcl::PointCloud<PointT> &object_points, const pcl::PointCloud<PointT>::ConstPtr &search_cloud,  std::vector<bool> &obj_mask)
{
    pcl::octree::OctreePointCloudSearch<PointT> octree(0.005f);
    octree.setInputCloud ( search_cloud );
    octree.addPointsFromInputCloud ();
    nnSearch(object_points, octree, obj_mask);
}

void
IOL::nnSearch(const pcl::PointCloud<PointT> &object_points, pcl::octree::OctreePointCloudSearch<PointT> &octree,  std::vector<bool> &obj_mask)
{
    //find neighbours from transferred object points
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    for(size_t i=0; i < object_points.points.size(); i++)
    {
        if ( ! pcl::isFinite(object_points.points[i]) )
        {
            PCL_WARN ("Warning: Point is NaN.\n");    // not sure if this causes somewhere else a problem. This condition should not be fulfilled.
            continue;
        }
        if ( octree.radiusSearch (object_points.points[i], param_.radius_, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
        {
            for( size_t nn_id = 0; nn_id < pointIdxRadiusSearch.size(); nn_id++)
                obj_mask[ pointIdxRadiusSearch[ nn_id ] ] = true;
        }
    }
}

std::vector<bool>
IOL::erodeIndices(const std::vector< bool > &obj_mask, const pcl::PointCloud<PointT> & cloud)
{
    assert (obj_mask.size() == cloud.height * cloud.width);

    cv::Mat mask = cv::Mat(cloud.height, cloud.width, CV_8UC1);
    std::vector<bool> mask_out(obj_mask.size());

    for(size_t i=0; i < obj_mask.size(); i++)
    {
        int r,c;
        r = i / mask.cols;
        c = i % mask.cols;

        if (obj_mask[i])
            mask.at<unsigned char>(r,c) = 255;
        else
            mask.at<unsigned char>(r,c) = 0;
    }

    cv::Mat const structure_elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat close_result;
    cv::morphologyEx(mask, close_result, cv::MORPH_CLOSE, structure_elem);

    cv::Mat mask_dst;
    cv::erode(close_result, mask_dst, cv::Mat(), cv::Point(-1,-1), 3);

    //        cv::imshow("mask", mask);
    //        cv::imshow("close_result", close_result);
    //        cv::imshow("mask_dst", mask_dst);
    //        cv::waitKey(0);

    for(int r=0; r < mask_dst.rows; r++)
    {
        for(int c=0; c< mask_dst.cols; c++)
        {
            const int idx = r * mask_dst.cols + c;

            if (mask_dst.at<unsigned char>(r,c) > 0 && pcl::isFinite( cloud.points[idx] ) && cloud.points[idx].z < param_.chop_z_)
                mask_out[idx] = true;
            else
                mask_out[idx] = false;
        }
    }
    return mask_out;
}

bool
IOL::write_model_to_disk (const std::string &models_dir, const std::string &model_name, bool save_views)
{
    const std::string export_to = models_dir + "/" + model_name;
    io::createDirIfNotExist(export_to);
    pcl::io::savePCDFileBinary(export_to + "/3D_model.pcd", *cloud_normals_oriented_);

    if (save_views) //save recognition data with new poses
    {
        io::createDirIfNotExist(export_to + "/views");

        for(size_t i=0; i < keyframes_used_.size(); i++)
        {
            std::stringstream view_file;
            view_file << export_to << "/views/cloud_" << setfill('0') << setw(8) << i << ".pcd";
            pcl::io::savePCDFileBinary (view_file.str (), *keyframes_used_[i]);
            std::cout << view_file.str() << std::endl;

            std::string path_pose (view_file.str());
            boost::replace_last (path_pose, "cloud_", "pose_");
            boost::replace_last (path_pose, ".pcd", ".txt");
            io::writeMatrixToFile(path_pose, cameras_used_[i]);
            std::cout << path_pose << std::endl;

            std::string path_obj_indices (view_file.str());
            boost::replace_last (path_obj_indices, "cloud_", "object_indices_");
            boost::replace_last (path_obj_indices, ".pcd", ".txt");

            std::ofstream mask_f (path_obj_indices);
            for(const auto &idx : object_indices_clouds_used_[i])
                mask_f << idx << std::endl;
            mask_f.close();
        }
    }
    return true;
}


bool
IOL::save_model (const std::string &models_dir, const std::string &model_name, bool save_individual_views)
{
    size_t num_frames = grph_.size();

    std::vector< pcl::PointCloud<pcl::Normal>::Ptr > normals_used (num_frames);
    keyframes_used_.resize(num_frames);
    cameras_used_.resize(num_frames);
    object_indices_clouds_used_.resize(num_frames);

    // only used keyframes with have object points in them
    size_t kept_keyframes=0;
    for (size_t view_id = 0; view_id < grph_.size(); view_id++)
    {
        if ( createIndicesFromMask<size_t>(grph_[view_id].obj_mask_step_.back()).size() )
        {
            keyframes_used_[ kept_keyframes ] = grph_[view_id].cloud_;
            normals_used [ kept_keyframes ] = grph_[view_id].normal_;
            cameras_used_ [ kept_keyframes ] = grph_[view_id].camera_pose_;
            object_indices_clouds_used_[ kept_keyframes ] = createIndicesFromMask<size_t>( grph_[view_id].obj_mask_step_.back() );
            kept_keyframes++;
        }
    }

    keyframes_used_.resize(kept_keyframes);
    normals_used.resize(kept_keyframes);
    cameras_used_.resize(kept_keyframes);
    object_indices_clouds_used_.resize(kept_keyframes);
    std::vector<std::vector<std::vector<float> > > pt_properties (kept_keyframes);

    if ( kept_keyframes > 0)
    {
        //compute noise weights
        for(size_t i=0; i < kept_keyframes; i++)
        {
            NguyenNoiseModel<PointT>::Parameter nm_param;
            nm_param.use_depth_edges_ = true;
            NguyenNoiseModel<PointT> nm (nm_param);
            nm.setInputCloud(keyframes_used_[i]);
            nm.setInputNormals(normals_used[i]);
            nm.compute();
            pt_properties[i] = nm.getPointProperties();
        }

        pcl::PointCloud<PointT>::Ptr octree_cloud(new pcl::PointCloud<PointT>);
        NMBasedCloudIntegration<PointT> nmIntegration (nm_int_param_);
        nmIntegration.setInputClouds(keyframes_used_);
        nmIntegration.setTransformations(cameras_used_);
        nmIntegration.setInputNormals(normals_used);
        nmIntegration.setIndices( object_indices_clouds_used_ );
        nmIntegration.setPointProperties( pt_properties );
        nmIntegration.compute(octree_cloud);

        pcl::PointCloud<pcl::Normal>::Ptr octree_normals;
        nmIntegration.getOutputNormals(octree_normals);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_with_normals_oriented (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        pcl::concatenateFields(*octree_normals, *octree_cloud, *filtered_with_normals_oriented);

        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBNormal> sor;
        sor.setInputCloud (filtered_with_normals_oriented);
        sor.setMeanK (50);
        sor.setStddevMulThresh (3.0);
        sor.filter (*cloud_normals_oriented_);

        std::cout << "Saving " << kept_keyframes << " keyframes from " << num_frames << " to " << models_dir << std::endl;
        write_model_to_disk(models_dir, model_name, save_individual_views);
    }
    return true;
}

void
IOL::extractPlanePoints(const pcl::PointCloud<PointT>::ConstPtr &cloud,
                             const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
                             std::vector<ClusterNormalsToPlanes::Plane::Ptr> &planes)
{
    ClusterNormalsToPlanes pest(p_param_);
    DataMatrix2D<Eigen::Vector3f>::Ptr kp_cloud( new DataMatrix2D<Eigen::Vector3f>() );
    DataMatrix2D<Eigen::Vector3f>::Ptr kp_normals( new DataMatrix2D<Eigen::Vector3f>() );
    convertCloud(*cloud, *kp_cloud);
    convertNormals(*normals, *kp_normals);

    std::vector<ClusterNormalsToPlanes::Plane::Ptr> all_planes;
    pest.compute(*kp_cloud, *kp_normals, all_planes);
    planes.resize(all_planes.size());
    size_t kept=0;
    for (size_t cluster_id=0; cluster_id<all_planes.size(); cluster_id++)
    {
        float min_z = std::numeric_limits<float>::max();
        for(size_t cluster_pt_id=0; cluster_pt_id<all_planes[cluster_id]->indices.size(); cluster_pt_id++)
        {
            const int id = all_planes[cluster_id]->indices[cluster_pt_id];
            if ( cloud->points[id].z < min_z ) // do not consider points that are further away than a certain threshold
                min_z = cloud->points[id].z;
        }
        if( pcl_isfinite(param_.chop_z_) || min_z < param_.chop_z_)
        {
            planes[kept] = all_planes[cluster_id];
            kept++;
        }
    }
    planes.resize(kept);
}

bool
IOL::merging_planes_reasonable(const modelView::SuperPlane &sp1, const modelView::SuperPlane &sp2) const
{
    float dist = std::abs(PlaneEstimationRANSAC::normalPointDist(sp1.pt, sp1.normal, sp2.pt));
    float dot  = sp1.normal.dot(sp2.normal);
//    std::cout << "dist: " << dist << ", dot: " << dot << std::endl;
    return (dist < 2 * p_param_.inlDist && dot > 0.95);
}

void
IOL::computePlaneProperties(const std::vector<ClusterNormalsToPlanes::Plane::Ptr> &planes,
                                       const std::vector< bool > &object_mask,
                                       const std::vector< bool > &occlusion_mask,
                                       const pcl::PointCloud<PointT>::ConstPtr &cloud,
                                       std::vector<modelView::SuperPlane> &super_planes) const
{
//    planes_not_on_object.resize(planes.size());
    super_planes.resize(planes.size());

//    size_t kept=0;
    for(size_t cluster_id=0; cluster_id<planes.size(); cluster_id++)
    {
        super_planes[cluster_id].pt = planes[cluster_id]->pt;
        super_planes[cluster_id].normal = planes[cluster_id]->normal;
        super_planes[cluster_id].is_plane = planes[cluster_id]->is_plane;
        super_planes[cluster_id].indices = planes[cluster_id]->indices;
        super_planes[cluster_id].visible_indices.resize( planes[cluster_id]->indices.size() );
        super_planes[cluster_id].object_indices.resize( planes[cluster_id]->indices.size() );
        super_planes[cluster_id].within_chop_z_indices.resize( planes[cluster_id]->indices.size() );

        size_t num_obj_pts = 0;
        size_t num_occluded_pts = 0;
        size_t num_plane_pts = 0;

        for (size_t cluster_pt_id=0; cluster_pt_id<planes[cluster_id]->indices.size(); cluster_pt_id++)
        {
            const int id = planes[cluster_id]->indices[cluster_pt_id];
            if ( cloud->points[id].z < param_.chop_z_ )
            {
                super_planes[cluster_id].within_chop_z_indices[ num_plane_pts++ ] = id;

                if ( object_mask[id] )
                     super_planes[cluster_id].object_indices[ num_obj_pts++ ] = id;

                if( !occlusion_mask[id] )
                    super_planes[cluster_id].visible_indices[ num_occluded_pts++ ] = id;

            }
        }
        super_planes[cluster_id].visible_indices.resize( num_occluded_pts );
        super_planes[cluster_id].object_indices.resize( num_obj_pts );
        super_planes[cluster_id].within_chop_z_indices.resize( num_plane_pts );


//        if ( num_plane_pts == 0 ||
//             ( (double)num_obj_pts/num_plane_pts < param_.ratio_cluster_obj_supported_ && (double)num_occluded_pts/num_plane_pts < param_.ratio_cluster_occluded_) )
//        {
//            planes_not_on_object[kept] = planes[cluster_id];
//            kept++;
//        }
    }
//    planes_not_on_object.resize(kept);
}

bool
IOL::learn_object (const pcl::PointCloud<PointT> &cloud, const Eigen::Matrix4f &camera_pose, const std::vector<size_t> &initial_indices)
{
    size_t id = grph_.size();
    std::cout << "Computing indices for cloud " << id << std::endl
              << "===================================" << std::endl;
    grph_.resize(id + 1);
    modelView& view = grph_.back();
    pcl::copyPointCloud(cloud, *(view.cloud_));
    view.id_ = id;
    view.tracking_pose_ = camera_pose; //common::RotTrans2Mat4f(cloud.sensor_orientation_, cloud.sensor_origin_);
    view.tracking_pose_set_ = true;
    view.camera_pose_ = view.tracking_pose_;

    boost::add_vertex(view.id_, gs_);

    pcl::PointCloud<pcl::Normal>::Ptr normals_filtered (new pcl::PointCloud<pcl::Normal>());
    std::vector<ClusterNormalsToPlanes::Plane::Ptr> planes;

    computeNormals<PointT>(view.cloud_, view.normal_, param_.normal_method_);
    extractPlanePoints(view.cloud_, view.normal_, planes);

    octree_.setInputCloud ( view.cloud_ );
    octree_.addPointsFromInputCloud ();

    if (initial_indices.size())   // for first frame use given initial indices and erode them
    {
        std::vector<bool> initial_mask = createMaskFromIndices(initial_indices, view.cloud_->points.size());
        remove_nan_points(*view.cloud_, initial_mask);
        view.obj_mask_step_.push_back( initial_mask );
        view.is_pre_labelled_ = true;

        // remove nan values and points further away than chop_z_ parameter
        std::vector<size_t> initial_indices_wo_nan (initial_indices.size());
        size_t kept=0;
        for(size_t idx=0; idx<initial_indices.size(); idx++)
        {
            if ( pcl::isFinite( view.cloud_->points[initial_indices[idx]]) && view.cloud_->points[initial_indices[idx]].z < param_.chop_z_)
            {
                initial_indices_wo_nan[kept] = initial_indices[idx];
                kept++;
            }
        }
        initial_indices_wo_nan.resize(kept);

        //erode mask
        pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>());
        boost::shared_ptr <std::vector<int> > ObjectIndicesPtr (new std::vector<int>());
        boost::shared_ptr <const std::vector<int> > FilteredObjectIndicesPtr (new std::vector<int>());

        *ObjectIndicesPtr = convertVecSizet2VecInt(initial_indices_wo_nan);

        pcl::copyPointCloud(*view.cloud_, initial_indices_wo_nan, *cloud_filtered);
        pcl::StatisticalOutlierRemoval<PointT> sor(true);
        sor.setInputCloud (view.cloud_);
        sor.setIndices(ObjectIndicesPtr);
        sor.setMeanK (sor_params_.meanK_);
        sor.setStddevMulThresh (sor_params_.std_mul_);
        sor.filter (*cloud_filtered);
        FilteredObjectIndicesPtr = sor.getRemovedIndices();

        const std::vector<bool> obj_mask_initial = createMaskFromIndices(initial_indices_wo_nan, view.cloud_->points.size());
        const std::vector<bool> outlier_mask = createMaskFromIndices(*FilteredObjectIndicesPtr, view.cloud_->points.size());
        const std::vector<bool> obj_mask_wo_outlier = binary_operation(obj_mask_initial, outlier_mask, BINARY_OPERATOR::AND_N);

        view.obj_mask_step_.push_back( obj_mask_wo_outlier);

        std::vector<bool> obj_mask_eroded = erodeIndices(obj_mask_wo_outlier, *view.cloud_);
        view.obj_mask_step_.push_back( obj_mask_eroded );
        computePlaneProperties(planes, view.obj_mask_step_[0],
                               std::vector<bool>(view.cloud_->points.size(), false),
                               view.cloud_, view.planes_);
    }
    else
    {
        std::vector<bool> is_occluded;
        for (size_t view_id = 0; view_id < grph_.size(); view_id++)
        {
            if( view.id_ != grph_[view_id].id_)
            {
                pcl::PointCloud<PointT>::Ptr new_search_pts, new_search_pts_aligned;
                new_search_pts.reset(new pcl::PointCloud<PointT>());
                new_search_pts_aligned.reset(new pcl::PointCloud<PointT>());
                pcl::copyPointCloud(*grph_[view_id].cloud_, grph_[view_id].obj_mask_step_.back(), *new_search_pts);
                const Eigen::Matrix4f tf = view.camera_pose_.inverse() * grph_[view_id].camera_pose_;
                pcl::transformPointCloud(*new_search_pts, *new_search_pts_aligned, tf);

                pcl::IterativeClosestPoint<PointT, PointT> icp;
                icp.setInputSource(new_search_pts_aligned);
                icp.setInputTarget(view.cloud_);
                icp.setMaxCorrespondenceDistance (0.02f);
                pcl::PointCloud<PointT>::Ptr icp_aligned_cloud (new pcl::PointCloud<PointT>());
                icp.align(*icp_aligned_cloud, Eigen::Matrix4f::Identity());
                *view.transferred_cluster_ += *icp_aligned_cloud;

                if (grph_[view_id].is_pre_labelled_)
                {
                    std::vector<bool> is_occluded_tmp = computeOccludedPoints(*grph_[view_id].cloud_,
                                                                                                   *view.cloud_,
                                                                                                   tf.inverse(),
                                                                                                        525.f, 0.01f, false);
                    if( is_occluded.size() == is_occluded_tmp.size())
                    {
                        is_occluded = binary_operation(is_occluded, is_occluded_tmp, BINARY_OPERATOR::AND); // is this correct?
                    }
                    else
                    {
                        is_occluded = is_occluded_tmp;
                    }
                }
            }
        }

        std::vector<bool> obj_mask_nn_search (view.cloud_->points.size(), false);
        nnSearch(*view.transferred_cluster_, octree_, obj_mask_nn_search);
        view.obj_mask_step_.push_back( obj_mask_nn_search);

        computePlaneProperties(planes, obj_mask_nn_search, is_occluded,
                               view.cloud_, view.planes_);
    }
    std::vector<bool> pixel_is_object = view.obj_mask_step_.back();

    // filter cloud based on planes not on object and not occluded in first frame
    std::vector<bool> pixel_is_neglected (view.cloud_->points.size(), false);
    for (size_t p_id=0; p_id<view.planes_.size(); p_id++)
    {
        for (size_t view_id = 0; view_id < grph_.size(); view_id++)
        {
            if( view.id_ != grph_[view_id].id_)
            {
                for (size_t p2_id=0; p2_id<grph_[view_id].planes_.size(); p2_id++)
                {
//                    std::cout << "Checking cluster new " << p_id << " and cluster old " <<
//                                 p2_id << " from view " << view_id << std::endl;
//                    std::cout << merging_planes_reasonable(view.planes_[p_id], grph_[view_id].planes_[p2_id]) << std::endl;

                    // if the planes can be merged (based on normals and distance), then filter new plane if old one has been filtered
                    if (grph_[view_id].planes_[p2_id].is_filtered && merging_planes_reasonable(view.planes_[p_id], grph_[view_id].planes_[p2_id]) && !plane_has_object(view.planes_[p_id]))
                        view.planes_[p_id].is_filtered = true;
                }
            }
        }
        if ( plane_is_filtered( view.planes_[p_id] ) )
            view.planes_[p_id].is_filtered = true;

        if ( view.planes_[p_id].is_filtered )
        {
            for (size_t c_pt_id=0; c_pt_id < view.planes_[p_id].indices.size(); c_pt_id++)
                pixel_is_neglected [ view.planes_[p_id].indices[ c_pt_id ] ] = true;
        }
    }
    for(size_t pt=0; pt<view.cloud_->points.size(); pt++)
    {
        if (view.cloud_->points[pt].z > param_.chop_z_)
            pixel_is_neglected[pt] = true;
    }

    view.scene_points_ = createIndicesFromMask<size_t>(pixel_is_neglected, true);
    view.obj_mask_step_.push_back( binary_operation(pixel_is_object, pixel_is_neglected, BINARY_OPERATOR::AND_N) );
    pcl::copyPointCloud(*view.normal_, view.scene_points_, *normals_filtered);

    std::vector<bool> obj_mask_enforced_by_supervoxel_consistency;
    updatePointNormalsFromSuperVoxels(view.cloud_,
                                      view.normal_,
                                      view.obj_mask_step_.back(),
                                      obj_mask_enforced_by_supervoxel_consistency,
                                      view.supervoxel_cloud_,
                                      view.supervoxel_cloud_organized_);
    view.obj_mask_step_.push_back( obj_mask_enforced_by_supervoxel_consistency );

    std::vector<bool> obj_mask_grown_by_smooth_surface = extractEuclideanClustersSmooth(view.cloud_,
                                                                                           *view.normal_,
                                                                                           obj_mask_enforced_by_supervoxel_consistency,
                                                                                           pixel_is_neglected);
    view.obj_mask_step_.push_back(obj_mask_grown_by_smooth_surface);

    std::vector<bool> obj_mask_eroded = erodeIndices(obj_mask_grown_by_smooth_surface, *view.cloud_);
    remove_nan_points(*view.cloud_, obj_mask_eroded);
    view.obj_mask_step_.push_back( obj_mask_eroded );

    for( size_t step_id = 0; step_id<view.obj_mask_step_.size(); step_id++)
    {
        std::cout << "step " << step_id << ": " << createIndicesFromMask<size_t>(view.obj_mask_step_[step_id]).size() << " points." << std::endl;
    }

    if( view.is_pre_labelled_ && createIndicesFromMask<size_t>(view.obj_mask_step_.back()).size() < param_.min_points_for_transferring_)
    {
        view.obj_mask_step_.back() = view.obj_mask_step_[0];
        std::cout << "After postprocessing the initial frame not enough points are left. Therefore taking the original provided indices." << std::endl;
    }
//    visualize();
    return true;
}

void
IOL::printParams(std::ostream &ostr) const
{
    ostr << "Started incremental object learning with parameters: " << std::endl
         << "===================================================" << std::endl
         << "radius: " << param_.radius_ << std::endl
         << "eps_angle: " << param_.eps_angle_ << std::endl
         << "dist_threshold_growing_: " << param_.dist_threshold_growing_ << std::endl
         << "voxel resolution: " << param_.voxel_resolution_ << std::endl
         << "seed resolution: " << param_.seed_resolution_ << std::endl
         << "ratio_supervoxel: " << param_.ratio_supervoxel_ << std::endl
         << "max z distance: " << param_.chop_z_ << std::endl
         << "do_erosion: " << param_.do_erosion_ << std::endl
         << "transferring object indices from latest frame only: " << param_.transfer_indices_from_latest_frame_only_ << std::endl
         << "min_points_for_transferring_: " << param_.min_points_for_transferring_ << std::endl
         << "normal_method_: " << param_.normal_method_ << std::endl
         << "apply minimimum spanning tree: " << param_.do_mst_refinement_ << std::endl
         << "ratio_cluster_obj_supported_: " << param_.ratio_cluster_obj_supported_ << std::endl
         << "ratio_cluster_occluded_: " << param_.ratio_cluster_occluded_ << std::endl
         << "smooth_clustering_param_inlDist: " << p_param_.inlDist << std::endl
         << "smooth_clustering_param_inlDistSmooth: " << p_param_.inlDistSmooth << std::endl
         << "smooth_clustering_param_least_squares_refinement: " << p_param_.least_squares_refinement << std::endl
         << "smooth_clustering_param_minPoints: " << p_param_.minPoints << std::endl
         << "smooth_clustering_param_minPointsSmooth: " << p_param_.minPointsSmooth << std::endl
         << "smooth_clustering_param_smooth_clustering: " << p_param_.smooth_clustering << std::endl
         << "smooth_clustering_param_thrAngle: " << p_param_.thrAngle << std::endl
         << "smooth_clustering_param_thrAngleSmooth: " << p_param_.thrAngleSmooth << std::endl
         << "statistical_outlier_removal_meanK_: " << sor_params_.meanK_ << std::endl
         << "statistical_outlier_removal_std_mul_: " << sor_params_.std_mul_ << std::endl
         << "===================================================" << std::endl << std::endl;
}
}
}
