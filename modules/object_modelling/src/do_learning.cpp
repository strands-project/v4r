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

#include "v4r/object_modelling/do_learning.h"

#include <stdlib.h>
#include <thread>
#include <iostream>

#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#include <v4r/keypoints/impl/convertCloud.hpp>
#include <v4r/keypoints/impl/convertNormals.hpp>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/features/sift_local_estimator.h>
#include <v4r/common/fast_icp_with_gc.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/noise_models.h>
#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <v4r/common/occlusion_reasoning.h>

#include <boost/graph/kruskal_min_spanning_tree.hpp>

#define USE_SIFT_GPU

#ifndef USE_SIFT_GPU
#include <v4r/features/opencv_sift_local_estimator.h>
#endif

namespace v4r
{
namespace object_modelling
{

float
DOL::calcEdgeWeightAndRefineTf (const pcl::PointCloud<PointT>::ConstPtr &cloud_src,
                                const pcl::PointCloud<PointT>::ConstPtr &cloud_dst,
                                Eigen::Matrix4f &refined_transform,
                                const Eigen::Matrix4f &transform)
{
    pcl::PointCloud<PointT>::Ptr cloud_src_wo_nan ( new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr cloud_dst_wo_nan ( new pcl::PointCloud<PointT>());

    pcl::PassThrough<PointT> pass;
    pass.setFilterLimits (0.f, 5.f);
    pass.setFilterFieldName ("z");
    pass.setInputCloud (cloud_src);
    pass.setKeepOrganized (true);
    pass.filter (*cloud_src_wo_nan);

    pcl::PassThrough<PointT> pass2;
    pass2.setFilterLimits (0.f, 5.f);
    pass2.setFilterFieldName ("z");
    pass2.setInputCloud (cloud_dst);
    pass2.setKeepOrganized (true);
    pass2.filter (*cloud_dst_wo_nan);

    float w_after_icp_ = std::numeric_limits<float>::max ();
    const float best_overlap_ = 0.75f;

    v4r::common::FastIterativeClosestPointWithGC<pcl::PointXYZRGB> icp;
    icp.setMaxCorrespondenceDistance ( 0.02f );
    icp.setInputSource ( cloud_src_wo_nan );
    icp.setInputTarget ( cloud_dst_wo_nan );
    icp.setUseNormals (true);
    icp.useStandardCG (true);
    icp.setNoCG(true);
    icp.setOverlapPercentage (best_overlap_);
    icp.setKeepMaxHypotheses (5);
    icp.setMaximumIterations (10);
    icp.align (transform);
    w_after_icp_ = icp.getFinalTransformation ( refined_transform );

    if ( w_after_icp_ < 0 || !pcl_isfinite ( w_after_icp_ ) )
        w_after_icp_ = std::numeric_limits<float>::max ();
    else
        w_after_icp_ = best_overlap_ - w_after_icp_;

    //    transform = icp_trans; // refined transformation
    return w_after_icp_;
}

bool
DOL::calcSiftFeatures (const pcl::PointCloud<PointT>::Ptr &cloud_src,
                       pcl::PointCloud<PointT>::Ptr &sift_keypoints,
                       std::vector< size_t > &sift_keypoint_indices,
                       pcl::PointCloud<FeatureT>::Ptr &sift_signatures,
                       std::vector<float> &sift_keypoint_scales)
{
    pcl::PointIndices sift_keypoint_pcl_indices;

#ifdef USE_SIFT_GPU
    boost::shared_ptr < v4r::SIFTLocalEstimation<PointT, FeatureT> > estimator;
    estimator.reset (new v4r::SIFTLocalEstimation<PointT, FeatureT>(sift_));

    bool ret = estimator->estimate (cloud_src, sift_keypoints, sift_signatures, sift_keypoint_scales);
    estimator->getKeypointIndices( sift_keypoint_pcl_indices );
#else
    (void)sift_keypoint_scales; //silences compiler warning of unused variable
    boost::shared_ptr < v4r::OpenCVSIFTLocalEstimation<PointT, pcl::Histogram<128> > > estimator;
    estimator.reset (new v4r::OpenCVSIFTLocalEstimation<PointT, pcl::Histogram<128> >);

    pcl::PointCloud<PointT>::Ptr processed_foo (new pcl::PointCloud<PointT>());

    bool ret = estimator->estimate (cloud_src, processed_foo, sift_keypoints, sift_signatures);
    estimator->getKeypointIndices( sift_keypoint_pcl_indices );

    sift_keypoint_indices = v4r::common::convertPCLIndices2VecSizet(sift_keypoint_pcl_indices);
#endif
    return ret;
}

void
DOL::estimateViewTransformationBySIFT(const pcl::PointCloud<PointT> &src_cloud,
                                      const pcl::PointCloud<PointT> &dst_cloud,
                                      const std::vector<size_t> &src_sift_keypoint_indices,
                                      const std::vector<size_t> &dst_sift_keypoint_indices,
                                      const pcl::PointCloud<FeatureT> &src_sift_signatures,
                                      boost::shared_ptr< flann::Index<DistT> > &dst_flann_index,
                                      std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transformations,
                                      bool use_gc )
{
    const int K = 1;
    flann::Matrix<int> indices = flann::Matrix<int> ( new int[K], 1, K );
    flann::Matrix<float> distances = flann::Matrix<float> ( new float[K], 1, K );

    boost::shared_ptr< pcl::PointCloud<PointT> > pSiftKeypointsSrc (new pcl::PointCloud<PointT>);
    boost::shared_ptr< pcl::PointCloud<PointT> > pSiftKeypointsDst (new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(src_cloud, src_sift_keypoint_indices, *pSiftKeypointsSrc );
    pcl::copyPointCloud(dst_cloud, dst_sift_keypoint_indices, *pSiftKeypointsDst);

    pcl::CorrespondencesPtr temp_correspondences ( new pcl::Correspondences );
    temp_correspondences->resize(pSiftKeypointsSrc->size ());

    for ( size_t keypointId = 0; keypointId < pSiftKeypointsSrc->points.size (); keypointId++ )
    {
        FeatureT searchFeature = src_sift_signatures[ keypointId ];
        int size_feat = sizeof ( searchFeature.histogram ) / sizeof ( float );
        v4r::common::nearestKSearch ( dst_flann_index, searchFeature.histogram, size_feat, K, indices, distances );

        pcl::Correspondence corr;
        corr.distance = distances[0][0];
        corr.index_query = keypointId;
        corr.index_match = indices[0][0];
        temp_correspondences->at(keypointId) = corr;
    }

    if(!use_gc)
    {
        pcl::registration::CorrespondenceRejectorSampleConsensus<PointT>::Ptr rej;
        rej.reset (new pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ());
        pcl::CorrespondencesPtr after_rej_correspondences (new pcl::Correspondences ());

        rej->setMaximumIterations (50000);
        rej->setInlierThreshold (0.02);
        rej->setInputTarget (pSiftKeypointsDst);
        rej->setInputSource (pSiftKeypointsSrc);
        rej->setInputCorrespondences (temp_correspondences);
        rej->getCorrespondences (*after_rej_correspondences);

        Eigen::Matrix4f refined_pose;
        transformations.push_back( rej->getBestTransformation () );
        pcl::registration::TransformationEstimationSVD<PointT, PointT> t_est;
        t_est.estimateRigidTransformation (*pSiftKeypointsSrc, *pSiftKeypointsDst, *after_rej_correspondences, refined_pose);
        transformations.back() = refined_pose;
    }
    else
    {
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > new_transforms;
        pcl::GeometricConsistencyGrouping<pcl::PointXYZRGB, pcl::PointXYZRGB> gcg_alg;

        gcg_alg.setGCThreshold (15);
        gcg_alg.setGCSize (0.01);
        gcg_alg.setInputCloud(pSiftKeypointsSrc);
        gcg_alg.setSceneCloud(pSiftKeypointsDst);
        gcg_alg.setModelSceneCorrespondences(temp_correspondences);

        std::vector<pcl::Correspondences> clustered_corrs;
        gcg_alg.recognize(new_transforms, clustered_corrs);
        transformations.insert(transformations.end(), new_transforms.begin(), new_transforms.end());
    }
}


std::vector<bool>
DOL::extractEuclideanClustersSmooth (
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
DOL::updatePointNormalsFromSuperVoxels(const pcl::PointCloud<PointT>::Ptr & cloud,
                                            pcl::PointCloud<pcl::Normal>::Ptr & normals,
                                            const std::vector<bool> &obj_mask,
                                            std::vector<bool> &obj_mask_out,
                                            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &supervoxel_cloud)
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
            if( label_count[label]  > param_.ratio_ * tot_pts_in_supervoxel)
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
DOL::nnSearch(const pcl::PointCloud<PointT> &object_points, const pcl::PointCloud<PointT>::ConstPtr &search_cloud,  std::vector<bool> &obj_mask)
{
    pcl::octree::OctreePointCloudSearch<PointT> octree(0.005f);
    octree.setInputCloud ( search_cloud );
    octree.addPointsFromInputCloud ();
    nnSearch(object_points, octree, obj_mask);
}

void
DOL::nnSearch(const pcl::PointCloud<PointT> &object_points, pcl::octree::OctreePointCloudSearch<PointT> &octree,  std::vector<bool> &obj_mask)
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
            {
                obj_mask[ pointIdxRadiusSearch[ nn_id ] ] = true;
            }
        }
    }
}

std::vector<bool>
DOL::erodeIndices(const std::vector< bool > &obj_mask, const pcl::PointCloud<PointT> & cloud)
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
DOL::save_model (const std::string &models_dir, const std::string &recognition_structure_dir, const std::string &model_name)
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
        if ( v4r::common::createIndicesFromMask(grph_[view_id].obj_mask_step_.back()).size() )
        {
            keyframes_used[ kept_keyframes ] = grph_[view_id].cloud_;
            normals_used [ kept_keyframes ] = grph_[view_id].normal_;
            cameras_used [ kept_keyframes ] = grph_[view_id].camera_pose_;
            indices_used[ kept_keyframes ] = v4r::common::createIndicesFromMask( grph_[view_id].obj_mask_step_.back() );

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

        std::stringstream export_to_rs;
        export_to_rs << recognition_structure_dir << "/" << model_name << "/";
        std::string export_to = export_to_rs.str();

        v4r::io::createDirIfNotExist(recognition_structure_dir);
        v4r::io::createDirIfNotExist(models_dir);
        v4r::io::createDirIfNotExist(export_to);

        std::cout << "Saving " << kept_keyframes << " keyframes from " << num_frames << "." << std::endl;

        //save recognition data with new poses
        for(size_t i=0; i < kept_keyframes; i++)
        {
            std::stringstream view_file;
            view_file << export_to << "/cloud_" << setfill('0') << setw(8) << i << ".pcd";
            pcl::io::savePCDFileBinary (view_file.str (), *(keyframes_used[i]));
            std::cout << view_file.str() << std::endl;

            std::string path_pose (view_file.str());
            boost::replace_last (path_pose, "cloud", "pose");
            boost::replace_last (path_pose, ".pcd", ".txt");
            v4r::io::writeMatrixToFile(path_pose, cameras_used[i]);
            std::cout << path_pose << std::endl;

            std::string path_obj_indices (view_file.str());
            boost::replace_last (path_obj_indices, "cloud", "object_indices");
            pcl::io::savePCDFileBinary (path_obj_indices, object_indices_clouds[i]);
            std::cout << path_obj_indices << std::endl;
        }

        std::stringstream path_model;
        path_model << models_dir << "/" << model_name;

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_with_normals_oriented (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        pcl::concatenateFields(*octree_normals, *octree_cloud, *filtered_with_normals_oriented);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_normals_oriented (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBNormal> sor;
        sor.setInputCloud (filtered_with_normals_oriented);
        sor.setMeanK (50);
        sor.setStddevMulThresh (3.0);
        sor.filter (*cloud_normals_oriented);
        pcl::io::savePCDFileBinary(path_model.str(), *cloud_normals_oriented);
    }
    return true;
}

void
DOL::extractPlanePoints(const pcl::PointCloud<PointT>::ConstPtr &cloud,
                             const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
                             std::vector<v4r::ClusterNormalsToPlanes::Plane::Ptr> &planes)
{
    v4r::ClusterNormalsToPlanes pest(p_param_);
    v4r::DataMatrix2D<Eigen::Vector3f>::Ptr kp_cloud( new v4r::DataMatrix2D<Eigen::Vector3f>() );
    v4r::DataMatrix2D<Eigen::Vector3f>::Ptr kp_normals( new v4r::DataMatrix2D<Eigen::Vector3f>() );
    v4r::convertCloud(*cloud, *kp_cloud);
    v4r::convertNormals(*normals, *kp_normals);
    pest.compute(*kp_cloud, *kp_normals, planes);
}

void
DOL::getPlanesNotSupportedByObjectMask(const std::vector<v4r::ClusterNormalsToPlanes::Plane::Ptr> &planes,
                                       const std::vector< bool > &object_mask,
                                       const std::vector< bool > &occlusion_mask,
                                       const pcl::PointCloud<PointT>::ConstPtr &cloud,
                                       std::vector< std::vector<int> > &planes_not_on_object,
                                       float ratio,
                                       float ratio_occ) const
{
    planes_not_on_object.resize(planes.size());

    size_t kept=0;
    for(size_t cluster_id=0; cluster_id<planes.size(); cluster_id++)
    {
        size_t num_obj_pts = 0;
        size_t num_occluded_pts = 0;
        size_t num_plane_pts = 0;

        if ( planes[cluster_id]->is_plane || !param_.filter_planes_only_ )
        {
            for (size_t cluster_pt_id=0; cluster_pt_id<planes[cluster_id]->indices.size(); cluster_pt_id++)
            {
                const int id = planes[cluster_id]->indices[cluster_pt_id];

                if ( cloud->points[id].z > param_.chop_z_ ) // do not consider points that are further away than a certain threshold
                    continue;

                if ( object_mask[id] )
                {
                    num_obj_pts++;
                }

                if( occlusion_mask[id] )
                    num_occluded_pts++;

                num_plane_pts++;
            }

            if ( num_plane_pts == 0 || (num_obj_pts < ratio * num_plane_pts && num_occluded_pts < ratio_occ * num_plane_pts) )
            {
                planes_not_on_object[kept] = planes[cluster_id]->indices;
                kept++;
            }
        }
    }
    planes_not_on_object.resize(kept);
}

void
DOL::computeAbsolutePosesRecursive (const Graph & grph,
                              const Vertex start,
                              const Eigen::Matrix4f &accum,
                              std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & absolute_poses,
                              std::vector<bool> &hop_list)
{
    boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, gs_);
    boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
    for (boost::tie (ei, ei_end) = boost::out_edges (start, grph); ei != ei_end; ++ei)
    {
        Vertex targ = boost::target (*ei, grph);
        size_t target_id = boost::target (*ei, grph);

        if(hop_list[target_id])
           continue;

        hop_list[target_id] = true;
        CamConnect my_e = weightmap[*ei];
        Eigen::Matrix4f intern_accum;
        Eigen::Matrix4f trans = my_e.transformation_;
        if( my_e.target_id_ != target_id)
        {
            Eigen::Matrix4f trans_inv;
            trans_inv = trans.inverse();
            trans = trans_inv;
        }
        intern_accum = accum * trans;
        absolute_poses[target_id] = intern_accum;
        computeAbsolutePosesRecursive (grph, targ, intern_accum, absolute_poses, hop_list);
    }
}

void
DOL::computeAbsolutePoses (const Graph & grph,
                     std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & absolute_poses)
{
  size_t num_frames = boost::num_vertices(grph);
  absolute_poses.resize( num_frames );
  std::vector<bool> hop_list (num_frames, false);
  Vertex source_view = 0;
  hop_list[0] = true;
  Eigen::Matrix4f accum = grph_[0].tracking_pose_;
  absolute_poses[0] = accum;
  computeAbsolutePosesRecursive (grph, source_view, accum, absolute_poses, hop_list);
}

std::vector<bool>
DOL::createMaskFromVecIndices( const std::vector< std::vector<int> > &v_indices,
                               size_t image_size)
{
    std::vector<bool> mask;

    if ( mask.size() != image_size )
        mask = std::vector<bool>( image_size, false );

    for(size_t i=0; i<v_indices.size(); i++)
    {
        std::vector<bool> mask_tmp = v4r::common::createMaskFromIndices(v_indices[i], image_size);

        if(mask.size())
            mask = logical_operation(mask, mask_tmp, MASK_OPERATOR::OR);
        else
            mask = mask_tmp;
    }

    return mask;
}

std::vector<bool>
DOL::logical_operation(const std::vector<bool> &mask1, const std::vector<bool> &mask2, int operation)
{
    assert(mask1.size() == mask2.size());

    std::vector<bool> output_mask;
    output_mask.resize(mask1.size());

    for(size_t i=0; i<mask1.size(); i++)
    {
        if(operation == MASK_OPERATOR::AND)
        {
            output_mask[i] = mask1[i] && mask2[i];
        }
        else
        if (operation == MASK_OPERATOR::AND_N)
        {
            output_mask[i] = mask1[i] && !mask2[i];
        }
        else
        if (operation == MASK_OPERATOR::OR)
        {
            output_mask[i] = mask1[i] || mask2[i];
        }
        else
        if (operation == MASK_OPERATOR::XOR)
        {
            output_mask[i] = (mask1[i] && !mask2[i]) || (!mask1[i] && mask2[i]);
        }
    }
    return output_mask;
}

bool
DOL::learn_object (const pcl::PointCloud<PointT> &cloud, const Eigen::Matrix4f &camera_pose, const std::vector<size_t> &initial_indices)
{
    size_t id = grph_.size();
    std::cout << "Computing indices for cloud " << id << std::endl
              << "===================================" << std::endl;
    grph_.resize(id + 1);
    modelView& view = grph_.back();
    pcl::copyPointCloud(cloud, *(view.cloud_));
    view.id_ = id;
    view.tracking_pose_ = camera_pose; //v4r::common::RotTrans2Mat4f(cloud.sensor_orientation_, cloud.sensor_origin_);
    view.tracking_pose_set_ = true;
    view.camera_pose_ = view.tracking_pose_;

    boost::add_vertex(view.id_, gs_);

    pcl::PointCloud<pcl::Normal>::Ptr normals_filtered (new pcl::PointCloud<pcl::Normal>());
    std::vector<v4r::ClusterNormalsToPlanes::Plane::Ptr> planes;
    std::vector< std::vector<int> > planes_not_on_obj;
    std::vector<bool> pixel_is_neglected;

    v4r::common::computeNormals(view.cloud_, view.normal_, param_.normal_method_);
    extractPlanePoints(view.cloud_, view.normal_, planes);

    octree_.setInputCloud ( view.cloud_ );
    octree_.addPointsFromInputCloud ();

    boost::shared_ptr<flann::Index<DistT> > flann_index;

    if ( param_.do_sift_based_camera_pose_estimation_ )
    {
        pcl::PointCloud<PointT>::Ptr sift_keypoints (new pcl::PointCloud<PointT>());
        std::vector<float> sift_keypoint_scales;
        try
        {
            calcSiftFeatures( view.cloud_, sift_keypoints, view.sift_keypoint_indices_, view.sift_signatures_, sift_keypoint_scales);
            v4r::common::convertToFLANN<FeatureT, DistT>(view.sift_signatures_, flann_index );
        }
        catch (int e)
        {
            param_.do_sift_based_camera_pose_estimation_ = false;
            std::cerr << "Something is wrong with the SIFT based camera pose estimation. Turning it off and using the given camera poses only." << std::endl;
        }
    }

    if (initial_indices.size())   // for first frame use given initial indices and erode them
    {
        view.obj_mask_step_.push_back(v4r::common::createMaskFromIndices(initial_indices, view.cloud_->points.size()));
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

        *ObjectIndicesPtr = v4r::common::convertVecSizet2VecInt(initial_indices_wo_nan);

        pcl::copyPointCloud(*view.cloud_, initial_indices_wo_nan, *cloud_filtered);
        pcl::StatisticalOutlierRemoval<PointT> sor(true);
        sor.setInputCloud (view.cloud_);
        sor.setIndices(ObjectIndicesPtr);
        sor.setMeanK (sor_params_.meanK_);
        sor.setStddevMulThresh (sor_params_.std_mul_);
        sor.filter (*cloud_filtered);
        FilteredObjectIndicesPtr = sor.getRemovedIndices();

        const std::vector<bool> obj_mask_initial = v4r::common::createMaskFromIndices(initial_indices_wo_nan, view.cloud_->points.size());
        const std::vector<bool> outlier_mask = v4r::common::createMaskFromIndices(*FilteredObjectIndicesPtr, view.cloud_->points.size());
        const std::vector<bool> obj_mask_wo_outlier = logical_operation(obj_mask_initial, outlier_mask, MASK_OPERATOR::AND_N);

        view.obj_mask_step_.push_back( obj_mask_wo_outlier);

        std::vector<bool> obj_mask_eroded = erodeIndices(obj_mask_wo_outlier, *view.cloud_);
        view.obj_mask_step_.push_back( obj_mask_eroded );
        getPlanesNotSupportedByObjectMask(planes,
                                          view.obj_mask_step_[0],
                                          std::vector<bool>(view.cloud_->points.size(), false),
                                          view.cloud_,
                                          planes_not_on_obj);
    }
    else
    {
        for (size_t view_id = 0; view_id < grph_.size(); view_id++)
        {
            if( view.id_ == grph_[view_id].id_)
                continue;

            std::vector<CamConnect> transforms;
            CamConnect edge;
            edge.model_name_ = "camera_tracking";
            edge.source_id_ = view.id_;
            edge.target_id_ = grph_[view_id].id_;
            edge.transformation_ = view.tracking_pose_.inverse() * grph_[view_id].tracking_pose_ ;
            transforms.push_back( edge );

            if ( param_.do_sift_based_camera_pose_estimation_ )
            {
                try
                {
                    edge.model_name_ = "sift_background_matching";
                    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > sift_transforms;
                    estimateViewTransformationBySIFT( *grph_[view_id].cloud_, *view.cloud_,
                                                      grph_[view_id].sift_keypoint_indices_, view.sift_keypoint_indices_,
                                                      *grph_[view_id].sift_signatures_, flann_index, sift_transforms);
                    for(size_t sift_tf_id = 0; sift_tf_id < sift_transforms.size(); sift_tf_id++)
                    {
                        edge.transformation_ = sift_transforms[sift_tf_id];
                        transforms.push_back(edge);
                    }
                }
                catch (int e)
                {
                    param_.do_sift_based_camera_pose_estimation_ = false;
                    std::cerr << "Something is wrong with the SIFT based camera pose estimation. Turning it off and using the given camera poses only." << std::endl;
                }
            }

            size_t best_transform_id = 0;
            float lowest_edge_weight = std::numeric_limits<float>::max();
            for ( size_t trans_id = 0; trans_id < transforms.size(); trans_id++ )
            {
                try
                {
                    Eigen::Matrix4f icp_refined_trans;
                    transforms[ trans_id ].edge_weight = calcEdgeWeightAndRefineTf( grph_[view_id].cloud_, view.cloud_, icp_refined_trans, transforms[ trans_id ].transformation_);
                    transforms[ trans_id ].transformation_ = icp_refined_trans,
                    std::cout << "Edge weight is " << transforms[ trans_id ].edge_weight << " for edge connecting vertex " <<
                                 transforms[ trans_id ].source_id_ << " and " << transforms[ trans_id ].target_id_ << " by " <<
                                 transforms[ trans_id ].model_name_ << std::endl;

                    if(transforms[ trans_id ].edge_weight < lowest_edge_weight)
                    {
                        lowest_edge_weight = transforms[ trans_id ].edge_weight;
                        best_transform_id = trans_id;
                    }
                }
                catch (int e)
                {
                    transforms[ trans_id ].edge_weight = std::numeric_limits<float>::max();
                    param_.do_sift_based_camera_pose_estimation_ = false;
                    std::cerr << "Something is wrong with the SIFT based camera pose estimation. Turning it off and using the given camera poses only." << std::endl;
                    break;
                }
            }
            boost::add_edge (transforms[best_transform_id].source_id_, transforms[best_transform_id].target_id_, transforms[best_transform_id], gs_);
        }

        if(param_.do_mst_refinement_)
        {
            boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, gs_);
            std::vector < Edge > spanning_tree;
            boost::kruskal_minimum_spanning_tree(gs_, std::back_inserter(spanning_tree));

            Graph grph_mst;
            std::cout << "Print the edges in the MST:" << std::endl;
            for (std::vector < Edge >::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei)
            {
                CamConnect my_e = weightmap[*ei];
                std::cout << "[" << source(*ei, gs_) << "->" << target(*ei, gs_) << "] with weight " << my_e.edge_weight << " by " << my_e.model_name_ << std::endl;
                boost::add_edge(source(*ei, gs_), target(*ei, gs_), weightmap[*ei], grph_mst);
            }

            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > absolute_poses;
            computeAbsolutePoses(grph_mst, absolute_poses);

            for(size_t view_id=0; view_id<absolute_poses.size(); view_id++)
            {
                grph_[ view_id ].camera_pose_ = absolute_poses [ view_id ];
            }
        }

        std::vector<bool> is_occluded;
        for (size_t view_id = 0; view_id < grph_.size(); view_id++)
        {
            if( view.id_ != grph_[view_id].id_)
            {
                pcl::PointCloud<PointT> new_search_pts, new_search_pts_aligned;
                pcl::copyPointCloud(*grph_[view_id].cloud_, grph_[view_id].obj_mask_step_.back(), new_search_pts);
                const Eigen::Matrix4f tf = view.camera_pose_.inverse() * grph_[view_id].camera_pose_;
                pcl::transformPointCloud(new_search_pts, new_search_pts_aligned, tf);
                *view.transferred_cluster_ += new_search_pts_aligned;

                if (grph_[view_id].is_pre_labelled_)
                {
                    std::vector<bool> is_occluded_tmp = v4r::occlusion_reasoning::computeOccludedPoints(*grph_[view_id].cloud_,
                                                                                                   *view.cloud_,
                                                                                                   tf.inverse(),
                                                                                                        525.f, 0.01f, false);
                    if( is_occluded.size() == is_occluded_tmp.size())
                    {
                        is_occluded = logical_operation(is_occluded, is_occluded_tmp, MASK_OPERATOR::AND); // is this correct?
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

        getPlanesNotSupportedByObjectMask(planes,
                                          obj_mask_nn_search,
                                          is_occluded,
                                          view.cloud_,
                                          planes_not_on_obj);
    }

    std::vector<bool> pixel_is_object = view.obj_mask_step_.back();
    pixel_is_neglected = v4r::common::createMaskFromVecIndices(planes_not_on_obj, view.cloud_->points.size());
    view.scene_points_ = v4r::common::createIndicesFromMask(pixel_is_neglected, true);
    view.obj_mask_step_.push_back( logical_operation(pixel_is_object, pixel_is_neglected, MASK_OPERATOR::AND_N) );
    pcl::copyPointCloud(*view.normal_, view.scene_points_, *normals_filtered);

    //#define DEBUG_SEGMENTATION
#ifdef DEBUG_SEGMENTATION
    {
        pcl::visualization::PCLVisualizer vis("segmented cloud");
        for(size_t cluster_id=0; cluster_id<planes.size(); cluster_id++)
        {
            vis.removeAllPointCloud();
            pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler(keyframes_.back());
            vis.addPointCloud(keyframes_.back(), rgb_handler, "original_cloud");


            pcl::PointCloud<PointT>::Ptr segmented (new pcl::PointCloud<PointT>());
            pcl::copyPointCloud(*keyframes_.back(), planes[cluster_id]->indices, *segmented);
            if (planes[cluster_id]->is_plane)
            {
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> red_source (segmented, 255, 0, 0);
                vis.addPointCloud(segmented, red_source, "segmented");
            }
            else
            {
                break;
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> green_source (segmented, 0, 255, 0);
                vis.addPointCloud(segmented, green_source, "segmented");
            }
//            vis.spin();
        }
        vis.removeAllPointClouds();
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler(keyframes_.back());
        vis.addPointCloud(keyframes_.back(), rgb_handler, "original_cloud");
//        vis.spin();
    }
#endif
    std::vector<bool> obj_mask_enforced_by_supervoxel_consistency;
    updatePointNormalsFromSuperVoxels(view.cloud_,
                                      view.normal_,
                                      view.obj_mask_step_.back(),
                                      obj_mask_enforced_by_supervoxel_consistency,
                                      view.supervoxel_cloud_);
    view.obj_mask_step_.push_back( obj_mask_enforced_by_supervoxel_consistency );

    std::vector<bool> obj_mask_grown_by_smooth_surface = extractEuclideanClustersSmooth(view.cloud_,
                                                                                           *view.normal_,
                                                                                           obj_mask_enforced_by_supervoxel_consistency,
                                                                                           pixel_is_neglected);
    view.obj_mask_step_.push_back(obj_mask_grown_by_smooth_surface);

    std::vector<bool> obj_mask_eroded = erodeIndices(obj_mask_grown_by_smooth_surface, *view.cloud_);
    view.obj_mask_step_.push_back( obj_mask_eroded );

    for( size_t step_id = 0; step_id<view.obj_mask_step_.size(); step_id++)
    {
        std::cout << "step " << step_id << ": " << v4r::common::createIndicesFromMask(view.obj_mask_step_[step_id]).size() << " points." << std::endl;
    }

    if( view.is_pre_labelled_ && v4r::common::createIndicesFromMask(view.obj_mask_step_.back()).size() < param_.min_points_for_transferring_)
    {
        view.obj_mask_step_.back() = view.obj_mask_step_[0];
        std::cout << "After postprocessing the initial frame not enough points are left. Therefore taking the original provided indices." << std::endl;
    }
//    visualize();
    return true;
}

void
DOL::initialize (int argc, char ** argv)
{
    if (param_.do_sift_based_camera_pose_estimation_)
    {
#ifdef USE_SIFT_GPU

        //-----Init-SIFT-GPU-Context--------
        static char kw[][16] = {"-m", "-fo", "-1", "-s", "-v", "1", "-pack"};
        char * argvv[] = {kw[0], kw[1], kw[2], kw[3],kw[4],kw[5],kw[6], NULL};

        int argcc = sizeof(argvv) / sizeof(char*);
        sift_ = new SiftGPU ();
        sift_->ParseParam (argcc, argvv);

        //create an OpenGL context for computation
        if (sift_->CreateContextGL () != SiftGPU::SIFTGPU_FULL_SUPPORTED)
            throw std::runtime_error ("PSiftGPU::PSiftGPU: No GL support!");
#endif
    }
}

void
DOL::printParams(std::ostream &ostr) const
{
    ostr << "Started dynamic object learning with parameters: " << std::endl
         << "===================================================" << std::endl
         << "radius: " << param_.radius_ << std::endl
         << "eps_angle: " << param_.eps_angle_ << std::endl
         << "seed resolution: " << param_.seed_resolution_ << std::endl
         << "voxel resolution: " << param_.voxel_resolution_ << std::endl
         << "ratio: " << param_.ratio_ << std::endl
         << "do_erosion: " << param_.do_erosion_ << std::endl
         << "max z distance: " << param_.chop_z_ << std::endl
         << "transferring object indices from latest frame only: " << param_.transfer_indices_from_latest_frame_only_ << std::endl
         << "apply minimimum spanning tree: " << param_.do_mst_refinement_ << std::endl
         << "Filter only planes (no other Euclidean clusters): " << param_.filter_planes_only_ << std::endl
         << "===================================================" << std::endl << std::endl;
}
}
}
