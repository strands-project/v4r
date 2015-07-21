/*
 * do_learning.cpp
 *
 * incrementally learning of objects by
 * transfering object indices from initial cloud to the remaining clouds
 * using given camera poses
 *
 *  Created on: June, 2015
 *      Author: Aitor Aldoma, Thomas Faeulhammer
 */


#ifndef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#endif

#include "v4r/object_modelling/do_learning.h"

#include <stdlib.h>
#include <thread>

#include <pcl/common/angles.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#include <v4r/common/keypoint/impl/convertCloud.hpp>
#include <v4r/common/keypoint/impl/convertImage.hpp>
#include <v4r/common/keypoint/impl/convertNormals.hpp>
//#include <v4r/KeypointSlam/KeypointSlamRGBD2.hh>
//#include <v4r/KeypointSlam/ProjBundleAdjuster.hh>
#include <v4r/common/keypoint/impl/DataMatrix2D.hpp>
//#include <v4r/KeypointTools/invPose.hpp>
//#include <v4r/KeypointTools/PoseIO.hpp>
//#include <v4r/KeypointTools/ScopeTime.hpp>
//#include <v4r/KeypointTools/toString.hpp>
//#include <v4r/KeypointTools/ZAdaptiveNormals.hh>
#include <v4r/registration/FeatureBasedRegistration.h>
#include <v4r/common/organized_edge_detection.h>
#include <v4r/common/features/sift_local_estimator.h>
#include <v4r/common/fast_icp_with_gc.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/noise_model_based_cloud_integration.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/common/io/filesystem_utils.h>


#define NUM_SUBWINDOWS 7

//#define USE_SIFT_GPU

#ifndef USE_SIFT_GPU
    #include <v4r/common/features/opencv_sift_local_estimator.h>
#endif

using namespace v4r::object_modelling;

float DOL::
calcEdgeWeightAndRefineTf (const pcl::PointCloud<PointT>::ConstPtr &cloud_src,
                const pcl::PointCloud<PointT>::ConstPtr &cloud_dst,
                Eigen::Matrix4f &transform)
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

//    std::vector<int> dummy_idx;
//    pcl::removeNaNFromPointCloud(*cloud_src, *cloud_src_wo_nan, dummy_idx);
//    pcl::removeNaNFromPointCloud(*cloud_dst, *cloud_dst_wo_nan, dummy_idx);

    float w_after_icp_ = std::numeric_limits<float>::max ();
    const float best_overlap_ = 0.75f;

    Eigen::Matrix4f icp_trans;
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
    w_after_icp_ = icp.getFinalTransformation ( icp_trans );

    if ( w_after_icp_ < 0 || !pcl_isfinite ( w_after_icp_ ) )
    {
        w_after_icp_ = std::numeric_limits<float>::max ();
    }
    else
    {
        w_after_icp_ = best_overlap_ - w_after_icp_;
    }

//    transform = icp_trans; // refined transformation

//        grph[edge].edge_weight = w_after_icp_;

    std::cout << "WEIGHT IS: " << w_after_icp_ << " coming from edge connecting ." << std::endl;
}

bool DOL::
calcSiftFeatures (const pcl::PointCloud<PointT>::Ptr &cloud_src,
                  pcl::PointCloud<PointT>::Ptr &sift_keypoints,
                  pcl::PointIndices &sift_keypoint_indices,
                  pcl::PointCloud<FeatureT>::Ptr &sift_signatures,
                  std::vector<float> &sift_keypoint_scales)
{


    //    if(use_table_plane)
    //        estimator->setIndices (*(grph[src].pIndices_above_plane));

#ifdef USE_SIFT_GPU
    boost::shared_ptr < faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, FeatureT> > estimator;
    estimator.reset (new faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, FeatureT>(sift_));

    bool ret = estimator->estimate (cloud_src, sift_keypoints, sift_signatures, sift_keypoint_scales);
    estimator->getKeypointIndices( sift_keypoint_indices );    
#else
      boost::shared_ptr < v4r::rec_3d_framework::OpenCVSIFTLocalEstimation<PointT, pcl::Histogram<128> > > estimator;
      estimator.reset (new v4r::rec_3d_framework::OpenCVSIFTLocalEstimation<PointT, pcl::Histogram<128> >);

      pcl::PointCloud<PointT>::Ptr processed_foo (new pcl::PointCloud<PointT>());

      bool ret = estimator->estimate (cloud_src, processed_foo, sift_keypoints, sift_signatures);
      estimator->getKeypointIndices( sift_keypoint_indices );

//      boost::shared_ptr < faat_pcl::rec_3d_framework::LocalEstimator<PointT, pcl::Histogram<128> > > cast_estimator;
//      cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::OpenCVSIFTLocalEstimation<PointT, pcl::Histogram<128> > > (estimator);
#endif
    return ret;
}

//void DOL::estimateViewTransformationBySIFT ( size_t src, size_t dst, boost::shared_ptr< flann::Index<DistT> > &flann_index, std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transformations, bool use_gc )
void
DOL::estimateViewTransformationBySIFT(const pcl::PointCloud<PointT> &src_cloud,
                                      const pcl::PointCloud<PointT> &dst_cloud,
                                      const std::vector<size_t> &src_sift_keypoint_indices,
                                      const std::vector<size_t> &dst_sift_keypoint_indices,
                                      const pcl::PointCloud<FeatureT> &dst_sift_signatures,
                                      boost::shared_ptr< flann::Index<DistT> > &src_flann_index,
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
        FeatureT searchFeature = dst_sift_signatures[ keypointId ];
        int size_feat = sizeof ( searchFeature.histogram ) / sizeof ( float );
        v4r::common::miscellaneous::nearestKSearch ( src_flann_index, searchFeature.histogram, size_feat, K, indices, distances );

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

        Eigen::Matrix4f bla = rej->getBestTransformation();
        Eigen::Matrix4f bla2;
        transformations.push_back( rej->getBestTransformation () );
        pcl::registration::TransformationEstimationSVD<PointT, PointT> t_est;
        t_est.estimateRigidTransformation (*pSiftKeypointsSrc, *pSiftKeypointsDst, *after_rej_correspondences, bla2);
        transformations.back() = bla2;
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
        transformations.insert(transformations.end(), transformations.begin(), transformations.end());
    }
}


void DOL::extractEuclideanClustersSmooth (
        const pcl::PointCloud<PointT>::ConstPtr &cloud,
        const pcl::PointCloud<pcl::Normal> &normals,
        const std::vector<int> &initial,
        std::vector<int> &cluster) const
{
    float tolerance = radius_;

    if (cloud->points.size () != normals.points.size ())
    {
        PCL_ERROR ("[pcl::extractEuclideanClusters] Number of points in the input point cloud (%zu) different than normals (%zu)!\n", cloud->points.size (), normals.points.size ());
        return;
    }

    pcl::octree::OctreePointCloudSearch<PointT> octree(0.005f);
    octree.setInputCloud ( cloud );
    octree.addPointsFromInputCloud ();

    cluster.clear(); // is this okay???

    // Create a bool vector of processed point indices, and initialize it to false
    std::vector<bool> to_grow (cloud->points.size (), false);
    std::vector<bool> in_cluster (cloud->points.size (), false);

    std::vector<int>::const_iterator it;
    for(it = initial.begin(); it != initial.end(); it++)
    {
        to_grow[*it] = true;
        in_cluster[*it] = true;
    }

    bool stop = false;

    while(!stop)
    {
        std::vector<int> nn_indices;
        std::vector<float> nn_distances;
        // Process all points in the indices vector
        for (size_t i = 0; i < cloud->points.size (); i++)
        {
            if (!to_grow.back())
                continue;

            to_grow.back() = false;

            if (octree.radiusSearch (cloud->points.back(), tolerance, nn_indices, nn_distances))
            {
                for (size_t j = 0; j < nn_indices.size (); j++) // is nn_indices[0] the same point?
                {
                    if(!in_cluster[nn_indices[j]])
                    {
                        //check smoothness constraint
                        Eigen::Vector3f n1 = normals.points.back().getNormalVector3fMap();
                        Eigen::Vector3f n2 = normals.points[nn_indices[j]].getNormalVector3fMap();
                        n1.normalize();
                        n2.normalize();
                        float dot_p = n1.dot(n2);
                        if (dot_p >= eps_angle_)
                        {
                            to_grow[nn_indices[j]] = true;
                            in_cluster[nn_indices[j]] = true;
                        }
                    }
                }
            }
        }

        size_t ngrow = 0;
        for (size_t i = 0; i < cloud->points.size (); ++i)
        {
            if(to_grow.back())
                ngrow++;
        }

        if(ngrow == 0)
            stop = true;
    }

    for (size_t i = 0; i < cloud->points.size (); ++i)
    {
        if(in_cluster.back())
            cluster.push_back(static_cast<int>(i));
    }
}

void DOL::updatePointNormalsFromSuperVoxels(const pcl::PointCloud<PointT>::Ptr & cloud,
                                            pcl::PointCloud<pcl::Normal>::Ptr & normals,
                                            const std::vector<int> & obj_points,
                                            std::vector<int> & good_neighbours,
                                            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &supervoxel_cloud)
{
    good_neighbours.clear();

    pcl::SupervoxelClustering<PointT> super (voxel_resolution_, seed_resolution_, false);
    super.setInputCloud (cloud);
    super.setColorImportance (0.f);
    super.setSpatialImportance (0.5f);
    super.setNormalImportance (2.f);
    super.setNormalCloud(normals);
    std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
    super.extract (supervoxel_clusters);
    super.refineSupervoxels(2, supervoxel_clusters);

    pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());

    const pcl::PointCloud<pcl::PointXYZL>::Ptr supervoxels_labels_cloud = super.getLabeledCloud();
    uint32_t max_label = super.getMaxLabel();

    supervoxel_cloud = super.getColoredVoxelCloud();

    const pcl::PointCloud<pcl::PointNormal>::Ptr sv_normal_cloud = super.makeSupervoxelNormalCloud (supervoxel_clusters);

    std::vector<int> label_to_idx;
    label_to_idx.resize(max_label + 1, -1);

    typename std::map <uint32_t, typename pcl::Supervoxel<PointT>::Ptr>::iterator sv_itr,sv_itr_end;
    sv_itr = supervoxel_clusters.begin ();
    sv_itr_end = supervoxel_clusters.end ();
    int i=0;
    for ( ; sv_itr != sv_itr_end; ++sv_itr, i++)
    {
        label_to_idx[sv_itr->first] = i;
    }

    //count total number of pixels for each supervoxel
    size_t sv_size = supervoxel_clusters.size ();
    std::vector<int> label_count;
    label_count.resize ( supervoxel_clusters.size(), 0 );

    for(size_t i=0; i < supervoxels_labels_cloud->size(); i++)
    {
        const int sv_idx = label_to_idx[supervoxels_labels_cloud->at(i).label];
        if(sv_idx < 0 || sv_idx >= sv_size)
            continue;

        const Eigen::Vector3f sv_normal = sv_normal_cloud->points[sv_idx].getNormalVector3fMap();
        normals->points.back().getNormalVector3fMap() = sv_normal;
        label_count[sv_idx]++;
    }

    //count for all labels how many pixels are in the initial indices
    std::vector<int> label_count_nn;
    label_count_nn.resize(sv_size, 0);

    for(size_t id = 0; id < obj_points.size(); id++)
    {
        const int sv_idx = label_to_idx[ supervoxels_labels_cloud->at( obj_points[ id ] ).label];
        if(sv_idx >= 0 && sv_idx < sv_size)
        {
            label_count_nn[sv_idx]++;
        }
    }

    for(size_t id = 0; id < obj_points.size(); id++)
    {
        const int sv_idx = label_to_idx[ supervoxels_labels_cloud->at( obj_points[ id ] ).label];
        if(sv_idx < 0 || sv_idx >= sv_size)
            continue;

        if( (label_count_nn[sv_idx] / (float)label_count[sv_idx]) > ratio_)
        {
            good_neighbours.push_back( obj_points[ id ] );
        }
    }
}

void DOL::transferIndicesAndNNSearch(size_t origin, size_t dest, std::vector<bool> &obj_mask, const Eigen::Matrix4f &transform)
{
    pcl::PointCloud<PointT>::Ptr cloud_origin_filtered(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr segmented(new pcl::PointCloud<PointT>);

    pcl::copyPointCloud(*keyframes_[origin], scene_points_[origin], *cloud_origin_filtered);

    if (do_erosion_)
    {
        pcl::copyPointCloud(*keyframes_[origin], obj_indices_eroded_to_original_[origin], *segmented);
    }
    else
    {
        pcl::copyPointCloud(*cloud_origin_filtered, obj_indices_2_to_filtered_[origin], *segmented);
    }

    pcl::PointCloud<PointT>::Ptr segmented_trans(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*segmented, *segmented_trans, transform);
    transferred_cluster_[dest]->points.insert(transferred_cluster_[dest]->points.end(),
                                              segmented_trans->points.begin(),
                                              segmented_trans->points.end());// = segmented_trans;

    //find neighbours from transferred object points
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    for(size_t i=0; i < segmented_trans->points.size(); i++)
    {
        //        if (octree.nearestKSearch (segmented_trans->points.back(), 1, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
        //            //if (octree.radiusSearch (segmented_trans->points.back(), radius_, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
        //        {
        //            if(pointRadiusSquaredDistance[0] <= (radius_ * radius_))
        //                all_neighbours.insert(pointIdxRadiusSearch.begin(), pointIdxRadiusSearch.end());
        //        }
        if ( ! pcl::isFinite(segmented_trans->points.back()) )
        {
            PCL_WARN ("Warning: Point is NaN.\n");    // not sure if this causes somewhere else a problem. This condition should not be fulfilled.
            continue;
        }
        if ( octree_.radiusSearch (segmented_trans->points.back(), radius_, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
        {
            for( size_t nn_id = 0; nn_id < pointIdxRadiusSearch.size(); nn_id++)
            {
                obj_mask[ pointIdxRadiusSearch[ nn_id ] ] = true;
            }
        }
    }
}

void DOL::erodeInitialIndices(const pcl::PointCloud<PointT> & cloud,
                              const pcl::PointIndices & initial_indices,
                              pcl::PointIndices & eroded_indices)
{
    cv::Mat mask = cv::Mat(cloud.height, cloud.width, CV_8UC1);
    cv::Mat mask_dst;
    mask.setTo(0);

    for(size_t i=0; i < initial_indices.indices.size(); i++)
    {
        int r,c;
        r = initial_indices.indices.back() / mask.cols;
        c = initial_indices.indices.back() % mask.cols;

        mask.at<unsigned char>(r,c) = 255;
    }

    cv::Mat const structure_elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat close_result;
    cv::morphologyEx(mask, close_result, cv::MORPH_CLOSE, structure_elem);

    cv::erode(close_result, mask_dst, cv::Mat(), cv::Point(-1,-1), 3);

//        cv::imshow("mask", mask);
//        cv::imshow("close_result", close_result);
//        cv::imshow("mask_dst", mask_dst);
//        cv::waitKey(0);

    eroded_indices.indices.clear();
    eroded_indices.indices.resize(mask_dst.rows * mask_dst.cols);
    size_t kept = 0;
    for(int r=0; r < mask_dst.rows; r++)
    {
        for(int c=0; c< mask_dst.cols; c++)
        {
            const int idx = r * mask_dst.cols + c;

            if (    mask_dst.at<unsigned char>(r,c) > 0
                && pcl::isFinite( cloud.points[idx] )
                && cloud.points[idx].z < chop_z_        )
            {
                eroded_indices.indices[kept] = idx;
                kept++;
            }
        }
    }
    eroded_indices.indices.resize(kept);
}


bool
DOL::save_model (const std::string &models_dir, const std::string &recognition_structure_dir, const std::string &model_name)
{
    std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > keyframes_used;
    std::vector< pcl::PointCloud<pcl::Normal>::Ptr > normals_used;
    std::vector<Eigen::Matrix4f> cameras_used;
    std::vector<pcl::PointCloud<IndexPoint> > object_indices_clouds;
    std::vector<std::vector<float> > weights;
    std::vector<std::vector<int> > indices;

    weights.resize(keyframes_.size());
    indices.resize(keyframes_.size());
    object_indices_clouds.resize(keyframes_.size());
    keyframes_used.resize(keyframes_.size());
    normals_used.resize(keyframes_.size());
    cameras_used.resize(keyframes_.size());


    // only used keyframes with have object points in them
    size_t kept_keyframes=0;
    for(size_t i=0; i < keyframes_.size(); i++)
    {
        if ( obj_indices_eroded_to_original_.back().indices. size() )
        {
            keyframes_used[ kept_keyframes ] = keyframes_.back();
            normals_used [ kept_keyframes ] = normals_.back();
            cameras_used [ kept_keyframes ] = cameras_.back();
            indices[ kept_keyframes ] = obj_indices_eroded_to_original_.back().indices;

            object_indices_clouds[ kept_keyframes ].points.resize( indices[ kept_keyframes ].size());

            for(size_t k=0; k < indices[ kept_keyframes ].size(); k++)
            {
                object_indices_clouds[ kept_keyframes ].points[k].idx = indices[ kept_keyframes ][k];
            }
            kept_keyframes++;
        }
    }
    weights.resize(kept_keyframes);
    indices.resize(kept_keyframes);
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
            nm.setInputCloud(keyframes_used.back());
            nm.setInputNormals(normals_used.back());
            nm.setLateralSigma(0.001);
            nm.setMaxAngle(60.f);
            nm.setUseDepthEdges(true);
            nm.compute();
            nm.getWeights(weights.back());
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        v4r::utils::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration;
        nmIntegration.setInputClouds(keyframes_used);
        nmIntegration.setResolution(0.002f);
        nmIntegration.setWeights(weights);
        nmIntegration.setTransformations(cameras_used);
        nmIntegration.setMinWeight(0.5f);
        nmIntegration.setInputNormals(normals_used);
        nmIntegration.setMinPointsPerVoxel(1);
        nmIntegration.setFinalResolution(0.002f);
        nmIntegration.setIndices(indices);
        nmIntegration.setThresholdSameSurface(0.01f);
        nmIntegration.compute(octree_cloud);

        pcl::PointCloud<pcl::Normal>::Ptr octree_normals;
        nmIntegration.getOutputNormals(octree_normals);

        std::stringstream export_to_rs;
        export_to_rs << recognition_structure_dir << "/" << model_name << "/";
        std::string export_to = export_to_rs.str();

        createDirIfNotExist(recognition_structure_dir);
        createDirIfNotExist(models_dir);
        createDirIfNotExist(export_to);

        std::cout << "Saving " << kept_keyframes << " keyframes from " << keyframes_.size() << "." << std::endl;

        //save recognition data with new poses
        for(size_t i=0; i < kept_keyframes; i++)
        {
            std::stringstream view_file;
            view_file << export_to << "/cloud_" << setfill('0') << setw(8) << i << ".pcd";
            pcl::io::savePCDFileBinary (view_file.str (), *(keyframes_used.back()));
            std::cout << view_file.str() << std::endl;

            std::string path_pose (view_file.str());
            boost::replace_last (path_pose, "cloud", "pose");
            boost::replace_last (path_pose, ".pcd", ".txt");
            v4r::common::io::writeMatrixToFile(path_pose, cameras_used.back());
            std::cout << path_pose << std::endl;

            std::string path_obj_indices (view_file.str());
            boost::replace_last (path_obj_indices, "cloud", "object_indices");
            pcl::io::savePCDFileBinary (path_obj_indices, object_indices_clouds.back());
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

//        sensor_msgs::PointCloud2 recognizedModelsRos;
//        pcl::toROSMsg (*cloud_normals_oriented, recognizedModelsRos);
//        recognizedModelsRos.header.frame_id = "world";
//        vis_pc_pub_.publish(recognizedModelsRos);
    }

    return true;
}

void DOL::extractPlanePoints(const pcl::PointCloud<PointT>::ConstPtr &cloud,
                             const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
                             const kp::ClusterNormalsToPlanes::Parameter p_param,
                             std::vector<kp::ClusterNormalsToPlanes::Plane::Ptr> &planes)
{
    kp::ClusterNormalsToPlanes pest(p_param);

    kp::DataMatrix2D<Eigen::Vector3f>::Ptr kp_cloud( new kp::DataMatrix2D<Eigen::Vector3f>() );
    kp::DataMatrix2D<Eigen::Vector3f>::Ptr kp_normals( new kp::DataMatrix2D<Eigen::Vector3f>() );
    kp::convertCloud(*cloud, *kp_cloud);
    kp::convertNormals(*normals, *kp_normals);
    pest.compute(*kp_cloud, *kp_normals, planes);
}

void DOL::getPlanesNotSupportedByObjectMask(const std::vector<kp::ClusterNormalsToPlanes::Plane::Ptr> &planes,
                                            const pcl::PointIndices object_mask,
                                            std::vector<kp::ClusterNormalsToPlanes::Plane::Ptr> &planes_dst,
                                            pcl::PointIndices &all_plane_indices_wo_object,
                                            float ratio)
{
    planes_dst.clear();
    all_plane_indices_wo_object.indices.clear();

    for(size_t cluster_id=0; cluster_id<planes.size(); cluster_id++)
    {
        size_t num_obj_pts = 0;

        if (planes[cluster_id]->is_plane)
        {
            for (size_t cluster_pt_id=0; cluster_pt_id<planes[cluster_id]->indices.size(); cluster_pt_id++)
            {
                for (size_t obj_pt_id=0; obj_pt_id<object_mask.indices.size(); obj_pt_id++)
                {
                    if (object_mask.indices[obj_pt_id] == planes[cluster_id]->indices[cluster_pt_id])
                    {
                        num_obj_pts++;
                    }
                }
            }

            if ( num_obj_pts < ratio * planes[cluster_id]->indices.size() )
            {
                planes_dst.push_back( planes[cluster_id] );
                all_plane_indices_wo_object.indices.insert(all_plane_indices_wo_object.indices.end(),
                                                           planes[cluster_id]->indices.begin(),
                                                           planes[cluster_id]->indices.end());
            }

        }
    }
}

void DOL::createMaskFromIndices( const std::vector<int> &objectIndices,
                            size_t image_size,
                            std::vector<bool> &mask)
{
    mask.resize( image_size );

    for (size_t pt_id=0; pt_id < image_size; pt_id++)
    {
        mask [pt_id] = false;
    }

    for (size_t obj_pt_id = 0; obj_pt_id < objectIndices.size(); obj_pt_id++)
    {
        mask [objectIndices[obj_pt_id]] = true;
    }
}

void DOL::updateIndicesConsideringMask(const std::vector<bool> &bg_mask,
                                       const std::vector<bool> &fg_mask,
                                       std::vector<int> &fg_indices,
                                       std::vector<int> &old_bg_indices)
{
    fg_indices.resize( fg_mask.size() );
    old_bg_indices.resize( fg_mask.size() );

    size_t kept=0;
    size_t num_new_pxs = 0;
    for(size_t old_px_id=0; old_px_id < bg_mask.size(); old_px_id++)
    {
        if( !bg_mask[old_px_id] && fg_mask[old_px_id] ) // pixel is not neglected and belongs to object -> save
        {
            fg_indices[kept] = num_new_pxs;
            kept++;
        }
        if( !bg_mask [old_px_id] )
        {
            old_bg_indices[num_new_pxs] = old_px_id;
            num_new_pxs++;
        }
    }
    fg_indices.resize(kept);
    old_bg_indices.resize(num_new_pxs);
}

bool
DOL::learn_object (const pcl::PointCloud<PointT> &cloud, std::vector<size_t> &initial_indices)
{
    modelView view;
    pcl::copyPointCloud(cloud, view.cloud_);

    size_t num_frames = keyframes_.size() + 1;
    reserveMem(num_frames);

        std::cout << "Computing indices for cloud " << num_frames << std::endl
                  << "===================================" << std::endl;

        pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>());
        pcl::PointCloud<pcl::Normal>::Ptr normals_filtered (new pcl::PointCloud<pcl::Normal>());
        std::vector<kp::ClusterNormalsToPlanes::Plane::Ptr> planes, planes_wo_obj;
        pcl::PointIndicesPtr obj_indices_updated (new pcl::PointIndices);
        pcl::PointIndices plane_and_nan_points;
        std::vector<bool> pixel_is_object;
        std::vector<bool> pixel_is_neglected;

//        cameras_.back() = fromGMTransform(req.transforms.back());

        Vertex vrtx = boost::add_vertex ( grph_ );
        grph_[vrtx].id = num_frames-1;

//        computeNormals(keyframes_.back(), normals_.back(), normal_method_);
        v4r::common::miscellaneous::computeNormals(view.cloud_, view.normal_, normal_method_);
//        {
//            pcl::visualization::PCLVisualizer vis("cloud");
//            pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler(keyframes_.back());
//            vis.addPointCloud(keyframes_.back(), rgb_handler, "original_cloud");
//            vis.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (keyframes_.back(), normals_.back(), 10, 0.05, "normals");
//            vis.spin();
//        }

        assert(view.cloud_.points.size() == view.normal_.points.size());

        octree_.setInputCloud ( view.cloud_ );
        octree_.addPointsFromInputCloud ();

        boost::shared_ptr<flann::Index<DistT> > flann_index;

        if ( do_sift_based_camera_pose_estimation_ )
        {
            pcl::PointCloud<PointT>::Ptr sift_keypoints (new pcl::PointCloud<PointT>());
            std::vector<float> sift_keypoint_scales;
            try
            {
                calcSiftFeatures( view.cloud_, sift_keypoints, view.sift_keypoint_indices_, view.sift_signatures_, sift_keypoint_scales);
                v4r::common::miscellaneous::convertToFLANN<FeatureT, DistT>(view.sift_signatures_, flann_index );
            }
            catch (int e)
            {
                do_sift_based_camera_pose_estimation_ = false;
                std::cerr << "Something is wrong with the SIFT based camera pose estimation. Turning it off and using the given camera poses only." << std::endl;
            }
        }

        if (initial_indices.size())   // for first frame use given initial indices and erode them
        {
            for(size_t i=0; i < initial_indices.size(); i++)
            {
                view.transferred_nn_points_.indices.push_back(initial_indices.back());
            }
            //erode mask
            erodeInitialIndices(*view.cloud_, view.transferred_nn_points_, view.obj_indices_eroded_to_original_);
            view.obj_indices_2_to_filtered_ = view.transferred_nn_points_;
        }
        else
        {
            std::vector<bool> obj_mask;
            obj_mask.resize(view.cloud_->points.size());
            for (size_t pt_id=0; pt_id < obj_mask.size(); pt_id++)
            {
                obj_mask [pt_id] = false;
            }

                size_t k;

                if (transfer_indices_from_latest_frame_only_)
                    k = i-1;
                else
                    k = 0; // transfer object indices from all frames

                for (k; k < i; k++)
                {
                    modelView other_view;
                    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms;
                    transforms.push_back( view.camera_pose_.inverse() * other_view.camera_pose_ );
                    if ( do_sift_based_camera_pose_estimation_ )
                    {
                        try
                        {
                            estimateViewTransformationBySIFT(i, k, flann_index, transforms);
                        }
                        catch (int e)
                        {
                            do_sift_based_camera_pose_estimation_ = false;
                            std::cerr << "Something is wrong with the SIFT based camera pose estimation. Turning it off and using the given camera poses only." << std::endl;
                        }
                    }

                    size_t best_transform_id = 0;
                    float lowest_edge_weight = std::numeric_limits<float>::max();

                    for ( size_t trans_id = 0; trans_id < transforms.size(); trans_id++ )
                    {
                        float edge_weight;
                        try
                        {
                            edge_weight = calcEdgeWeightAndRefineTf( keyframes_[k], keyframes_.back(), transforms[ trans_id ]);
                            if(edge_weight < lowest_edge_weight)
                            {
                                lowest_edge_weight = edge_weight;
                                best_transform_id = trans_id;
                            }
                        }
                        catch (int e)
                        {
                            do_sift_based_camera_pose_estimation_ = false;
                            std::cerr << "Something is wrong with the SIFT based camera pose estimation. Turning it off and using the given camera poses only." << std::endl;
                            break;
                        }

//                        pcl::visualization::PCLVisualizer vis("sift based pose estimate");
//                        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler(keyframes_.back());
//                        vis.addPointCloud(keyframes_.back(), rgb_handler, "src_cloud");
//                        pcl::PointCloud<PointT>::Ptr transf_cloud (new pcl::PointCloud<PointT>());
//                        pcl::transformPointCloud(*keyframes_[k], *transf_cloud, transforms[0]);
//                        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler2( transf_cloud );
//                        vis.addPointCloud( transf_cloud, rgb_handler2, "dst_cloud");
//                        vis.spin();
                    }
                    Edge e_cpy; bool b;
//                    tie ( e_cpy, b ) = add_edge ( vrtx_src, vrtx_trgt, grph_final_ );
                    transferIndicesAndNNSearch(k, i, obj_mask, transforms[ best_transform_id ]); //transferred_nn_points_.back().indices);
                }

            transferred_nn_points_.back().indices.resize( keyframes_[0]->points.size() );
            size_t kept = 0;
            for (size_t pt_id=0; pt_id < obj_mask.size(); pt_id++)
            {
                if( obj_mask [pt_id] )
                {
                    transferred_nn_points_.back().indices[kept] = pt_id;
                    kept++;
                }
            }
            transferred_nn_points_.back().indices.resize( kept );
        }


        createMaskFromIndices(transferred_nn_points_.back().indices, keyframes_.back()->points.size(), pixel_is_object);
        extractPlanePoints(keyframes_.back(), normals_.back(), p_param_, planes);
        getPlanesNotSupportedByObjectMask(planes,
                                          transferred_nn_points_.back(),
                                          planes_wo_obj,
                                          plane_and_nan_points);

        // append indices with nan values or points further away than chop_z_ parameter
        std::vector<int> nan_indices;
        nan_indices.resize( keyframes_.back()->points.size() );
        size_t kept=0;
        for(size_t pt_id = 0; pt_id<keyframes_.back()->points.size(); pt_id++)
        {
            if ( !pcl::isFinite(keyframes_.back()->points[pt_id]) || keyframes_.back()->points[pt_id].z > chop_z_)
            {
                nan_indices[kept] = pt_id;
                kept++;
            }
        }
        nan_indices.resize(kept);
        plane_and_nan_points.indices.insert( plane_and_nan_points.indices.end(),
                                                    nan_indices.begin(), nan_indices.end() );

        createMaskFromIndices(plane_and_nan_points.indices, keyframes_.back()->points.size(), pixel_is_neglected);

        updateIndicesConsideringMask(pixel_is_neglected, pixel_is_object, obj_indices_updated->indices, scene_points_.back().indices);
        pcl::copyPointCloud(*keyframes_.back(), scene_points_.back(), *cloud_filtered);
        pcl::copyPointCloud(*normals_.back(), scene_points_.back(), *normals_filtered);


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
                vis.spin();
            }
            vis.removeAllPointClouds();
            pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler(keyframes_.back());
            vis.addPointCloud(keyframes_.back(), rgb_handler, "original_cloud");
            vis.spin();
        }
#endif

        updatePointNormalsFromSuperVoxels(cloud_filtered,
                                          normals_filtered,
                                          obj_indices_updated->indices,
                                          initial_indices_good_to_unfiltered_.back().indices,
                                          supervoxeled_clouds_.back());

        extractEuclideanClustersSmooth(cloud_filtered, *normals_filtered,
                                       initial_indices_good_to_unfiltered_.back().indices,
                                       obj_indices_2_to_filtered_.back().indices);


        // Transferring indices corresponding to filtered cloud such that they correspond to original (unfiltered) cloud.
        obj_indices_3_to_original_.back().indices.resize( obj_indices_2_to_filtered_.back().indices.size() );
        for (size_t obj_pt_id=0; obj_pt_id<obj_indices_2_to_filtered_.back().indices.size(); obj_pt_id++)
        {
            obj_indices_3_to_original_.back().indices[obj_pt_id] = scene_points_.back().indices[ obj_indices_2_to_filtered_.back().indices[obj_pt_id] ];
        }

        erodeInitialIndices(*keyframes_.back(), obj_indices_3_to_original_.back(), obj_indices_eroded_to_original_.back());

        std::cout << "Found " << transferred_nn_points_.back().indices.size() << " nearest neighbors before growing." << std::endl
                  << "After updatePointNormalsFromSuperVoxels size: " << initial_indices_good_to_unfiltered_.back().indices.size() << std::endl
                  << "size of final cluster: " << obj_indices_2_to_filtered_.back().indices.size() << std::endl
                  << "size of final cluster after erosion: " << obj_indices_eroded_to_original_.back().indices.size() << std::endl << std::endl;

        if( i==0 && obj_indices_eroded_to_original_[0].indices.size() < min_points_for_transferring_)
        {
            obj_indices_eroded_to_original_[0].indices = transferred_nn_points_[0].indices;
            std::cout << "After postprocessing the initial frame not enough points are left. Therefore taking the original provided indices." << std::endl;
        }

    return true;
}

void DOL::createBigCloud()
{
    for(size_t i=0; i < keyframes_.size(); i++)
    {
        pcl::PointCloud<PointT>::Ptr cloud_trans (new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr cloud_trans_filtered (new pcl::PointCloud<PointT>());
        pcl::transformPointCloud(*keyframes_.back(), *cloud_trans, cameras_.back());
        pcl::copyPointCloud(*cloud_trans, scene_points_.back(), *cloud_trans_filtered);
        *big_cloud_ += *cloud_trans_filtered;

        pcl::PointCloud<PointT>::Ptr segmented_trans (new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*cloud_trans_filtered, obj_indices_2_to_filtered_.back(), *segmented_trans);
        *big_cloud_segmented_ += *segmented_trans;
    }
}

bool DOL::visualizeROS(do_learning_srv_definitions::visualize::Request & req,
                    do_learning_srv_definitions::visualize::Response & response)
{
    visualize();
    return true;
}

void DOL::visualize()
{
#ifndef USE_REMOTE_PCL_VISUALIZER
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
#endif
    if (!vis_) {
#ifdef USE_REMOTE_PCL_VISUALIZER
        vis_.reset(new RemotePCLVisualizer());
#else
        vis_.reset(new pcl::visualization::PCLVisualizer());
#endif
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

#ifdef USE_REMOTE_PCL_VISUALIZER
    vis_viewpoint_ = vis_->createFramework (keyframes_.size(), NUM_SUBWINDOWS, subwindow_title);
#else
    vis_viewpoint_ = v4r::common::pcl_visualizer::visualization_framework (vis_, keyframes_.size(), NUM_SUBWINDOWS, subwindow_title);
#endif
    for(size_t i=0; i < keyframes_.size(); i++)
    {
        pcl::PointCloud<PointT>::Ptr cloud_trans (new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr cloud_trans_filtered (new pcl::PointCloud<PointT>());
        pcl::transformPointCloud(*keyframes_.back(), *cloud_trans, cameras_.back());
        pcl::copyPointCloud(*cloud_trans, scene_points_.back(), *cloud_trans_filtered);

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sv_trans (new pcl::PointCloud<pcl::PointXYZRGBA>());
        pcl::transformPointCloud(*supervoxeled_clouds_.back(), *sv_trans, cameras_.back());

        pcl::PointCloud<PointT>::Ptr segmented (new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr segmented_trans (new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*keyframes_.back(), transferred_nn_points_.back(), *segmented);
        pcl::transformPointCloud(*segmented, *segmented_trans, cameras_.back());

        pcl::PointCloud<PointT>::Ptr segmented2_trans (new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*cloud_trans_filtered, initial_indices_good_to_unfiltered_.back(), *segmented2_trans);

        pcl::PointCloud<PointT>::Ptr segmented3_trans (new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*cloud_trans_filtered, obj_indices_2_to_filtered_.back(), *segmented3_trans);

        pcl::PointCloud<PointT>::Ptr segmented_eroded_trans (new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*cloud_trans, obj_indices_eroded_to_original_.back(), *segmented_eroded_trans);

        std::stringstream cloud_name;
        cloud_name << "cloud_" << i;
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler0(cloud_trans);
        vis_->addPointCloud(cloud_trans, rgb_handler0, cloud_name.str(), vis_viewpoint_[i * NUM_SUBWINDOWS + 0]);

        size_t subwindow_id=0;

        if (i>0)
        {
            cloud_name << "_search_pts";
            pcl::PointCloud<PointT>::Ptr cloud_trans_tmp (new pcl::PointCloud<PointT>());
            pcl::transformPointCloud(*transferred_cluster_.back(), *cloud_trans_tmp, cameras_.back());
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> red_source (cloud_trans_tmp, 0, 255, 0);
            vis_->addPointCloud(cloud_trans_tmp, red_source, cloud_name.str(), vis_viewpoint_[i * NUM_SUBWINDOWS + subwindow_id++]);
        }
        else
        {
            cloud_name << "_search_pts";
            pcl::PointCloud<PointT>::Ptr cloud_trans_tmp (new pcl::PointCloud<PointT>());
            pcl::PointCloud<PointT>::Ptr obj_trans_tmp (new pcl::PointCloud<PointT>());
            pcl::transformPointCloud(*keyframes_.back(), *cloud_trans_tmp, cameras_.back());
            pcl::copyPointCloud(*cloud_trans_tmp, transferred_nn_points_.back(), *obj_trans_tmp);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> red_source (obj_trans_tmp, 255, 0, 0);
            vis_->addPointCloud(obj_trans_tmp, red_source, cloud_name.str(), vis_viewpoint_[i * NUM_SUBWINDOWS + subwindow_id++]);
        }

        cloud_name << "_filtered";
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler1(cloud_trans_filtered);
        vis_->addPointCloud(cloud_trans_filtered, rgb_handler1, cloud_name.str(), vis_viewpoint_[i * NUM_SUBWINDOWS + subwindow_id++]);

        cloud_name << "_supervoxellized";
        vis_->addPointCloud(sv_trans, cloud_name.str(), vis_viewpoint_[i * NUM_SUBWINDOWS + subwindow_id++]);

        cloud_name << "_nearest_neighbor";
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler2(segmented_trans);
        vis_->addPointCloud(segmented_trans, rgb_handler2, cloud_name.str(), vis_viewpoint_[i * NUM_SUBWINDOWS + subwindow_id++]);

        cloud_name << "_good";
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler3(segmented2_trans);
        vis_->addPointCloud(segmented2_trans, rgb_handler3, cloud_name.str(), vis_viewpoint_[i * NUM_SUBWINDOWS + subwindow_id++]);

        cloud_name << "_region_grown";
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler4(segmented3_trans);
        vis_->addPointCloud(segmented3_trans, rgb_handler4, cloud_name.str(), vis_viewpoint_[i * NUM_SUBWINDOWS + subwindow_id++]);

        cloud_name << "_eroded";
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler5(segmented_eroded_trans);
        vis_->addPointCloud(segmented_eroded_trans, rgb_handler5, cloud_name.str(), vis_viewpoint_[i * NUM_SUBWINDOWS + subwindow_id++]);
    }
#ifndef USE_REMOTE_PCL_VISUALIZER
    vis_->spin();
#endif
}


void DOL::initialize (int argc, char ** argv)
{
    double inlDist, minPoints;

    if (do_sift_based_camera_pose_estimation_)
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

void DOL::printParams(std::ostream &ostr)
{
    ostr << "Started dynamic object learning with parameters: " << std::endl
              << "===================================================" << std::endl
              << "radius: " << radius_ << std::endl
              << "eps_angle: " << eps_angle_ << std::endl
              << "seed resolution: " << seed_resolution_ << std::endl
              << "voxel resolution: " << voxel_resolution_ << std::endl
              << "ratio: " << ratio_ << std::endl
              << "do_erosion: " << do_erosion_ << std::endl
              << "max z distance: " << chop_z_ << std::endl
              << "transferring object indices from latest frame only: " << transfer_indices_from_latest_frame_only_ << std::endl
              << "===================================================" << std::endl << std::endl;
}
