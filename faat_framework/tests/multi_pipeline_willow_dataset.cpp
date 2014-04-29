/*
 * local_recognition_mian_dataset.cpp
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */
#include <pcl/console/parse.h>
#include <faat_pcl/3d_rec_framework/pc_source/partial_pcd_source.h>
#include <faat_pcl/3d_rec_framework/pc_source/registered_views_source.h>
#include <faat_pcl/3d_rec_framework/pipeline/hough_grouping_local_recognizer.h>
#include <faat_pcl/3d_rec_framework/pipeline/local_recognizer.h>
#include <faat_pcl/3d_rec_framework/pipeline/global_nn_recognizer_cvfh.h>
#include <faat_pcl/3d_rec_framework/pipeline/recognizer.h>
#include <faat_pcl/3d_rec_framework/pipeline/multi_pipeline_recognizer.h>
#include <faat_pcl/3d_rec_framework/utils/metrics.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/shot_local_estimator.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/shot_local_estimator_omp.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/fpfh_local_estimator.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/color_ourcvfh_estimator.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/organized_color_ourcvfh_estimator.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/recognition/cg/correspondence_grouping.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <faat_pcl/recognition/cg/graph_geometric_consistency.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/image/sift_local_estimator.h>
#include <faat_pcl/recognition/hv/hv_go_1.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/planar_polygon_fusion.h>
#include <pcl/segmentation/plane_coefficient_comparator.h>
#include <pcl/segmentation/euclidean_plane_coefficient_comparator.h>
#include <pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl/segmentation/edge_aware_plane_comparator.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <faat_pcl/3d_rec_framework/tools/or_evaluator.h>
#include <pcl/filters/fast_bilateral.h>
#include <faat_pcl/3d_rec_framework/segmentation/multiplane_segmentation.h>
#include <pcl/apps/dominant_plane_segmentation.h>
#include <pcl/filters/fast_bilateral.h>
#include <faat_pcl/utils/noise_models.h>
#include <pcl/features/organized_edge_detection.h>
#include <faat_pcl/utils/miscellaneous.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/image/opencv_sift_local_estimator.h>
#include "v4r/SurfaceSegmenter/segmentation.hpp"

float VX_SIZE_ICP_ = 0.005f;
bool PLAY_ = false;
std::string go_log_file_ = "test.txt";
float Z_DIST_ = 1.5f;
std::string GT_DIR_ = "";
std::string MODELS_DIR_;
std::string MODELS_DIR_FOR_VIS_;
float model_scale = 1.f;
bool use_HV = true;
bool use_highest_plane_ = false;

bool segmentation_plane_unorganized_ = false;

//do a segmentation that instead of the table plane, returns all indices that are not planes
template<typename PointT>
void
doSegmentation (typename pcl::PointCloud<PointT>::Ptr & xyz_points,
                pcl::PointCloud<pcl::Normal>::Ptr & normal_cloud,
                std::vector<pcl::PointIndices> & indices,
                std::vector<int> & indices_above_plane,
                bool use_highest_plane,
                int seg = 0)
{
    Eigen::Vector4f table_plane;

    std::cout << "Start segmentation..." << std::endl;
    int min_cluster_size_ = 500;
    int num_plane_inliers = 1000;

    if(seg == 1)
    {
        pcl::apps::DominantPlaneSegmentation<PointT> dps;
        dps.setInputCloud (xyz_points);
        dps.setMaxZBounds (Z_DIST_);
        dps.setObjectMinHeight (0.01);
        dps.setMinClusterSize (min_cluster_size_);
        dps.setWSize (9);
        dps.setDistanceBetweenClusters (0.03f);
        std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
        dps.setDownsamplingSize (0.01f);
        dps.compute_full (clusters);

        std::vector<pcl::PointIndices> indices_clusters;
        dps.getIndicesClusters (indices_clusters);
        dps.getTableCoefficients (table_plane);

        std::cout << table_plane << std::endl;

        for(size_t i=0; i < xyz_points->points.size(); i++)
        {
            Eigen::Vector3f xyz_p = xyz_points->points[i].getVector3fMap();
            if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
                continue;

            float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];
            if(val > 0.01f)
            {
                indices_above_plane.push_back(i);
            }
        }

        std::cout << indices_above_plane.size() << std::endl;

        return;
    }

    pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
    mps.setMinInliers (num_plane_inliers);
    mps.setAngularThreshold (0.017453 * 2.f); // 2 degrees
    mps.setDistanceThreshold (0.01); // 1cm
    mps.setInputNormals (normal_cloud);
    mps.setInputCloud (xyz_points);

    std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
    std::vector<pcl::ModelCoefficients> model_coefficients;
    std::vector<pcl::PointIndices> inlier_indices;
    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
    std::vector<pcl::PointIndices> label_indices;
    std::vector<pcl::PointIndices> boundary_indices;
    std::vector<bool> plane_labels;

    typename pcl::PlaneRefinementComparator<PointT, pcl::Normal, pcl::Label>::Ptr ref_comp (
                new pcl::PlaneRefinementComparator<PointT,
                pcl::Normal, pcl::Label> ());
    ref_comp->setDistanceThreshold (0.01f, false);
    ref_comp->setAngularThreshold (0.017453 * 2);
    mps.setRefinementComparator (ref_comp);
    mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);

    std::cout << "Number of planes found:" << model_coefficients.size () << std::endl;
    if(model_coefficients.size() == 0)
        return;

    if(use_highest_plane)
    {
        int table_plane_selected = 0;
        int max_inliers_found = -1;
        std::vector<int> plane_inliers_counts;
        plane_inliers_counts.resize (model_coefficients.size ());

        for (size_t i = 0; i < model_coefficients.size (); i++)
        {
            Eigen::Vector4f table_plane = Eigen::Vector4f (model_coefficients[i].values[0], model_coefficients[i].values[1],
                                                           model_coefficients[i].values[2], model_coefficients[i].values[3]);

            std::cout << "Number of inliers for this plane:" << inlier_indices[i].indices.size () << std::endl;
            int remaining_points = 0;
            typename pcl::PointCloud<PointT>::Ptr plane_points (new pcl::PointCloud<PointT> (*xyz_points));
            for (int j = 0; j < plane_points->points.size (); j++)
            {
                Eigen::Vector3f xyz_p = plane_points->points[j].getVector3fMap ();

                if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
                    continue;

                float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

                if (std::abs (val) > 0.01)
                {
                    plane_points->points[j].x = std::numeric_limits<float>::quiet_NaN ();
                    plane_points->points[j].y = std::numeric_limits<float>::quiet_NaN ();
                    plane_points->points[j].z = std::numeric_limits<float>::quiet_NaN ();
                }
                else
                    remaining_points++;
            }

            plane_inliers_counts[i] = remaining_points;

            if (remaining_points > max_inliers_found)
            {
                table_plane_selected = i;
                max_inliers_found = remaining_points;
            }
        }

        size_t itt = static_cast<size_t> (table_plane_selected);
        table_plane = Eigen::Vector4f (model_coefficients[itt].values[0], model_coefficients[itt].values[1],
                                       model_coefficients[itt].values[2], model_coefficients[itt].values[3]);

        Eigen::Vector3f normal_table = Eigen::Vector3f (model_coefficients[itt].values[0], model_coefficients[itt].values[1],
                                                        model_coefficients[itt].values[2]);

        int inliers_count_best = plane_inliers_counts[itt];

        //check that the other planes with similar normal are not higher than the table_plane_selected
        for (size_t i = 0; i < model_coefficients.size (); i++)
        {
            Eigen::Vector4f model = Eigen::Vector4f (model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2],
                                                     model_coefficients[i].values[3]);

            Eigen::Vector3f normal = Eigen::Vector3f (model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2]);

            int inliers_count = plane_inliers_counts[i];

            std::cout << "Dot product is:" << normal.dot (normal_table) << std::endl;
            if ((normal.dot (normal_table) > 0.95) && (inliers_count_best * 0.5 <= inliers_count))
            {
                //check if this plane is higher, projecting a point on the normal direction
                std::cout << "Check if plane is higher, then change table plane" << std::endl;
                std::cout << model[3] << " " << table_plane[3] << std::endl;
                if (model[3] < table_plane[3])
                {
                    PCL_WARN ("Changing table plane...");
                    table_plane_selected = i;
                    table_plane = model;
                    normal_table = normal;
                    inliers_count_best = inliers_count;
                }
            }
        }

        table_plane = Eigen::Vector4f (model_coefficients[table_plane_selected].values[0], model_coefficients[table_plane_selected].values[1],
                                       model_coefficients[table_plane_selected].values[2], model_coefficients[table_plane_selected].values[3]);

        label_indices.resize(2);
        //create two labels, 1 one for points belonging to or under the plane, 1 for points above the plane
        for (int j = 0; j < xyz_points->points.size (); j++)
        {
            Eigen::Vector3f xyz_p = xyz_points->points[j].getVector3fMap ();

            if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
                continue;

            float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

            if (val >= 0.01f) //object
            {
                labels->points[j].label = 1;
                label_indices[1].indices.push_back(j);
            }
            else //plane or below
            {
                labels->points[j].label = 0;
                label_indices[0].indices.push_back(j);
            }
        }

        plane_labels.resize (2, false);
        plane_labels[0] = true;
    }
    else
    {
        label_indices.resize(inlier_indices.size() + 1);
        plane_labels.resize (inlier_indices.size () + 1, false);

        for (int j = 0; j < xyz_points->points.size (); j++)
        {
            labels->points[j].label = 0;
        }

        //filter out all big planes
        int max_plane_inliers = 15000;
        int l=1;
        for (size_t i = 0; i < inlier_indices.size (); i++,l++)
        {

            if(inlier_indices[i].indices.size() > max_plane_inliers)
            {
                //its a big plane
                plane_labels[l] = true;
            }
            else
            {
                //its a small plane (potentially an object)
                plane_labels[l] = false;
            }

            for (size_t j = 0; j < inlier_indices[i].indices.size (); j++)
            {
                labels->points[inlier_indices[i].indices[j]].label = l;
                label_indices[l].indices.push_back(inlier_indices[i].indices[j]);
            }
        }

        for(size_t j=0; j < labels->points.size(); j++)
        {
            if(labels->points[j].label == 0)
            {
                label_indices[0].indices.push_back(j);
            }
        }
    }

    if(seg == 0) //connected component segmentation
    {
        //cluster..
        typename pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr
                euclidean_cluster_comparator_ (
                    new pcl::EuclideanClusterComparator<
                    PointT,
                    pcl::Normal,
                    pcl::Label> ());

        euclidean_cluster_comparator_->setInputCloud (xyz_points);
        euclidean_cluster_comparator_->setLabels (labels);
        euclidean_cluster_comparator_->setExcludeLabels (plane_labels);
        euclidean_cluster_comparator_->setDistanceThreshold (0.035f, true);

        pcl::PointCloud<pcl::Label> euclidean_labels;
        std::vector<pcl::PointIndices> euclidean_label_indices;
        pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label> euclidean_segmentation (euclidean_cluster_comparator_);
        euclidean_segmentation.setInputCloud (xyz_points);
        euclidean_segmentation.segment (euclidean_labels, euclidean_label_indices);

        for (size_t i = 0; i < euclidean_label_indices.size (); i++)
        {
            if (euclidean_label_indices[i].indices.size () >= min_cluster_size_)
            {
                indices.push_back (euclidean_label_indices[i]);
                indices_above_plane.insert(indices_above_plane.end(), euclidean_label_indices[i].indices.begin(),
                                                                      euclidean_label_indices[i].indices.end());
            }
        }
    }
    else if(seg == 2)
    {

        //use label_indices and plane_labels to set points to NaN
        boost::shared_ptr<segmentation::Segmenter> segmenter_;

        typename pcl::PointCloud<PointT>::Ptr scene_cloud(new pcl::PointCloud<PointT>(*xyz_points));
        for(size_t i=0; i < xyz_points->points.size(); i++)
        {
            if(xyz_points->points[i].z > Z_DIST_)
            {
                scene_cloud->points[i].x = std::numeric_limits<float>::quiet_NaN ();
                scene_cloud->points[i].y = std::numeric_limits<float>::quiet_NaN ();
                scene_cloud->points[i].z = std::numeric_limits<float>::quiet_NaN ();
            }
        }

        for(size_t i=0; i < plane_labels.size(); i++)
        {
            if(!plane_labels[i])
                continue; //is not a plane, do nothing

            //if the label belongs to a plane, set indices to NaN
            for(size_t k=0; k < label_indices[i].indices.size(); k++)
            {
                scene_cloud->points[label_indices[i].indices[k]].x
                        = scene_cloud->points[label_indices[i].indices[k]].y
                        = scene_cloud->points[label_indices[i].indices[k]].z
                        = std::numeric_limits<float>::quiet_NaN ();
            }
        }


        segmenter_.reset(new segmentation::Segmenter);
        segmenter_->setModelFilename("data_rgbd_segmenter/ST-TrainAll.model.txt");
        segmenter_->setScaling("data_rgbd_segmenter/ST-TrainAll.scalingparams.txt");
        segmenter_->setUsePlanesNotNurbs(false);
        segmenter_->setPointCloud(scene_cloud);
        segmenter_->segment();

        std::vector<std::vector<int> > clusters = segmenter_->getSegmentedObjectsIndices();

        indices_above_plane.resize(0);
        for (size_t i = 0; i < clusters.size (); i++)
        {
            if (clusters[i].size () >= min_cluster_size_)
            {
                pcl::PointIndices indx;
                indx.indices = clusters[i];
                indices.push_back (indx);

                indices_above_plane.insert(indices_above_plane.end(), clusters[i].begin(), clusters[i].end());
            }
        }
    }
}

//do a segmentation that instead of the table plane, returns all indices that are not planes
template<typename PointT>
void
doSegmentation (typename pcl::PointCloud<PointT>::Ptr & xyz_points,
                pcl::PointCloud<pcl::Normal>::Ptr & normal_cloud,
                std::vector<pcl::PointIndices> & indices,
                std::vector<int> & indices_above_plane)
{
    Eigen::Vector4f table_plane;

    std::cout << "Start segmentation..." << std::endl;
    int min_cluster_size_ = 500;

    pcl::apps::DominantPlaneSegmentation<PointT> dps;
    dps.setInputCloud (xyz_points);
    dps.setMaxZBounds (Z_DIST_);
    dps.setObjectMinHeight (0.01);
    dps.setMinClusterSize (min_cluster_size_);
    dps.setWSize (9);
    dps.setDistanceBetweenClusters (0.03f);
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    dps.setDownsamplingSize (0.01f);
    dps.compute_full (clusters);

    std::vector<pcl::PointIndices> indices_clusters;
    dps.getIndicesClusters (indices_clusters);
    dps.getTableCoefficients (table_plane);

    std::cout << table_plane << std::endl;

    std::vector<int> below_plane;

    for(size_t i=0; i < xyz_points->points.size(); i++)
    {
        Eigen::Vector3f xyz_p = xyz_points->points[i].getVector3fMap();
        if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
            continue;

        float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];
        if(val > 0.01f)
        {
            indices_above_plane.push_back(i);
        }
        else
        {
            below_plane.push_back(i);
        }
    }

    std::cout << indices_above_plane.size() << std::endl;

    //use label_indices and plane_labels to set points to NaN
    boost::shared_ptr<segmentation::Segmenter> segmenter_;

    typename pcl::PointCloud<PointT>::Ptr scene_cloud(new pcl::PointCloud<PointT>(*xyz_points));
    for(size_t i=0; i < xyz_points->points.size(); i++)
    {
        if(xyz_points->points[i].z > Z_DIST_)
        {
            scene_cloud->points[i].x = std::numeric_limits<float>::quiet_NaN ();
            scene_cloud->points[i].y = std::numeric_limits<float>::quiet_NaN ();
            scene_cloud->points[i].z = std::numeric_limits<float>::quiet_NaN ();
        }
    }

    for(size_t i=0; i < below_plane.size(); i++)
    {
            scene_cloud->points[below_plane[i]].x
                    = scene_cloud->points[below_plane[i]].y
                    = scene_cloud->points[below_plane[i]].z
                    = std::numeric_limits<float>::quiet_NaN ();
    }


    segmenter_.reset(new segmentation::Segmenter);
    segmenter_->setModelFilename("data_rgbd_segmenter/ST-TrainAll.model.txt");
    segmenter_->setScaling("data_rgbd_segmenter/ST-TrainAll.scalingparams.txt");
    segmenter_->setUsePlanesNotNurbs(false);
    segmenter_->setPointCloud(scene_cloud);
    segmenter_->segment();

    {
        std::vector<std::vector<int> > clusters = segmenter_->getSegmentedObjectsIndices();

        for (size_t i = 0; i < clusters.size (); i++)
        {
            if (clusters[i].size () >= min_cluster_size_)
            {
                pcl::PointIndices indx;
                indx.indices = clusters[i];
                indices.push_back (indx);
            }
        }
    }
}

template<typename PointT>
void
doSegmentation (typename pcl::PointCloud<PointT>::Ptr & xyz_points,
                pcl::PointCloud<pcl::Normal>::Ptr & normal_cloud,
                std::vector<pcl::PointIndices> & indices,
                Eigen::Vector4f & table_plane,
                int seg = 0)
{
    std::cout << "Start segmentation..." << std::endl;
    int min_cluster_size_ = 500;

    if(seg == 0)
    {

        int num_plane_inliers = 1000;

        pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
        mps.setMinInliers (num_plane_inliers);
        mps.setAngularThreshold (0.017453 * 2.f); // 2 degrees
        mps.setDistanceThreshold (0.01); // 1cm
        mps.setInputNormals (normal_cloud);
        mps.setInputCloud (xyz_points);

        std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
        std::vector<pcl::ModelCoefficients> model_coefficients;
        std::vector<pcl::PointIndices> inlier_indices;
        pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
        std::vector<pcl::PointIndices> label_indices;
        std::vector<pcl::PointIndices> boundary_indices;

        typename pcl::PlaneRefinementComparator<PointT, pcl::Normal, pcl::Label>::Ptr ref_comp (
                    new pcl::PlaneRefinementComparator<PointT,
                    pcl::Normal, pcl::Label> ());
        ref_comp->setDistanceThreshold (0.01f, false);
        ref_comp->setAngularThreshold (0.017453 * 2);
        mps.setRefinementComparator (ref_comp);
        mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);

        std::cout << "Number of planes found:" << model_coefficients.size () << std::endl;
        if(model_coefficients.size() == 0)
            return;

        int table_plane_selected = 0;
        int max_inliers_found = -1;
        std::vector<int> plane_inliers_counts;
        plane_inliers_counts.resize (model_coefficients.size ());

        for (size_t i = 0; i < model_coefficients.size (); i++)
        {
            Eigen::Vector4f table_plane = Eigen::Vector4f (model_coefficients[i].values[0], model_coefficients[i].values[1],
                                                           model_coefficients[i].values[2], model_coefficients[i].values[3]);

            std::cout << "Number of inliers for this plane:" << inlier_indices[i].indices.size () << std::endl;
            int remaining_points = 0;
            typename pcl::PointCloud<PointT>::Ptr plane_points (new pcl::PointCloud<PointT> (*xyz_points));
            for (int j = 0; j < plane_points->points.size (); j++)
            {
                Eigen::Vector3f xyz_p = plane_points->points[j].getVector3fMap ();

                if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
                    continue;

                float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

                if (std::abs (val) > 0.01)
                {
                    plane_points->points[j].x = std::numeric_limits<float>::quiet_NaN ();
                    plane_points->points[j].y = std::numeric_limits<float>::quiet_NaN ();
                    plane_points->points[j].z = std::numeric_limits<float>::quiet_NaN ();
                }
                else
                    remaining_points++;
            }

            plane_inliers_counts[i] = remaining_points;

            if (remaining_points > max_inliers_found)
            {
                table_plane_selected = i;
                max_inliers_found = remaining_points;
            }
        }

        size_t itt = static_cast<size_t> (table_plane_selected);
        table_plane = Eigen::Vector4f (model_coefficients[itt].values[0], model_coefficients[itt].values[1],
                                       model_coefficients[itt].values[2], model_coefficients[itt].values[3]);

        Eigen::Vector3f normal_table = Eigen::Vector3f (model_coefficients[itt].values[0], model_coefficients[itt].values[1],
                                                        model_coefficients[itt].values[2]);

        int inliers_count_best = plane_inliers_counts[itt];

        //check that the other planes with similar normal are not higher than the table_plane_selected
        for (size_t i = 0; i < model_coefficients.size (); i++)
        {
            Eigen::Vector4f model = Eigen::Vector4f (model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2],
                                                     model_coefficients[i].values[3]);

            Eigen::Vector3f normal = Eigen::Vector3f (model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2]);

            int inliers_count = plane_inliers_counts[i];

            std::cout << "Dot product is:" << normal.dot (normal_table) << std::endl;
            if ((normal.dot (normal_table) > 0.95) && (inliers_count_best * 0.5 <= inliers_count))
            {
                //check if this plane is higher, projecting a point on the normal direction
                std::cout << "Check if plane is higher, then change table plane" << std::endl;
                std::cout << model[3] << " " << table_plane[3] << std::endl;
                if (model[3] < table_plane[3])
                {
                    PCL_WARN ("Changing table plane...");
                    table_plane_selected = i;
                    table_plane = model;
                    normal_table = normal;
                    inliers_count_best = inliers_count;
                }
            }
        }

        table_plane = Eigen::Vector4f (model_coefficients[table_plane_selected].values[0], model_coefficients[table_plane_selected].values[1],
                                       model_coefficients[table_plane_selected].values[2], model_coefficients[table_plane_selected].values[3]);

        //cluster..
        typename pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr
                euclidean_cluster_comparator_ (
                    new pcl::EuclideanClusterComparator<
                    PointT,
                    pcl::Normal,
                    pcl::Label> ());

        //create two labels, 1 one for points belonging to or under the plane, 1 for points above the plane
        label_indices.resize (2);

        for (int j = 0; j < xyz_points->points.size (); j++)
        {
            Eigen::Vector3f xyz_p = xyz_points->points[j].getVector3fMap ();

            if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
                continue;

            float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

            if (val >= 0.01f)
            {
                /*plane_points->points[j].x = std::numeric_limits<float>::quiet_NaN ();
         plane_points->points[j].y = std::numeric_limits<float>::quiet_NaN ();
         plane_points->points[j].z = std::numeric_limits<float>::quiet_NaN ();*/
                labels->points[j].label = 1;
                label_indices[0].indices.push_back (j);
            }
            else
            {
                labels->points[j].label = 0;
                label_indices[1].indices.push_back (j);
            }
        }

        std::vector<bool> plane_labels;
        plane_labels.resize (label_indices.size (), false);
        plane_labels[0] = true;

        euclidean_cluster_comparator_->setInputCloud (xyz_points);
        euclidean_cluster_comparator_->setLabels (labels);
        euclidean_cluster_comparator_->setExcludeLabels (plane_labels);
        euclidean_cluster_comparator_->setDistanceThreshold (0.035f, true);

        pcl::PointCloud<pcl::Label> euclidean_labels;
        std::vector<pcl::PointIndices> euclidean_label_indices;
        pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label> euclidean_segmentation (euclidean_cluster_comparator_);
        euclidean_segmentation.setInputCloud (xyz_points);
        euclidean_segmentation.segment (euclidean_labels, euclidean_label_indices);

        for (size_t i = 0; i < euclidean_label_indices.size (); i++)
        {
            if (euclidean_label_indices[i].indices.size () >= min_cluster_size_)
            {
                indices.push_back (euclidean_label_indices[i]);
            }
        }
    }
    else if(seg == 1)
    {
        pcl::apps::DominantPlaneSegmentation<PointT> dps;
        dps.setInputCloud (xyz_points);
        dps.setMaxZBounds (Z_DIST_);
        dps.setObjectMinHeight (0.01);
        dps.setMinClusterSize (min_cluster_size_);
        dps.setWSize (9);
        dps.setDistanceBetweenClusters (0.03f);
        std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
        dps.setDownsamplingSize (0.01f);
        dps.compute_fast (clusters);
        dps.getIndicesClusters (indices);
        dps.getTableCoefficients (table_plane);
    }
    else if(seg == 2)
    {

        boost::shared_ptr<segmentation::Segmenter> segmenter_;

        typename pcl::PointCloud<PointT>::Ptr scene_cloud(new pcl::PointCloud<PointT>(*xyz_points));
        for(size_t i=0; i < xyz_points->points.size(); i++)
        {
            if(xyz_points->points[i].z > Z_DIST_)
            {
                scene_cloud->points[i].x = std::numeric_limits<float>::quiet_NaN ();
                scene_cloud->points[i].y = std::numeric_limits<float>::quiet_NaN ();
                scene_cloud->points[i].z = std::numeric_limits<float>::quiet_NaN ();
            }
        }


        segmenter_.reset(new segmentation::Segmenter);
        segmenter_->setModelFilename("data_rgbd_segmenter/ST-TrainAll.model.txt");
        segmenter_->setScaling("data_rgbd_segmenter/ST-TrainAll.scalingparams.txt");
        segmenter_->setUsePlanesNotNurbs(false);
        segmenter_->setPointCloud(scene_cloud);
        segmenter_->segment();

        std::vector<std::vector<int> > clusters = segmenter_->getSegmentedObjectsIndices();

        for (size_t i = 0; i < clusters.size (); i++)
        {
            if (clusters[i].size () >= min_cluster_size_)
            {
                pcl::PointIndices indx;
                indx.indices = clusters[i];
                indices.push_back (indx);
            }
        }
    }
}

void
getModelsInDirectory (bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext)
{
    bf::directory_iterator end_itr;
    for (bf::directory_iterator itr (dir); itr != end_itr; ++itr)
    {
        //check if its a directory, then get models in it
        if (bf::is_directory (*itr))
        {
#if BOOST_FILESYSTEM_VERSION == 3
            std::string so_far = rel_path_so_far + (itr->path ().filename ()).string() + "/";
#else
            std::string so_far = rel_path_so_far + (itr->path ()).filename () + "/";
#endif

            bf::path curr_path = itr->path ();
            getModelsInDirectory (curr_path, so_far, relative_paths, ext);
        }
        else
        {
            //check that it is a ply file and then add, otherwise ignore..
            std::vector < std::string > strs;
#if BOOST_FILESYSTEM_VERSION == 3
            std::string file = (itr->path ().filename ()).string();
#else
            std::string file = (itr->path ()).filename ();
#endif

            boost::split (strs, file, boost::is_any_of ("."));
            std::string extension = strs[strs.size () - 1];

            if (extension.compare (ext) == 0)
            {
#if BOOST_FILESYSTEM_VERSION == 3
                std::string path = rel_path_so_far + (itr->path ().filename ()).string();
#else
                std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif

                relative_paths.push_back (path);
            }
        }
    }
}

class go_params {
public:
    float go_resolution;
    float go_iterations;
    float go_inlier_thres;
    float radius_clutter;
    float regularizer;
    float clutter_regularizer;
    bool go_use_replace_moves;
    int go_opt_type;
    float init_temp;
    float radius_normals_go_;
    float require_normals;
    bool go_init;
    bool detect_clutter;
    bool use_super_voxels_;
    float color_sigma_;

    void writeParamsToFile(std::ofstream & of) {
        of << "Params: \t" << go_opt_type << "\t";
        of << static_cast<int>(go_init) << "\t";
        of << radius_clutter << "\t" << clutter_regularizer << "\t";
        of << radius_normals_go_ << "\t" << static_cast<int>(go_use_replace_moves);
        of << std::endl;
    }
};

go_params parameters_for_go;
bool BILATERAL_FILTER_ = false;
std::string STATISTIC_OUTPUT_FILE_ = "stats.txt";
std::string POSE_STATISTICS_OUTPUT_FILE_ = "pose_translation.txt";
std::string POSE_STATISTICS_ANGLE_OUTPUT_FILE_ = "pose_error.txt";
bool use_histogram_specification_ = false;
bool use_table_plane_ = true;
bool non_organized_planes_ = true;
std::string RESULTS_OUTPUT_DIR_ = "";

template<typename PointT>
void
recognizeAndVisualize (typename boost::shared_ptr<faat_pcl::rec_3d_framework::MultiRecognitionPipeline<PointT> > & local,
                       std::string & scene_file, int seg, bool add_planes=true)
{


    bool gt_available = true;
    if(GT_DIR_.compare("") == 0)
    {
        gt_available = false;
    }

    faat_pcl::rec_3d_framework::or_evaluator::OREvaluator<PointT> or_eval;

    if(gt_available)
    {
        or_eval.setGTDir(GT_DIR_);
        or_eval.setModelsDir(MODELS_DIR_);
        or_eval.setModelFileExtension("pcd");
        or_eval.setReplaceModelExtension(false);
        or_eval.useMaxOcclusion(false);
        or_eval.setMaxOcclusion(0.9f);
        or_eval.setCheckPose(true);
        or_eval.setMaxCentroidDistance(0.03f);
    }

    boost::shared_ptr<faat_pcl::GlobalHypothesesVerification_1<PointT, PointT> > go (
                new faat_pcl::GlobalHypothesesVerification_1<PointT,
                PointT>);
    go->setSmoothSegParameters(0.1, 0.035, 0.005);
    //go->setRadiusNormals(0.03f);
    go->setResolution (parameters_for_go.go_resolution);
    go->setMaxIterations (parameters_for_go.go_iterations);
    go->setInlierThreshold (parameters_for_go.go_inlier_thres);
    go->setRadiusClutter (parameters_for_go.radius_clutter);
    go->setRegularizer (parameters_for_go.regularizer);
    go->setClutterRegularizer (parameters_for_go.clutter_regularizer);
    go->setDetectClutter (parameters_for_go.detect_clutter);
    go->setOcclusionThreshold (0.01f);
    go->setOptimizerType(parameters_for_go.go_opt_type);
    go->setUseReplaceMoves(parameters_for_go.go_use_replace_moves);
    go->setInitialTemp(parameters_for_go.init_temp);
    go->setRadiusNormals(parameters_for_go.radius_normals_go_);
    go->setRequiresNormals(parameters_for_go.require_normals);
    go->setInitialStatus(parameters_for_go.go_init);
    go->setIgnoreColor(false);
    go->setColorSigma(parameters_for_go.color_sigma_);
    go->setUseSuperVoxels(parameters_for_go.use_super_voxels_);
    go->setHistogramSpecification(use_histogram_specification_);

    boost::shared_ptr<faat_pcl::HypothesisVerification<PointT, PointT> > cast_hv_alg;
    cast_hv_alg = boost::static_pointer_cast<faat_pcl::HypothesisVerification<PointT, PointT> > (go);

    typename boost::shared_ptr<faat_pcl::rec_3d_framework::Source<PointT> > model_source_ = local->getDataSource ();
    typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointInTPtr;
    typedef faat_pcl::rec_3d_framework::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    local->setVoxelSizeICP (VX_SIZE_ICP_);

    pcl::visualization::PCLVisualizer vis ("Recognition results");
    int v1, v2, v3, v4, v5, v6;
    vis.createViewPort (0.0, 0.0, 0.33, 0.5, v1);
    vis.createViewPort (0.33, 0, 0.66, 0.5, v2);
    vis.createViewPort (0.0, 0.5, 0.33, 1, v3);
    vis.createViewPort (0.33, 0.5, 0.66, 1, v4);
    vis.createViewPort (0.66, 0.5, 1, 1, v6);
    vis.createViewPort (0.66, 0, 1, 0.5, v5);

    vis.addText ("go segmentation", 1, 30, 18, 1, 0, 0, "go_smooth", v3);
    vis.addText ("Ground truth", 1, 30, 18, 1, 0, 0, "gt_text", v4);
    vis.addText ("Scene", 1, 30, 18, 1, 0, 0, "scene_texttt", v5);
    vis.addText ("Hypotheses", 1, 30, 18, 1, 0, 0, "hypotheses_text", v6);
    vis.addText ("Final Results", 1, 30, 18, 1, 0, 0, "final_res_text", v2);

    bf::path input = scene_file;
    std::vector<std::string> files_to_recognize;

    if (bf::is_directory (input))
    {
        std::vector < std::string > files;
        std::string start = "";
        std::string ext = std::string ("pcd");
        bf::path dir = input;
        getModelsInDirectory (dir, start, files, ext);
        std::cout << "Number of scenes in directory is:" << files.size () << std::endl;
        for (size_t i = 0; i < files.size (); i++)
        {
            typename pcl::PointCloud<PointT>::Ptr scene_cloud (new pcl::PointCloud<PointT>);
            std::cout << files[i] << std::endl;
            std::stringstream filestr;
            filestr << scene_file << files[i];
            std::string file = filestr.str ();
            files_to_recognize.push_back (file);
        }

        std::sort(files_to_recognize.begin(),files_to_recognize.end());

        if(gt_available)
        {
            or_eval.setScenesDir(scene_file);
            or_eval.setDataSource(local->getDataSource());
            or_eval.loadGTData();
        }
    }
    else
    {
        files_to_recognize.push_back (scene_file);
    }

    std::cout << "is segmentation required:" << local->isSegmentationRequired() << std::endl;

    double time_in_seconds = 0;
    pcl::StopWatch total_time;
    for(size_t i=0; i < files_to_recognize.size(); i++)
    {
        pcl::ScopeTime t_scene("scene, including loading and visualization...");
        std::stringstream screenshot_name_str;
        screenshot_name_str << "screenshots/screenshot_" << std::setw(8) << std::setfill('0') << i << ".png";
        std::string screenshot_name = screenshot_name_str.str();

        std::cout << "recognizing " << files_to_recognize[i] << std::endl;
        typename pcl::PointCloud<PointT>::Ptr scene (new pcl::PointCloud<PointT>);
        pcl::io::loadPCDFile (files_to_recognize[i], *scene);

        typename pcl::PointCloud<PointT>::Ptr occlusion_cloud (new pcl::PointCloud<PointT>(*scene));

        std::string file_to_recognize(files_to_recognize[i]);
        boost::replace_all (file_to_recognize, scene_file, "");
        boost::replace_all (file_to_recognize, ".pcd", "");

        std::string id_1 = file_to_recognize;

        std::cout << "Scene is:" << id_1 << std::endl;
        if(Z_DIST_ > 0)
        {
            pcl::PassThrough<PointT> pass_;
            pass_.setFilterLimits (0.f, Z_DIST_);
            pass_.setFilterFieldName ("z");
            pass_.setInputCloud (scene);
            pass_.setKeepOrganized (true);
            pass_.filter (*scene);
        }

        pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
        ne.setRadiusSearch(0.02f);
        ne.setInputCloud (scene);
        ne.compute (*normal_cloud);

        //Multiplane segmentation
        faat_pcl::MultiPlaneSegmentation<PointT> mps;
        mps.setInputCloud(scene);
        mps.setMinPlaneInliers(1000);
        mps.setResolution(parameters_for_go.go_resolution);
        mps.setNormals(normal_cloud);
        mps.setMergePlanes(true);
        std::vector<faat_pcl::PlaneModel<PointT> > planes_found;
        mps.segment();
        planes_found = mps.getModels();

        if(planes_found.size() == 0 && scene->isOrganized() && non_organized_planes_)
        {
            PCL_WARN("No planes found, doing segmentation with standard method\n");
            mps.segment(true);
            planes_found = mps.getModels();
        }

        std::vector<pcl::PointIndices> indices;
        //Eigen::Vector4f table_plane;
        std::vector<int> indices_above_plane;

        /*if(use_table_plane_)
        {
            doSegmentation<PointT>(scene, normal_cloud, indices, table_plane, seg);

            for(size_t k=0; k < indices.size(); k++)
            {
                std::stringstream cname;
                cname << "cluster_" << k << ".pcd";
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::copyPointCloud(*scene, *cluster);

                std::vector<bool> negative_indices(scene->points.size(), true);
                for(size_t i=0; i < indices[k].indices.size(); i++)
                {
                    negative_indices[indices[k].indices[i]] = false;
                }

                for(size_t i=0; i < cluster->points.size(); i++)
                {
                    if(negative_indices[i])
                    {
                        cluster->points[i].x = cluster->points[i].z = cluster->points[i].y = std::numeric_limits<float>::quiet_NaN();
                    }
                }

                pcl::io::savePCDFileBinary(cname.str(), *cluster);
            }

            //use table plane to define indices for the local pipeline as well...

            {
                for (int k = 0; k < scene->points.size (); k++)
                {
                    Eigen::Vector3f xyz_p = scene->points[k].getVector3fMap ();

                    if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
                        continue;

                    float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

                    if (val >= 0.01)
                    {
                        indices_above_plane.push_back (static_cast<int> (k));
                    }
                }
            }

            local->setSegmentation(indices);
            local->setIndices(indices_above_plane);
        }*/

        if(use_table_plane_)
        {

            if(segmentation_plane_unorganized_)
            {
                indices_above_plane.resize(0);
                doSegmentation<PointT>(scene, normal_cloud, indices, indices_above_plane);
            }
            else
            {
                indices_above_plane.resize(0);
                doSegmentation<PointT>(scene, normal_cloud, indices, indices_above_plane, use_highest_plane_, seg);
            }

            std::cout << indices_above_plane.size() << std::endl;
            local->setSegmentation(indices);
            local->setIndices(indices_above_plane);

            /*pcl::visualization::PCLVisualizer vis("above plane");
            vis.addPointCloud(scene, "scene");

            typename pcl::PointCloud<PointT>::Ptr above_plane_cloud(new pcl::PointCloud<PointT>);
            pcl::copyPointCloud(*scene, indices_above_plane, *above_plane_cloud);

            pcl::visualization::PointCloudColorHandlerCustom<PointT> random_handler (above_plane_cloud, 255, 255, 0);
            vis.addPointCloud<PointT> (above_plane_cloud, random_handler, "above plane");
            vis.spin();*/
        }

        local->setSceneNormals(normal_cloud);
        local->setInputCloud (scene);
        {
            pcl::ScopeTime ttt ("Recognition");
            local->recognize ();
        }

        //HV
        //transforms models
        boost::shared_ptr < std::vector<ModelTPtr> > models = local->getModels ();
        boost::shared_ptr < std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms = local->getTransforms ();

        if(use_HV)
        {
            std::vector<typename pcl::PointCloud<PointT>::ConstPtr> aligned_models;
            std::vector<typename pcl::PointCloud<pcl::Normal>::ConstPtr> aligned_normals;
            aligned_models.resize (models->size ());
            aligned_normals.resize (models->size ());

            std::vector<std::string> model_ids;
            for (size_t kk = 0; kk < models->size (); kk++)
            {
                ConstPointInTPtr model_cloud = models->at (kk)->getAssembled (parameters_for_go.go_resolution);
                pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud = models->at (kk)->getNormalsAssembled (parameters_for_go.go_resolution);

                typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
                pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (kk));
                aligned_models[kk] = model_aligned;

                typename pcl::PointCloud<pcl::Normal>::Ptr normal_aligned (new pcl::PointCloud<pcl::Normal>);
                faat_pcl::utils::miscellaneous::transformNormals(normal_cloud, normal_aligned, transforms->at (kk));
                aligned_normals[kk] = normal_aligned;
                model_ids.push_back(models->at (kk)->id_);
            }

            std::vector<bool> mask_hv;
            if(use_HV && model_ids.size() > 0)
            {
                //compute edges
                //compute depth discontinuity edges
                pcl::OrganizedEdgeBase<PointT, pcl::Label> oed;
                oed.setDepthDisconThreshold (0.02f); //at 1m, adapted linearly with depth
                oed.setMaxSearchNeighbors(100);
                oed.setEdgeType (pcl::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_OCCLUDING
                | pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDED
                | pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_NAN_BOUNDARY
                );
                oed.setInputCloud (occlusion_cloud);

                pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
                std::vector<pcl::PointIndices> indices2;
                oed.compute (*labels, indices2);

                pcl::PointCloud<pcl::PointXYZ>::Ptr occ_edges_full(new pcl::PointCloud<pcl::PointXYZ>);
                occ_edges_full->points.resize(occlusion_cloud->points.size());
                occ_edges_full->width = occlusion_cloud->width;
                occ_edges_full->height = occlusion_cloud->height;
                occ_edges_full->is_dense = occlusion_cloud->is_dense;

                for(size_t ik=0; ik < occ_edges_full->points.size(); ik++)
                {
                    occ_edges_full->points[ik].x =
                    occ_edges_full->points[ik].y =
                    occ_edges_full->points[ik].z = std::numeric_limits<float>::quiet_NaN();
                }

                for (size_t j = 0; j < indices2.size (); j++)
                {
                  for (size_t i = 0; i < indices2[j].indices.size (); i++)
                  {
                    occ_edges_full->points[indices2[j].indices[i]].getVector3fMap() = occlusion_cloud->points[indices2[j].indices[i]].getVector3fMap();
                  }
                }

                go->setOcclusionEdges(occ_edges_full);
                go->setSceneCloud (scene);
                go->setNormalsForClutterTerm(normal_cloud);
                go->setOcclusionCloud (occlusion_cloud);

                //addModels
                go->setRequiresNormals(true);
                go->addNormalsClouds(aligned_normals);
                go->addModels (aligned_models, true);

                std::cout << "normal and models size:" << aligned_models.size() << " " << aligned_normals.size() << std::endl;
                //append planar models
                if(add_planes)
                {
                    go->addPlanarModels(planes_found);
                    for(size_t kk=0; kk < planes_found.size(); kk++)
                    {
                        std::stringstream plane_id;
                        plane_id << "plane_" << kk;
                        model_ids.push_back(plane_id.str());
                    }
                }

                go->setObjectIds(model_ids);
                //verify
                {
                    pcl::ScopeTime t("Go verify");
                    go->verify ();
                }
                go->getMask (mask_hv);
            }
            else
            {
                mask_hv.resize(aligned_models.size(), true);
            }

            std::vector<int> coming_from;
            coming_from.resize(aligned_models.size() + planes_found.size());
            for(size_t j=0; j < aligned_models.size(); j++)
                coming_from[j] = 0;

            for(size_t j=0; j < planes_found.size(); j++)
                coming_from[aligned_models.size() + j] = 1;

            //clear last round from the visualizer
            vis.removePointCloud ("scene_cloud");
            vis.removePointCloud ("scene_cloud_z_coloured");
            vis.removeShape ("scene_text");
            vis.removeAllShapes(v2);
            vis.removeAllShapes(v1);
            vis.removeAllPointClouds();

            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> outlier_clouds_;
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> outlier_clouds_color, outlier_clouds_3d;

            if(use_HV)
            {
                go->getOutliersForAcceptedModels(outlier_clouds_);
                go->getOutliersForAcceptedModels(outlier_clouds_color, outlier_clouds_3d);

                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr smooth_cloud_ =  go->getSmoothClustersRGBCloud();
                if(smooth_cloud_)
                {
                    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> random_handler (smooth_cloud_);
                    vis.addPointCloud<pcl::PointXYZRGBA> (smooth_cloud_, random_handler, "smooth_cloud", v3);
                }
            }

            {

                //vis.addPointCloud(occ_edges_full, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(occ_edges_full, 255,0,0), "occlusion_edges", v5);
                //vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "occlusion_edges");

                pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler (scene);
                vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud_z_coloured", v5);
            }

            for(size_t kk=0; kk < planes_found.size(); kk++)
            {
                std::stringstream pname;
                pname << "plane_" << kk;

                pcl::visualization::PointCloudColorHandlerRandom<PointT> scene_handler(planes_found[kk].plane_cloud_);
                vis.addPointCloud<PointT> (planes_found[kk].plane_cloud_, scene_handler, pname.str(), v6);

                pname << "chull";
                vis.addPolygonMesh (*planes_found[kk].convex_hull_, pname.str(), v6);
            }

            pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler (scene, 125, 125, 125);
            vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud", v1);
            vis.addText (files_to_recognize[i], 1, 30, 14, 1, 0, 0, "scene_text", v1);
            if(local->isSegmentationRequired() && use_table_plane_)
            {
                //visualize segmentation
                for (size_t c = 0; c < indices.size (); c++)
                {
                    /*if (indices[c].indices.size () < 500)
              continue;*/

                    std::stringstream name;
                    name << "cluster_" << c;

                    typename pcl::PointCloud<PointT>::Ptr cluster (new pcl::PointCloud<PointT>);
                    pcl::copyPointCloud (*scene, indices[c].indices, *cluster);

                    pcl::visualization::PointCloudColorHandlerRandom<PointT> handler_rgb (cluster);
                    vis.addPointCloud<PointT> (cluster, handler_rgb, name.str (), v1);
                }
            }

            if(gt_available)
                or_eval.visualizeGroundTruth(vis, id_1, v4);

            boost::shared_ptr<std::vector<ModelTPtr> > verified_models(new std::vector<ModelTPtr>);
            boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > verified_transforms;
            verified_transforms.reset(new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);

            if(models)
            {
                vtkSmartPointer < vtkTransform > scale_models = vtkSmartPointer<vtkTransform>::New ();
                scale_models->Scale(model_scale, model_scale, model_scale);

                for (size_t j = 0; j < mask_hv.size (); j++)
                {
                    std::stringstream name;
                    name << "cloud_" << j;

                    if(!mask_hv[j])
                    {
                        if(coming_from[j] == 0)
                        {
                            ConstPointInTPtr model_cloud = models->at (j)->getAssembled (VX_SIZE_ICP_);
                            typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
                            pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (j));

                            //pcl::visualization::PointCloudColorHandlerRandom<PointT> random_handler (model_aligned);
                            pcl::visualization::PointCloudColorHandlerRGBField<PointT> random_handler (model_aligned);
                            vis.addPointCloud<PointT> (model_aligned, random_handler, name.str (), v6);
                        }
                        continue;
                    }

                    if(coming_from[j] == 0)
                    {
                        verified_models->push_back(models->at(j));
                        verified_transforms->push_back(transforms->at(j));

                        ConstPointInTPtr model_cloud = models->at (j)->getAssembled (0.002f);
                        typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
                        pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (j));

                        std::cout << models->at (j)->id_ << std::endl;

                        pcl::visualization::PointCloudColorHandlerRGBField<PointT> random_handler (model_aligned);
                        vis.addPointCloud<PointT> (model_aligned, random_handler, name.str (), v2);

                        /*pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud = models->at (j)->getNormalsAssembled (0.002f);
                        typename pcl::PointCloud<pcl::Normal>::Ptr normal_aligned (new pcl::PointCloud<pcl::Normal>);
                        faat_pcl::utils::miscellaneous::transformNormals(normal_cloud, normal_aligned, transforms->at (j));

                        name << "_normals";
                        vis.addPointCloudNormals<PointT, pcl::Normal>(model_aligned, normal_aligned, 10, 0.01, name.str(), v2);*/
                    }
                    else
                    {
                        std::stringstream pname;
                        pname << "plane_v2_" << j;

                        pcl::visualization::PointCloudColorHandlerRandom<PointT> scene_handler(planes_found[j - models->size()].plane_cloud_);
                        vis.addPointCloud<PointT> (planes_found[j - models->size()].plane_cloud_, scene_handler, pname.str(), v2);

                        pname << "chull_v2";
                        vis.addPolygonMesh (*planes_found[j - models->size()].convex_hull_, pname.str(), v2);
                    }
                }

                for (size_t j = 0; j < outlier_clouds_.size (); j++)
                {
                    std::cout << outlier_clouds_3d[j]->points.size() << " " << outlier_clouds_color[j]->points.size() << std::endl;

                    /*{
                        std::stringstream name;
                        name << "cloud_outliers" << j;
                        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> random_handler (outlier_clouds_[j], 255, 255, 0);
                        vis.addPointCloud<pcl::PointXYZ> (outlier_clouds_[j], random_handler, name.str (), v2);
                        vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 16, name.str());
                    }*/

                    {
                        std::stringstream name;
                        name << "cloud_outliers" << j;
                        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> random_handler (outlier_clouds_3d[j], 255, 255, 0);
                        vis.addPointCloud<pcl::PointXYZ> (outlier_clouds_3d[j], random_handler, name.str (), v2);
                        vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, name.str());
                    }

                    {
                        std::stringstream name;
                        name << "cloud_outliers_color" << j;
                        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> random_handler (outlier_clouds_color[j], 255, 0, 255);
                        vis.addPointCloud<pcl::PointXYZ> (outlier_clouds_color[j], random_handler, name.str (), v2);
                        vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, name.str());
                    }
                }
            }

            if(gt_available)
                or_eval.addRecognitionResults(id_1, verified_models, verified_transforms);

        }
        else
        {
            if(gt_available)
                or_eval.addRecognitionResults(id_1, models, transforms);
        }

        vis.setBackgroundColor(0.0,0.0,0.0);
        vis.addCoordinateSystem(0.1f, v5);

        if(PLAY_) {
            vis.spinOnce (500.f, true);
        } else {
            vis.spin ();
        }
    }

    double seconds_elapsed = total_time.getTimeSeconds();
    if(gt_available)
    {
        or_eval.computeStatistics();
        or_eval.saveStatistics(STATISTIC_OUTPUT_FILE_);
        or_eval.savePoseStatistics(POSE_STATISTICS_OUTPUT_FILE_);
        or_eval.savePoseStatisticsRotation(POSE_STATISTICS_ANGLE_OUTPUT_FILE_);

        if(RESULTS_OUTPUT_DIR_.compare("") != 0)
            or_eval.saveRecognitionResults(RESULTS_OUTPUT_DIR_);

        or_eval.computeStatistics();
    }

    std::cout << "Total time:" << seconds_elapsed << std::endl;
    std::cout << "Average per scene:" << seconds_elapsed / static_cast<int>(files_to_recognize.size()) << std::endl;
}

typedef pcl::ReferenceFrame RFType;

int CG_SIZE_ = 3;
float CG_THRESHOLD_ = 0.005f;

/*
 * ./bin/mp_training_recognition -pcd_file /home/aitor/data/ECCV_dataset/pcd_files_reduced/ -models_dir /home/aitor/data/ECCV_dataset/cad_models_2/ -training_dir /home/aitor/data/eccv_trained/ -idx_flann_fn eccv_flann_new.idx -go_require_normals 0 -GT_DIR /home/aitor/data/ECCV_dataset/gt_or_format/ -tes_level 0 -model_scale 0.001 -icp_iterations 10 -pipelines_to_use shot_omp,our_cvfh -go_opt_type 0 -gc_size 5 -gc_threshold 0.01 -splits 32 -test_sampling_density 0.01 -icp_type 1
 * ./bin/mp_training_recognition -pcd_file /home/aitor/data/ECCV_dataset/pcd_files_reduced/ -models_dir /home/aitor/data/ECCV_dataset/cad_models_reduced/ -training_dir /home/aitor/data/eccv_trained_level_1/ -idx_flann_fn eccv_flann_shot_cad_models_reduced_tes_level1.idx -go_require_normals 0 -GT_DIR /home/aitor/data/ECCV_dataset/gt_or_format/ -tes_level 1 -model_scale 0.001 -pipelines_to_use shot_omp -go_opt_type 0 -gc_size 5 -gc_threshold 0.015 -splits 64 -test_sampling_density 0.005 -icp_type 1 -training_dir_shot /home/aitor/data/eccv_trained_level_1/ -icp_iterations 5 -max_our_cvfh_hyp_ 20 -load_views 0 -normalize_ourcvfh_bins 0 -thres_hyp 0
 */

/*
 * ./bin/mp_training_recognition -pcd_file /home/aitor/data/ECCV_dataset/pcd_files/ -models_dir /home/aitor/data/ECCV_dataset/cad_models/ -training_dir /home/aitor/data/eccv_trained_level_1/ -idx_flann_fn eccv_flann_new.idx -go_require_normals 0 -GT_DIR /home/aitor/data/ECCV_dataset/gt_or_format/ -tes_level 0 -model_scale 0.001 -icp_iterations 10 -pipelines_to_use shot_omp,our_cvfh -go_opt_type 0 -gc_size 5 -gc_threshold 0.01 -splits 32 -test_sampling_density 0.005 -icp_type 1 -training_dir_shot /home/aitor/data/eccv_trained_level_0/ -Z_DIST 1.7 -normalize_ourcvfh_bins 0 -detect_clutter 1 -go_resolution 0.005 -go_regularizer 1 -go_inlier_thres 0.005 -PLAY 0 -max_our_cvfh_hyp_ 20 -seg_type 1 -add_planes 1 -use_codebook 1
 */

struct camPosConstraints
{
    bool
    operator() (const Eigen::Vector3f & pos) const
    {
        if (pos[2] > 0)
            return true;

        return false;
    }
    ;
};

/*
 * ./bin/mp_willow -pcd_file /home/aitor/data/willow_dataset/ -models_dir /home/aitor/data/willow/models -training_dir /home/aitor/data/mp_willow_trained/ -idx_flann_fn shot_flann.idx -go_require_normals 0 -GT_DIR /home/aitor/data/willow/willow_dataset_gt/ -tes_level 1 -model_scale 1 -icp_iterations 10 -pipelines_to_use shot_omp,sift -go_opt_type 0 -gc_size 3 -gc_threshold 0.01 -splits 32 -test_sampling_density 0.005 -icp_type 1 -training_dir_shot /home/aitor/data/willow/shot_trained/ -Z_DIST 1.5 -normalize_ourcvfh_bins 0 -detect_clutter 1 -go_resolution 0.005 -go_regularizer 1 -go_inlier_thres 0.01 -PLAY 0 -max_our_cvfh_hyp_ 20 -seg_type 0 -add_planes 1 -use_codebook 0 -load_views 1 -training_input_structure /home/aitor/data/willow/recognizer_structure/ -training_dir_sift /home/aitor/data/willow/sift_trained/ -knn_sift 10 -knn_shot 2
 */

/*

//only sift...
  ./bin/mp_willow -pcd_file /home/aitor/data/willow_dataset/ -models_dir /home/aitor/data/willow/models/ -training_dir /home/aitor/data/willow/ourcvfh_organized -idx_flann_fn shot_flann.idx -go_require_normals 0 -GT_DIR /home/aitor/data/willow/willow_dataset_gt/ -tes_level 1 -model_scale 1 -icp_iterations 15 -pipelines_to_use sift -go_opt_type 0 -gc_size 5 -gc_threshold 0.015 -splits 512 -test_sampling_density 0.01 -icp_type 1 -training_dir_shot /home/aitor/data/willow/shot_trained -Z_DIST 1.5 -normalize_ourcvfh_bins 1 -detect_clutter 1 -go_resolution 0.005 -go_regularizer 2 -go_inlier_thres 0.01 -PLAY 1 -max_our_cvfh_hyp_ 20 -add_planes 1 -use_codebook 0 -load_views 0 -training_input_structure /home/aitor/data/willow/recognizer_structure/ -training_dir_sift /home/aitor/data/willow/sift_trained/ -use_cache 1 -gc_min_dist_cf 0 -seg_type 0 -gc_dot_threshold 0.2 -prune_by_cc 1 -go_use_supervoxels 0 -ransac_threshold_cg_ 0.01 -knn_sift 10 -debug_level 2 -knn_shot 10 -tes_level_our_cvfh 1 -output_dir_before_hv /home/aitor/data/willow/hypotheses_before_hv

*/

int
main (int argc, char ** argv)
{

    boost::function<bool (const Eigen::Vector3f &)> campos_constraints;
    campos_constraints = camPosConstraints ();

    std::string path = "";
    std::string desc_name = "shot_omp";
    std::string training_dir = "trained_models/";
    std::string training_dir_shot = "";
    std::string pcd_file = "";
    int force_retrain = 0;
    int icp_iterations = 20;
    int use_cache = 1;
    int splits = 512;
    int scene = -1;
    int detect_clutter = 1;
    float thres_hyp_ = 0.2f;
    float desc_radius = 0.04f;
    int icp_type = 0;
    int go_opt_type = 2;
    int go_iterations = 7000;
    bool go_use_replace_moves = true;
    float go_inlier_thres = 0.01f;
    float go_resolution = 0.005f;
    float go_regularizer = 2.f;
    float go_clutter_regularizer = 5.f;
    float go_radius_clutter = 0.05f;
    float init_temp = 1000.f;
    std::string idx_flann_fn;
    float radius_normals_go_ = 0.02f;
    bool go_require_normals = false;
    bool go_log = true;
    bool go_init = false;
    float test_sampling_density = 0.005f;
    int tes_level_ = 1;
    int tes_level_our_cvfh_ = 1;
    std::string pipelines_to_use_ = "shot_omp,our_cvfh";
    bool normalize_ourcvfh_bins = false;
    int max_our_cvfh_hyp_ = 30;
    bool use_hough = false;
    bool load_views = true;
    int seg_type = 0;
    bool add_planes = true;
    bool cg_prune_hyp = false;
    bool use_codebook = false;
    std::string training_input_structure = "";
    std::string training_dir_sift = "";
    bool visualize_graph = false;
    float min_dist_cf_ = 1.f;
    float gc_dot_threshold_ = 1.f;
    bool prune_by_cc = false;
    float go_color_sigma = 0.25f;
    bool go_use_supervoxels = false;
    int knn_sift_ = 1;
    int knn_shot_ = 1;
    int our_cvfh_debug_level = 0;
    float ourcvfh_max_distance = 0.35f;
    bool check_normals_orientation = true;
    bool shot_use_iss = false;
    bool gcg_use_graph_ = true;
    float uke_max_distance_ = 1.5f;
    float max_time_cliques_ms_ = 100;

    int max_taken_ = 5;
    std::string idx_flann_sift = "willow_sift.idx";

    pcl::console::parse_argument (argc, argv, "-max_time_cliques_ms", max_time_cliques_ms_);
    pcl::console::parse_argument (argc, argv, "-uke_max_distance", uke_max_distance_);
    pcl::console::parse_argument (argc, argv, "-segmentation_plane_unorganized", segmentation_plane_unorganized_);
    pcl::console::parse_argument (argc, argv, "-gcg_graph", gcg_use_graph_);
    pcl::console::parse_argument (argc, argv, "-non_organized_planes", non_organized_planes_);
    pcl::console::parse_argument (argc, argv, "-use_table_plane", use_table_plane_);
    pcl::console::parse_argument (argc, argv, "-idx_flann_sift", idx_flann_sift);
    pcl::console::parse_argument (argc, argv, "-max_taken", max_taken_);
    pcl::console::parse_argument (argc, argv, "-pose_stats_file", POSE_STATISTICS_OUTPUT_FILE_);
    pcl::console::parse_argument (argc, argv, "-pose_stats_file_angle", POSE_STATISTICS_ANGLE_OUTPUT_FILE_);
    pcl::console::parse_argument (argc, argv, "-use_histogram_specification", use_histogram_specification_);
    pcl::console::parse_argument (argc, argv, "-shot_use_iss", shot_use_iss);
    pcl::console::parse_argument (argc, argv, "-check_normals_orientation", check_normals_orientation);
    pcl::console::parse_argument (argc, argv, "-ourcvfh_max_distance", ourcvfh_max_distance);
    pcl::console::parse_argument (argc, argv, "-BF", BILATERAL_FILTER_);
    pcl::console::parse_argument (argc, argv, "-tes_level_our_cvfh", tes_level_our_cvfh_);
    pcl::console::parse_argument (argc, argv, "-debug_level", our_cvfh_debug_level);
    pcl::console::parse_argument (argc, argv, "-knn_shot", knn_shot_);
    pcl::console::parse_argument (argc, argv, "-knn_sift", knn_sift_);
    pcl::console::parse_argument (argc, argv, "-go_color_sigma", go_color_sigma);
    pcl::console::parse_argument (argc, argv, "-go_use_supervoxels", go_use_supervoxels);
    pcl::console::parse_argument (argc, argv, "-prune_by_cc", prune_by_cc);
    pcl::console::parse_argument (argc, argv, "-visualize_graph", visualize_graph);
    pcl::console::parse_argument (argc, argv, "-training_dir_sift", training_dir_sift);
    pcl::console::parse_argument (argc, argv, "-training_input_structure", training_input_structure);
    pcl::console::parse_argument (argc, argv, "-use_codebook", use_codebook);
    pcl::console::parse_argument (argc, argv, "-cg_prune_hyp", cg_prune_hyp);
    pcl::console::parse_argument (argc, argv, "-seg_type", seg_type);
    pcl::console::parse_argument (argc, argv, "-add_planes", add_planes);
    pcl::console::parse_argument (argc, argv, "-use_hv", use_HV);
    pcl::console::parse_argument (argc, argv, "-max_our_cvfh_hyp_", max_our_cvfh_hyp_);
    pcl::console::parse_argument (argc, argv, "-normalize_ourcvfh_bins", normalize_ourcvfh_bins);
    pcl::console::parse_argument (argc, argv, "-models_dir", path);
    pcl::console::parse_argument (argc, argv, "-training_dir", training_dir);
    pcl::console::parse_argument (argc, argv, "-training_dir_shot", training_dir_shot);
    pcl::console::parse_argument (argc, argv, "-descriptor_name", desc_name);
    pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
    pcl::console::parse_argument (argc, argv, "-force_retrain", force_retrain);
    pcl::console::parse_argument (argc, argv, "-icp_iterations", icp_iterations);
    pcl::console::parse_argument (argc, argv, "-use_cache", use_cache);
    pcl::console::parse_argument (argc, argv, "-splits", splits);
    pcl::console::parse_argument (argc, argv, "-gc_size", CG_SIZE_);
    pcl::console::parse_argument (argc, argv, "-gc_threshold", CG_THRESHOLD_);

    float ransac_threshold_cg_ = CG_THRESHOLD_;

    pcl::console::parse_argument (argc, argv, "-scene", scene);
    pcl::console::parse_argument (argc, argv, "-detect_clutter", detect_clutter);
    pcl::console::parse_argument (argc, argv, "-thres_hyp", thres_hyp_);
    pcl::console::parse_argument (argc, argv, "-icp_type", icp_type);
    pcl::console::parse_argument (argc, argv, "-vx_size_icp", VX_SIZE_ICP_);
    pcl::console::parse_argument (argc, argv, "-model_scale", model_scale);
    pcl::console::parse_argument (argc, argv, "-go_opt_type", go_opt_type);
    pcl::console::parse_argument (argc, argv, "-go_iterations", go_iterations);
    pcl::console::parse_argument (argc, argv, "-go_use_replace_moves", go_use_replace_moves);
    pcl::console::parse_argument (argc, argv, "-go_inlier_thres", go_inlier_thres);
    pcl::console::parse_argument (argc, argv, "-go_resolution", go_resolution);
    pcl::console::parse_argument (argc, argv, "-go_initial_temp", init_temp);
    pcl::console::parse_argument (argc, argv, "-go_require_normals", go_require_normals);
    pcl::console::parse_argument (argc, argv, "-go_regularizer", go_regularizer);
    pcl::console::parse_argument (argc, argv, "-go_clutter_regularizer", go_clutter_regularizer);
    pcl::console::parse_argument (argc, argv, "-go_radius_clutter", go_radius_clutter);
    pcl::console::parse_argument (argc, argv, "-go_log", go_log);
    pcl::console::parse_argument (argc, argv, "-go_init", go_init);
    pcl::console::parse_argument (argc, argv, "-idx_flann_fn", idx_flann_fn);
    pcl::console::parse_argument (argc, argv, "-PLAY", PLAY_);
    pcl::console::parse_argument (argc, argv, "-go_log_file", go_log_file_);
    pcl::console::parse_argument (argc, argv, "-test_sampling_density", test_sampling_density);
    pcl::console::parse_argument (argc, argv, "-tes_level", tes_level_);
    pcl::console::parse_argument (argc, argv, "-Z_DIST", Z_DIST_);
    pcl::console::parse_argument (argc, argv, "-pipelines_to_use", pipelines_to_use_);
    pcl::console::parse_argument (argc, argv, "-use_hough", use_hough);
    pcl::console::parse_argument (argc, argv, "-ransac_threshold_cg_", ransac_threshold_cg_);
    pcl::console::parse_argument (argc, argv, "-load_views", load_views);
    pcl::console::parse_argument (argc, argv, "-gc_min_dist_cf", min_dist_cf_);
    pcl::console::parse_argument (argc, argv, "-gc_dot_threshold", gc_dot_threshold_);
    pcl::console::parse_argument (argc, argv, "-stat_file", STATISTIC_OUTPUT_FILE_);

    pcl::console::parse_argument (argc, argv, "-output_dir_before_hv", RESULTS_OUTPUT_DIR_);
    pcl::console::parse_argument (argc, argv, "-use_highest_plane", use_highest_plane_);

    MODELS_DIR_FOR_VIS_ = path;
    pcl::console::parse_argument (argc, argv, "-models_dir_vis", MODELS_DIR_FOR_VIS_);
    pcl::console::parse_argument (argc, argv, "-GT_DIR", GT_DIR_);
    MODELS_DIR_ = path;

    typedef pcl::PointXYZRGB PointT;

    std::cout << "VX_SIZE_ICP_" << VX_SIZE_ICP_ << std::endl;
    if (pcd_file.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing scenes\n");
        return -1;
    }

    if (path.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing the models using the -models_dir [dir] option\n");
        return -1;
    }

    bf::path models_dir_path = path;
    if (!bf::exists (models_dir_path))
    {
        PCL_ERROR("Models dir path %s does not exist, use -models_dir [dir] option\n", path.c_str());
        return -1;
    }
    else
    {
        std::vector < std::string > files;
        std::string start = "";
        std::string ext = std::string ("pcd");
        bf::path dir = models_dir_path;
        getModelsInDirectory (dir, start, files, ext);
        std::cout << "Number of models in directory is:" << files.size() << std::endl;
    }

    parameters_for_go.radius_normals_go_ = radius_normals_go_;
    parameters_for_go.radius_clutter = go_radius_clutter;
    parameters_for_go.clutter_regularizer = go_clutter_regularizer;
    parameters_for_go.regularizer = go_regularizer;
    parameters_for_go.init_temp = init_temp;
    parameters_for_go.go_init = go_init;
    parameters_for_go.go_inlier_thres = go_inlier_thres;
    parameters_for_go.go_iterations = go_iterations;
    parameters_for_go.go_opt_type = go_opt_type;
    parameters_for_go.require_normals = go_require_normals;
    parameters_for_go.go_resolution = go_resolution;
    parameters_for_go.go_use_replace_moves = go_use_replace_moves;
    parameters_for_go.detect_clutter = static_cast<bool>(detect_clutter);
    parameters_for_go.color_sigma_ = go_color_sigma;
    parameters_for_go.use_super_voxels_ = go_use_supervoxels;

    //configure normal estimator
    boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
    normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
    normal_estimator->setCMR (false);
    normal_estimator->setDoVoxelGrid (true);
    normal_estimator->setRemoveOutliers (true);
    normal_estimator->setValuesForCMRFalse (0.003f, 0.02f);

    //configure keypoint extractor
    boost::shared_ptr<faat_pcl::rec_3d_framework::UniformSamplingExtractor<PointT> > uniform_keypoint_extractor ( new faat_pcl::rec_3d_framework::UniformSamplingExtractor<PointT>);
    uniform_keypoint_extractor->setSamplingDensity (0.01f);
    uniform_keypoint_extractor->setFilterPlanar (true);
    uniform_keypoint_extractor->setThresholdPlanar(0.05);
    uniform_keypoint_extractor->setMaxDistance(uke_max_distance_);

    boost::shared_ptr<faat_pcl::rec_3d_framework::KeypointExtractor<PointT> > keypoint_extractor;
    keypoint_extractor = boost::static_pointer_cast<faat_pcl::rec_3d_framework::KeypointExtractor<PointT> > (uniform_keypoint_extractor);

    //ISS
    if(shot_use_iss)
    {
        boost::shared_ptr<faat_pcl::rec_3d_framework::ISSKeypointExtractor<PointT> > issk_extractor ( new faat_pcl::rec_3d_framework::ISSKeypointExtractor<PointT>);
        issk_extractor->setSupportRadius(0.04f);
        issk_extractor->setNonMaximaRadius(0.005f);
        keypoint_extractor = boost::static_pointer_cast<faat_pcl::rec_3d_framework::KeypointExtractor<PointT> > (issk_extractor);
    }

    //configure cg algorithm (geometric consistency grouping)
    boost::shared_ptr<pcl::CorrespondenceGrouping<PointT, PointT> > cast_cg_alg;

    boost::shared_ptr<pcl::Hough3DGrouping<PointT, PointT, pcl::ReferenceFrame, pcl::ReferenceFrame> >
            hough_3d_voting_cg_alg(new pcl::Hough3DGrouping<PointT, PointT, pcl::ReferenceFrame, pcl::ReferenceFrame>);

    hough_3d_voting_cg_alg->setHoughBinSize (CG_THRESHOLD_);
    hough_3d_voting_cg_alg->setHoughThreshold (CG_SIZE_);
    hough_3d_voting_cg_alg->setUseInterpolation (false);
    hough_3d_voting_cg_alg->setUseDistanceWeight (false);

    boost::shared_ptr<pcl::GeometricConsistencyGrouping<PointT, PointT> > gc_alg (
                new pcl::GeometricConsistencyGrouping<PointT,
                PointT>);


    gc_alg->setGCThreshold (CG_SIZE_);
    gc_alg->setGCSize (CG_THRESHOLD_);

    cast_cg_alg = boost::static_pointer_cast<pcl::CorrespondenceGrouping<PointT, PointT> > (gc_alg);

    boost::shared_ptr<faat_pcl::GraphGeometricConsistencyGrouping<PointT, PointT> > gcg_alg (
                new faat_pcl::GraphGeometricConsistencyGrouping<PointT,
                PointT>);
    gcg_alg->setGCThreshold (CG_SIZE_);
    gcg_alg->setGCSize (CG_THRESHOLD_);
    gcg_alg->setRansacThreshold (ransac_threshold_cg_);
    gcg_alg->setUseGraph(gcg_use_graph_);
    gcg_alg->setVisualizeGraph(visualize_graph);
    gcg_alg->setPrune(cg_prune_hyp);
    gcg_alg->setDotDistance(gc_dot_threshold_);
    gcg_alg->setDistForClusterFactor(min_dist_cf_);
    gcg_alg->pruneByCC(prune_by_cc);
    gcg_alg->setMaxTaken(max_taken_); //std::numeric_limits<int>::max());
    gcg_alg->setSortCliques(true);
    gcg_alg->setCheckNormalsOrientation(check_normals_orientation);
    gcg_alg->setMaxTimeForCliquesComputation(max_time_cliques_ms_);

    cast_cg_alg = boost::static_pointer_cast<pcl::CorrespondenceGrouping<PointT, PointT> > (gcg_alg);
    //TODO: Fix normals for SIFT, so we can use the other correspondence grouping (DONE)


#ifdef _MSC_VER
    _CrtCheckMemory();
#endif

    std::vector<std::string> strs;
    boost::split (strs, pipelines_to_use_, boost::is_any_of (","));

    boost::shared_ptr<faat_pcl::rec_3d_framework::MultiRecognitionPipeline<PointT> > multi_recog;
    multi_recog.reset(new faat_pcl::rec_3d_framework::MultiRecognitionPipeline<PointT>);

    for(size_t i=0; i < strs.size(); i++)
    {

        boost::shared_ptr<faat_pcl::rec_3d_framework::Recognizer<PointT> > cast_recog;

        if (strs[i].compare ("sift") == 0)
        {
            desc_name = std::string ("sift");

            boost::shared_ptr<faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, pcl::Histogram<128> > > estimator;
            estimator.reset (new faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, pcl::Histogram<128> >);

            boost::shared_ptr<faat_pcl::rec_3d_framework::LocalEstimator<PointT, pcl::Histogram<128> > > cast_estimator;
            cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, pcl::Histogram<128> > > (estimator);

            boost::shared_ptr<faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> >
                    mesh_source (
                        new faat_pcl::rec_3d_framework::RegisteredViewsSource<
                        pcl::PointXYZRGBNormal,
                        pcl::PointXYZRGB,
                        pcl::PointXYZRGB>);
            mesh_source->setPath (path);
            mesh_source->setModelStructureDir (training_input_structure);
            mesh_source->generate (training_dir_sift);

            boost::shared_ptr<faat_pcl::rec_3d_framework::Source<PointT> > cast_source;
            cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > (mesh_source);


#define SIFT_FLANN_L1
#ifdef SIFT_FLANN_L1
            boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<128> > > local;
            local.reset(new faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<128> > (idx_flann_sift, "willow_sift_codebook.txt"));
            cast_recog = boost::static_pointer_cast<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<128> > > (local);
#endif

#ifdef SIFT_FLANN_L2
            boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L2, PointT, pcl::Histogram<128> > > local;
            local.reset(new faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L2, PointT, pcl::Histogram<128> > (idx_flann_sift, "willow_sift_codebook.txt"));
            cast_recog = boost::static_pointer_cast<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L2, PointT, pcl::Histogram<128> > > (local);
#endif
            local->setDataSource (cast_source);
            local->setTrainingDir (training_dir_sift);
            local->setDescriptorName (desc_name);
            local->setFeatureEstimator (cast_estimator);
            local->setCGAlgorithm (cast_cg_alg);

            local->setUseCache (static_cast<bool> (use_cache));
            local->setVoxelSizeICP (VX_SIZE_ICP_);
            local->setThresholdAcceptHyp (thres_hyp_);
            local->setICPIterations (0);
            local->setKdtreeSplits (splits);
            local->setICPType(icp_type);
            local->setUseCodebook(use_codebook);
            local->initialize (static_cast<bool> (force_retrain));
            local->setKnn(knn_sift_);
            local->setCorrespondenceDistanceConstantWeight(1.f);
            multi_recog->addRecognizer(cast_recog);

        }

        if (strs[i].compare ("sift_opencv") == 0)
        {
            desc_name = std::string ("sift_opencv");
            std::string idx_flann_sift = "sift_opencv_willow.idx";

            boost::shared_ptr<faat_pcl::rec_3d_framework::OpenCVSIFTLocalEstimation<PointT, pcl::Histogram<128> > > estimator;
            estimator.reset (new faat_pcl::rec_3d_framework::OpenCVSIFTLocalEstimation<PointT, pcl::Histogram<128> >);

            boost::shared_ptr<faat_pcl::rec_3d_framework::LocalEstimator<PointT, pcl::Histogram<128> > > cast_estimator;
            cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::OpenCVSIFTLocalEstimation<PointT, pcl::Histogram<128> > > (estimator);

            boost::shared_ptr<faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> >
                    mesh_source (
                        new faat_pcl::rec_3d_framework::RegisteredViewsSource<
                        pcl::PointXYZRGBNormal,
                        pcl::PointXYZRGB,
                        pcl::PointXYZRGB>);
            mesh_source->setPath (path);
            mesh_source->setModelStructureDir (training_input_structure);
            mesh_source->generate (training_dir_sift);

            boost::shared_ptr<faat_pcl::rec_3d_framework::Source<PointT> > cast_source;
            cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > (mesh_source);


#define SIFT_FLANN_L1
#ifdef SIFT_FLANN_L1
            boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<128> > > local;
            local.reset(new faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<128> > (idx_flann_sift, "willow_sift_codebook.txt"));
            cast_recog = boost::static_pointer_cast<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<128> > > (local);
#endif

#ifdef SIFT_FLANN_L2
            boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L2, PointT, pcl::Histogram<128> > > local;
            local.reset(new faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L2, PointT, pcl::Histogram<128> > (idx_flann_sift, "willow_sift_codebook.txt"));
            cast_recog = boost::static_pointer_cast<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L2, PointT, pcl::Histogram<128> > > (local);
#endif
            local->setDataSource (cast_source);
            local->setTrainingDir (training_dir_sift);
            local->setDescriptorName (desc_name);
            local->setFeatureEstimator (cast_estimator);
            local->setCGAlgorithm (cast_cg_alg);

            local->setUseCache (static_cast<bool> (use_cache));
            local->setVoxelSizeICP (VX_SIZE_ICP_);
            local->setThresholdAcceptHyp (thres_hyp_);
            local->setICPIterations (0);
            local->setKdtreeSplits (splits);
            local->setICPType(icp_type);
            local->setUseCodebook(use_codebook);
            local->initialize (static_cast<bool> (force_retrain));
            local->setKnn(knn_sift_);
            multi_recog->addRecognizer(cast_recog);

        }

        if (strs[i].compare ("shot_omp") == 0)
        {
            desc_name = std::string ("shot");
            boost::shared_ptr<faat_pcl::rec_3d_framework::SHOTLocalEstimationOMP<PointT, pcl::Histogram<352> > > estimator;
            estimator.reset (new faat_pcl::rec_3d_framework::SHOTLocalEstimationOMP<PointT, pcl::Histogram<352> >);
            estimator->setNormalEstimator (normal_estimator);
            estimator->addKeypointExtractor (keypoint_extractor);
            estimator->setSupportRadius (desc_radius);
            estimator->setAdaptativeMLS (false);

            boost::shared_ptr<faat_pcl::rec_3d_framework::LocalEstimator<PointT, pcl::Histogram<352> > > cast_estimator;
            cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::LocalEstimator<PointT, pcl::Histogram<352> > > (estimator);

            /*boost::shared_ptr<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> >
                    source (new faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal,pcl::PointXYZRGB>);

            source->setPath (path);
            source->setModelScale (1.f);
            source->setRadiusSphere (1.f);
            source->setTesselationLevel (tes_level_);
            source->setDotNormal (-1.f);
            source->setLoadViews (load_views);
            source->setCamPosConstraints (campos_constraints);
            source->setLoadIntoMemory(false);
            source->generate (training_dir_shot);*/

            /*boost::shared_ptr<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> >
                    source (
                        new faat_pcl::rec_3d_framework::PartialPCDSource<
                        pcl::PointXYZRGBNormal,
                        pcl::PointXYZRGB>);
            source->setPath (path);
            source->setModelScale (1.f);
            source->setRadiusSphere (1.f);
            source->setTesselationLevel (tes_level_);
            source->setDotNormal (-1.f);
            source->setLoadViews (load_views);
            source->setCamPosConstraints (campos_constraints);
            source->setLoadIntoMemory(false);
            source->setGenOrganized(true);
            source->setWindowSizeAndFocalLength(640, 480, 575.f);
            source->generate (training_dir_shot);

            boost::shared_ptr<faat_pcl::rec_3d_framework::Source<pcl::PointXYZRGB> > cast_source;
            cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> > (source);*/

            boost::shared_ptr<faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> >
                    source (
                        new faat_pcl::rec_3d_framework::RegisteredViewsSource<
                        pcl::PointXYZRGBNormal,
                        pcl::PointXYZRGB,
                        pcl::PointXYZRGB>);
            source->setPath (path);
            source->setModelStructureDir (training_input_structure);
            source->generate (training_dir_shot);

            boost::shared_ptr<faat_pcl::rec_3d_framework::Source<pcl::PointXYZRGB> > cast_source;
            cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> > (source);

            if(use_hough)
            {

                cast_cg_alg = boost::static_pointer_cast<pcl::Hough3DGrouping<PointT, PointT> > (hough_3d_voting_cg_alg);

                boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionHoughGroupingPipeline<flann::L1, PointT, pcl::Histogram<352> > > local;
                local.reset(new faat_pcl::rec_3d_framework::LocalRecognitionHoughGroupingPipeline<flann::L1, PointT, pcl::Histogram<352> > (idx_flann_fn));
                local->setDataSource (cast_source);
                local->setTrainingDir (training_dir_shot);
                local->setDescriptorName (desc_name);
                local->setFeatureEstimator (cast_estimator);
                local->setCGAlgorithm (cast_cg_alg);

                local->setUseCache (static_cast<bool> (use_cache));
                local->setVoxelSizeICP (VX_SIZE_ICP_);
                local->setThresholdAcceptHyp (thres_hyp_);
                uniform_keypoint_extractor->setSamplingDensity (test_sampling_density);
                local->setICPIterations (0);
                local->setKdtreeSplits (splits);
                local->setICPType(icp_type);

                local->initialize (static_cast<bool> (force_retrain));

                cast_recog = boost::static_pointer_cast<faat_pcl::rec_3d_framework::LocalRecognitionHoughGroupingPipeline<flann::L1, PointT, pcl::Histogram<352> > > (local);
                multi_recog->addRecognizer(cast_recog);
            }
            else
            {
                std::string codebook_name = "trash.txt";
                if(use_codebook)
                {
                    codebook_name = "shot_willow_codebook.txt";
                }

                boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> > > local;
                local.reset(new faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> > (idx_flann_fn, codebook_name));
                local->setDataSource (cast_source);
                local->setTrainingDir (training_dir_shot);
                local->setDescriptorName (desc_name);
                local->setFeatureEstimator (cast_estimator);
                local->setCGAlgorithm (cast_cg_alg);
                local->setKnn(knn_shot_);
                local->setUseCache (static_cast<bool> (use_cache));
                local->setVoxelSizeICP (VX_SIZE_ICP_);
                local->setThresholdAcceptHyp (thres_hyp_);
                uniform_keypoint_extractor->setSamplingDensity (test_sampling_density);
                local->setICPIterations (0);
                local->setKdtreeSplits (splits);
                local->setICPType(icp_type);
                local->setUseCodebook(use_codebook);
                local->initialize (static_cast<bool> (force_retrain));

                cast_recog = boost::static_pointer_cast<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> > > (local);
                multi_recog->addRecognizer(cast_recog);
            }
        }

        int nn_ = 45;
        if (strs[i].compare ("rf_our_cvfh_color") == 0)
        {


            /*boost::shared_ptr<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> >
                    source (
                        new faat_pcl::rec_3d_framework::PartialPCDSource<
                        pcl::PointXYZRGBNormal,
                        pcl::PointXYZRGB>);
            source->setPath (path);
            source->setModelScale (1.f);
            source->setRadiusSphere (1.f);
            source->setTesselationLevel (tes_level_our_cvfh_);
            source->setDotNormal (-1.f);
            source->setUseVertices(false);
            source->setLoadViews (load_views);
            source->setCamPosConstraints (campos_constraints);
            source->setLoadIntoMemory(false);
            source->setGenOrganized(true);
            source->setWindowSizeAndFocalLength(640, 480, 575.f);
            source->generate (training_dir);

            boost::shared_ptr<faat_pcl::rec_3d_framework::Source<pcl::PointXYZRGB> > cast_source;
            cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> > (source);
            */

            boost::shared_ptr<faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> >
                    source (
                        new faat_pcl::rec_3d_framework::RegisteredViewsSource<
                        pcl::PointXYZRGBNormal,
                        pcl::PointXYZRGB,
                        pcl::PointXYZRGB>);
            source->setPath (path);
            source->setModelStructureDir (training_input_structure);
            source->generate (training_dir);

            boost::shared_ptr<faat_pcl::rec_3d_framework::Source<pcl::PointXYZRGB> > cast_source;
            cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> > (source);

            //configure normal estimator
            boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
            normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
            normal_estimator->setCMR (false);
            normal_estimator->setDoVoxelGrid (false);
            normal_estimator->setRemoveOutliers (false);
            normal_estimator->setValuesForCMRFalse (0.001f, 0.02f);
            normal_estimator->setForceUnorganized(true);

            //boost::shared_ptr<faat_pcl::rec_3d_framework::ColorOURCVFHEstimator<PointT, pcl::Histogram<1327> > > vfh_estimator;
            //vfh_estimator.reset (new faat_pcl::rec_3d_framework::ColorOURCVFHEstimator<PointT, pcl::Histogram<1327> >);

            boost::shared_ptr<faat_pcl::rec_3d_framework::OrganizedColorOURCVFHEstimator<PointT, pcl::Histogram<1327> > > vfh_estimator;
            vfh_estimator.reset (new faat_pcl::rec_3d_framework::OrganizedColorOURCVFHEstimator<PointT, pcl::Histogram<1327> >);
            vfh_estimator->setNormalEstimator (normal_estimator);
            vfh_estimator->setNormalizeBins (normalize_ourcvfh_bins);
            vfh_estimator->setUseRFForColor (true);
            //vfh_estimator->setRefineClustersParam (2.5f);
            vfh_estimator->setRefineClustersParam (100.f);
            vfh_estimator->setAdaptativeMLS (false);

            vfh_estimator->setAxisRatio (1.f);
            vfh_estimator->setMinAxisValue (1.f);

            {
                //segmentation parameters for training
                std::vector<float> eps_thresholds, cur_thresholds, clus_thresholds;
                eps_thresholds.push_back (0.15);
                cur_thresholds.push_back (0.015f);
                //cur_thresholds.push_back (0.02f);
                cur_thresholds.push_back (1.f);
                clus_thresholds.push_back (10.f);

                vfh_estimator->setClusterToleranceVector (clus_thresholds);
                vfh_estimator->setEpsAngleThresholdVector (eps_thresholds);
                vfh_estimator->setCurvatureThresholdVector (cur_thresholds);
            }

            //vfh_estimator->setCVFHParams (0.125f, 0.0175f, 2.5f);
            //vfh_estimator->setCVFHParams (0.15f, 0.0175f, 3.f); //willow_challenge_trained

            std::string desc_name = "rf_our_cvfh_color";
            if (normalize_ourcvfh_bins)
            {
                desc_name = "rf_our_cvfh_color_normalized";
            }

            std::cout << "Descriptor name:" << desc_name << std::endl;

            boost::shared_ptr<faat_pcl::rec_3d_framework::OURCVFHEstimator<pcl::PointXYZRGB, pcl::Histogram<1327> > > cast_estimator;
            //cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::ColorOURCVFHEstimator<pcl::PointXYZRGB, pcl::Histogram<1327> > > (vfh_estimator);
            cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::OrganizedColorOURCVFHEstimator<pcl::PointXYZRGB, pcl::Histogram<1327> > > (vfh_estimator);

            boost::shared_ptr<faat_pcl::rec_3d_framework::GlobalNNCVFHRecognizer<faat_pcl::Metrics::HistIntersectionUnionDistance, PointT, pcl::Histogram<1327> > > rf_color_ourcvfh_global_;
            rf_color_ourcvfh_global_.reset(new faat_pcl::rec_3d_framework::GlobalNNCVFHRecognizer<faat_pcl::Metrics::HistIntersectionUnionDistance, PointT, pcl::Histogram<1327> >);
            rf_color_ourcvfh_global_->setDataSource (cast_source);
            rf_color_ourcvfh_global_->setTrainingDir (training_dir);
            rf_color_ourcvfh_global_->setDescriptorName (desc_name);
            rf_color_ourcvfh_global_->setFeatureEstimator (cast_estimator);
            rf_color_ourcvfh_global_->setNN (nn_);
            rf_color_ourcvfh_global_->setICPIterations (0);
            rf_color_ourcvfh_global_->setNoise (0.0f);
            rf_color_ourcvfh_global_->setUseCache (use_cache);
            rf_color_ourcvfh_global_->setMaxHyp(max_our_cvfh_hyp_);
            rf_color_ourcvfh_global_->setMaxDescDistance(ourcvfh_max_distance);
            rf_color_ourcvfh_global_->initialize (force_retrain);
            rf_color_ourcvfh_global_->setDebugLevel(our_cvfh_debug_level);
            {
                //segmentation parameters for recognition
                std::vector<float> eps_thresholds, cur_thresholds, clus_thresholds;
                eps_thresholds.push_back (0.15);
                //cur_thresholds.push_back (0.015f);

                if(BILATERAL_FILTER_)
                {
                    cur_thresholds.push_back (0.015f);
                    cur_thresholds.push_back (0.02f);
                }
                else
                {
                    cur_thresholds.push_back (0.02f);
                    cur_thresholds.push_back (0.03f);
                }
                cur_thresholds.push_back (1.f);
                //cur_thresholds.push_back (0.03f);
                clus_thresholds.push_back (10.f);

                vfh_estimator->setClusterToleranceVector (clus_thresholds);
                vfh_estimator->setEpsAngleThresholdVector (eps_thresholds);
                vfh_estimator->setCurvatureThresholdVector (cur_thresholds);

                vfh_estimator->setAxisRatio (0.8f);
                vfh_estimator->setMinAxisValue (0.8f);

                vfh_estimator->setAdaptativeMLS (false);
            }

            cast_recog = boost::static_pointer_cast<faat_pcl::rec_3d_framework::GlobalNNCVFHRecognizer<faat_pcl::Metrics::HistIntersectionUnionDistance, PointT, pcl::Histogram<1327> > > (rf_color_ourcvfh_global_);
            multi_recog->addRecognizer(cast_recog);
        }
    }

    multi_recog->setCGAlgorithm(gcg_alg);
    multi_recog->setVoxelSizeICP(VX_SIZE_ICP_);
    multi_recog->setICPType(icp_type);
    multi_recog->setICPIterations(icp_iterations);
    multi_recog->initialize();
    recognizeAndVisualize<PointT> (multi_recog, pcd_file, seg_type, add_planes);
}
