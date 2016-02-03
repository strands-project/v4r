/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2012 Aitor Aldoma, Federico Tombari
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#include <v4r/common/color_transforms.h>
#include <v4r/common/normals.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/recognition/ghv.h>
#include <functional>
#include <numeric>
#include <pcl/common/angles.h>
#include <pcl/common/time.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <omp.h>

namespace v4r {

template<typename ModelT, typename SceneT>
mets::gol_type
GHV<ModelT, SceneT>::evaluateSolution (const std::vector<bool> & active, int changed)
{
    const HVRecognitionModel<ModelT> &rm = *recognition_models_[changed];

    int sign = 1;
    if ( !active[changed]) //it has been deactivated
        sign = -1;

    updateExplainedVector (rm.explained_scene_indices_, rm.distances_to_explained_scene_indices_, sign, changed);

    if(param_.detect_clutter_)
        updateUnexplainedVector (rm.unexplained_in_neighborhood, rm.unexplained_in_neighborhood_weights, rm.explained_scene_indices_, sign);

    updateCMDuplicity (rm.complete_cloud_occupancy_indices_, sign);

    double duplicity = previous_duplicity_;
    //duplicity = 0.f; //ATTENTION!!
    double good_info = previous_explained_value_;

    double unexplained_info = previous_unexplained_;

    if(!param_.detect_clutter_)
        unexplained_info = 0.f;

    double bad_info = previous_bad_info_ + (rm.getOutliersWeight() * rm.outlier_indices_.size ()) * sign;

    previous_bad_info_ = bad_info;

    double duplicity_cm = (double)previous_duplicity_complete_models_ * param_.w_occupied_multiple_cm_;

    double cost = -(good_info - bad_info - duplicity - unexplained_info - duplicity_cm - countActiveHypotheses(active) - countPointsOnDifferentPlaneSides(active));

//    std::cout << "COST: " << cost << " (good info: " << good_info << ", bad _info: " << bad_info << ", duplicity:" << duplicity <<
//                 ", unexplained_info: " << unexplained_info << ", duplicity_cm: " << duplicity_cm <<
//                 ", ActiveHypotheses: " << countActiveHypotheses (active) <<
//                 ", PointsOnDifferentPlaneSides: " <<  countPointsOnDifferentPlaneSides(active) << ")" << std::endl;

    if(cost_logger_) {
        cost_logger_->increaseEvaluated();
        cost_logger_->addCostEachTimeEvaluated(cost);
    }

    return static_cast<mets::gol_type> (cost); //return the dual to our max problem
}


template<typename ModelT, typename SceneT>
double
GHV<ModelT, SceneT>::countActiveHypotheses (const std::vector<bool> & sol)
{
    double c = 0;
    for (size_t i = 0; i < sol.size (); i++)
    {
        const HVRecognitionModel<ModelT> &rm = *recognition_models_[i];
        if (sol[i])
            c += static_cast<double>(rm.explained_scene_indices_.size()) / 2.f * rm.hyp_penalty_ + min_contribution_;
    }

    return c;
    //return static_cast<float> (c) * active_hyp_penalty_;
}

template<typename ModelT, typename SceneT>
double
GHV<ModelT, SceneT>::
countPointsOnDifferentPlaneSides (const std::vector<bool> & sol)
{
    if(!param_.use_points_on_plane_side_)
        return 0;

    double c=0;
    for(size_t i=0; i < recognition_models_.size(); i++)
    {
        HVRecognitionModel<ModelT> &rm = *recognition_models_[i];

        if(sol[i] && rm.is_planar_)
        {
            for(size_t j=0; j<recognition_models_.size(); j++)
            {
                if(sol[j]) // do we have to check again if j is a plane?
                    c += points_on_plane_sides_(i,j);
            }
        }
    }
    return c;
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::addPlanarModels(std::vector<PlaneModel<ModelT> > & planar_models)
{
    size_t existing_models = recognition_models_.size();
    recognition_models_.resize( existing_models + planar_models.size() );

#pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i < planar_models.size(); i++)
    {
        recognition_models_[existing_models + i].reset(new HVRecognitionModel<ModelT>);
        HVRecognitionModel<ModelT> &rm = *recognition_models_[existing_models + i];
        rm.is_planar_ = true;
        rm.visible_cloud_.reset( new pcl::PointCloud<ModelT> );
        rm.complete_cloud_ = planar_models[i].projectPlaneCloud();

        ZBuffering<ModelT, SceneT> zbuffer_scene (param_.zbuffer_scene_resolution_, param_.zbuffer_scene_resolution_, 1.f);
        if (!occlusion_cloud_->isOrganized ())
            zbuffer_scene.computeDepthMap (*occlusion_cloud_, true);

        //self-occlusions
        typename pcl::PointCloud<ModelT>::Ptr filtered (new pcl::PointCloud<ModelT> ( *rm.complete_cloud_ ));
        typename pcl::PointCloud<ModelT>::ConstPtr const_filtered(new pcl::PointCloud<ModelT> (*filtered));

        std::vector<int> indices_cloud_occlusion;

        if (occlusion_cloud_->isOrganized ())
            filtered = filter<ModelT,SceneT> (*occlusion_cloud_, *const_filtered, param_.focal_length_, param_.occlusion_thres_, indices_cloud_occlusion);
        else
            zbuffer_scene.filter (*const_filtered, *filtered, param_.occlusion_thres_);

        rm.visible_cloud_ = filtered;

        rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal> ());
        rm.visible_cloud_normals_->points.resize(filtered->points.size());
        rm.plane_model_ = planar_models[i];
        Eigen::Vector3f plane_normal;
        plane_normal[0] = rm.plane_model_.coefficients_.values[0];
        plane_normal[1] = rm.plane_model_.coefficients_.values[1];
        plane_normal[2] = rm.plane_model_.coefficients_.values[2];

        for(size_t k=0; k < rm.visible_cloud_normals_->points.size(); k++)
            rm.visible_cloud_normals_->points[k].getNormalVector3fMap() = plane_normal;

        rm.complete_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal>());
        rm.complete_cloud_normals_->points.resize( rm.complete_cloud_->points.size() );
        for(size_t k=0; k<rm.complete_cloud_->points.size(); k++)
            rm.complete_cloud_normals_->points[k].getNormalVector3fMap() = plane_normal;
    }
}

//clutter segmentation performed by supervoxels is only allowed with pcl::PointXYZRGB types
//this happens only for PCL versions below 1.7.2, since pcl::SuperVoxelClustering cannot be instantiated
//thats the reason for this hacks...

template<typename SceneT>
bool
isSuperVoxelClutterSegmentationPossible()
{
    return true;
}

template<>
bool
isSuperVoxelClutterSegmentationPossible<pcl::PointXYZ>()
{
    return false;
}

template<typename SceneT>
void
superVoxelClutterSegmentation(const typename pcl::PointCloud<SceneT>::Ptr & scene_cloud_downsampled,
                              pcl::PointCloud<pcl::PointXYZRGBA> & clusters_cloud_rgb,
                              pcl::PointCloud<pcl::PointXYZL> & clusters_cloud,
                              float radius_neighborhood_GO)
{
    float voxel_resolution = 0.005f;
    float seed_resolution = radius_neighborhood_GO;
    typename pcl::SupervoxelClustering<SceneT> super (voxel_resolution, seed_resolution, false);
    super.setInputCloud (scene_cloud_downsampled);
    super.setColorImportance (0.f);
    super.setSpatialImportance (1.f);
    super.setNormalImportance (1.f);
    std::map <uint32_t, typename pcl::Supervoxel<SceneT>::Ptr > supervoxel_clusters;
    pcl::console::print_highlight ("Extracting supervoxels!\n");
    super.extract (supervoxel_clusters);
    pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());
    pcl::PointCloud<pcl::PointXYZL>::Ptr supervoxels_labels_cloud = super.getLabeledCloud();

    //merge faces...
    uint32_t max_label = super.getMaxLabel();

    pcl::PointCloud<pcl::PointNormal>::Ptr sv_normal_cloud = super.makeSupervoxelNormalCloud (supervoxel_clusters);

    std::vector<int> label_to_idx;
    label_to_idx.resize(max_label + 1, -1);
    typename std::map <uint32_t, typename pcl::Supervoxel<SceneT>::Ptr>::iterator sv_itr,sv_itr_end;
    sv_itr = supervoxel_clusters.begin ();
    sv_itr_end = supervoxel_clusters.end ();
    int idx=0;
    for ( ; sv_itr != sv_itr_end; ++sv_itr, idx++)
    {
        label_to_idx[sv_itr->first] = idx;
    }

    std::vector< std::vector<bool> > adjacent;
    adjacent.resize(supervoxel_clusters.size());
    for(size_t sv_id=0; sv_id < supervoxel_clusters.size(); sv_id++)
        adjacent[sv_id].resize(supervoxel_clusters.size(), false);

    std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
    super.getSupervoxelAdjacency (supervoxel_adjacency);
    //To make a graph of the supervoxel adjacency, we need to iterate through the supervoxel adjacency multimap
    std::multimap<uint32_t,uint32_t>::iterator label_itr = supervoxel_adjacency.begin ();
    std::cout << "super voxel adjacency size:" << supervoxel_adjacency.size() << std::endl;
    for ( ; label_itr != supervoxel_adjacency.end (); )
    {
        //First get the label
        uint32_t supervoxel_label = label_itr->first;
        Eigen::Vector3f normal_super_voxel = sv_normal_cloud->points[label_to_idx[supervoxel_label]].getNormalVector3fMap();
        normal_super_voxel.normalize();
        //Now we need to iterate through the adjacent supervoxels and make a point cloud of them
        std::multimap<uint32_t,uint32_t>::iterator adjacent_itr = supervoxel_adjacency.equal_range (supervoxel_label).first;
        for ( ; adjacent_itr!=supervoxel_adjacency.equal_range (supervoxel_label).second; ++adjacent_itr)
        {
            Eigen::Vector3f normal_neighbor_supervoxel = sv_normal_cloud->points[label_to_idx[adjacent_itr->second]].getNormalVector3fMap();
            normal_neighbor_supervoxel.normalize();

            if(normal_super_voxel.dot(normal_neighbor_supervoxel) > 0.95f)
            {
                adjacent[label_to_idx[supervoxel_label]][label_to_idx[adjacent_itr->second]] = true;
            }
        }

        //Move iterator forward to next label
        label_itr = supervoxel_adjacency.upper_bound (supervoxel_label);
    }

    typedef boost::adjacency_matrix<boost::undirectedS, int> Graph;
    Graph G(supervoxel_clusters.size());
    for(size_t i=0; i < supervoxel_clusters.size(); i++)
    {
        for(size_t j=(i+1); j < supervoxel_clusters.size(); j++)
        {
            if(adjacent[i][j])
                boost::add_edge(i, j, G);
        }
    }

    std::vector<int> components (boost::num_vertices (G));
    int n_cc = static_cast<int> (boost::connected_components (G, &components[0]));
    std::cout << "Number of connected components..." << n_cc << std::endl;

    std::vector<int> cc_sizes;
    std::vector<std::vector<int> > ccs;
    std::vector<uint32_t> original_labels_to_merged;
    original_labels_to_merged.resize(supervoxel_clusters.size());

    ccs.resize(n_cc);
    cc_sizes.resize (n_cc, 0);
    typename boost::graph_traits<Graph>::vertex_iterator vertexIt, vertexEnd;
    boost::tie (vertexIt, vertexEnd) = vertices (G);
    for (; vertexIt != vertexEnd; ++vertexIt)
    {
        int c = components[*vertexIt];
        cc_sizes[c]++;
        ccs[c].push_back(*vertexIt);
        original_labels_to_merged[*vertexIt] = c;
    }

    for(size_t i=0; i < supervoxels_labels_cloud->points.size(); i++)
    {
        //std::cout << supervoxels_labels_cloud->points[i].label << " " << label_to_idx.size() << " " << original_labels_to_merged.size() << " " << label_to_idx[supervoxels_labels_cloud->points[i].label] << std::endl;
        if(label_to_idx[supervoxels_labels_cloud->points[i].label] < 0)
            continue;

        supervoxels_labels_cloud->points[i].label = original_labels_to_merged[label_to_idx[supervoxels_labels_cloud->points[i].label]];
    }

    std::cout << scene_cloud_downsampled->points.size () << " " << supervoxels_labels_cloud->points.size () << std::endl;

    //clusters_cloud_rgb_= super.getColoredCloud();

    clusters_cloud = *supervoxels_labels_cloud;

    std::vector<uint32_t> label_colors_;
    //int max_label = label;
    label_colors_.reserve (max_label + 1);
    srand (static_cast<unsigned int> (time (0)));
    while (label_colors_.size () <= max_label )
    {
        uint8_t r = static_cast<uint8_t>( (rand () % 256));
        uint8_t g = static_cast<uint8_t>( (rand () % 256));
        uint8_t b = static_cast<uint8_t>( (rand () % 256));
        label_colors_.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
    }

    clusters_cloud_rgb.points.resize (scene_cloud_downsampled->points.size ());
    clusters_cloud_rgb.width = scene_cloud_downsampled->points.size();
    clusters_cloud_rgb.height = 1;

    {
        for(size_t i=0; i < clusters_cloud.points.size(); i++)
        {
            //if (clusters_cloud_->points[i].label != 0)
            //{
            clusters_cloud_rgb.points[i].getVector3fMap() = clusters_cloud.points[i].getVector3fMap();
            clusters_cloud_rgb.points[i].rgb = label_colors_[clusters_cloud.points[i].label];

            //}
        }
    }
}

template<>
void
superVoxelClutterSegmentation<pcl::PointXYZ>
(const pcl::PointCloud<pcl::PointXYZ>::Ptr & /*scene_cloud_downsampled_*/,
 pcl::PointCloud<pcl::PointXYZRGBA> & /*clusters_cloud_rgb_*/,
 pcl::PointCloud<pcl::PointXYZL> & /*clusters_cloud_*/,
 float /*radius_neighborhood_GO_*/)
{
    PCL_WARN("Super voxel clutter segmentation not allowed for pcl::PointXYZ scene types\n");
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::segmentScene()
{
    if(param_.use_super_voxels_)
    {
        if( isSuperVoxelClutterSegmentationPossible<SceneT>() ) { //check if its possible at all
            if(!clusters_cloud_rgb_)
                clusters_cloud_rgb_.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
            if(!clusters_cloud_)
                clusters_cloud_.reset(new pcl::PointCloud<pcl::PointXYZL>);
            superVoxelClutterSegmentation<SceneT>(scene_cloud_downsampled_, *clusters_cloud_rgb_, *clusters_cloud_, param_.radius_neighborhood_clutter_);
        }
        else
        {
            PCL_WARN("Not possible to use superVoxelClutter segmentation for pcl::PointXYZ types with pcl version < 1.7.2\n");
            PCL_WARN("See comments in code (ghv.hpp)\n");
            param_.use_super_voxels_ = false;
        }
    }
    else
    {
        if(!clusters_cloud_rgb_)
            clusters_cloud_rgb_.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
        if(!clusters_cloud_)
            clusters_cloud_.reset(new pcl::PointCloud<pcl::PointXYZL>);

        if(scene_cloud_->isOrganized() && scene_normals_for_clutter_term_ && (scene_normals_for_clutter_term_->points.size() == scene_cloud_->points.size()))
        {
            // scene cloud is organized, filter points with high curvature and cluster the rest in smooth patches

            typename pcl::EuclideanClusterComparator<SceneT, pcl::Normal, pcl::Label>::Ptr
                    euclidean_cluster_comparator (new pcl::EuclideanClusterComparator<SceneT, pcl::Normal, pcl::Label> ());

            pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
            labels->points.resize(scene_cloud_->points.size());
            labels->width = scene_cloud_->width;
            labels->height = scene_cloud_->height;
            labels->is_dense = scene_cloud_->is_dense;

            for (size_t j = 0; j < scene_cloud_->points.size (); j++)
            {
                // check XYZ, normal and curvature
                float curvature = scene_normals_for_clutter_term_->points[j].curvature;
                if ( pcl::isFinite( scene_cloud_->points[j] )
                     && pcl::isFinite ( scene_normals_for_clutter_term_->points[j] )
                     && curvature <= (param_.curvature_threshold_ * (std::min(1.f,scene_cloud_->points[j].z))))
                {
                    //label_indices[1].indices[label_count[1]++] = static_cast<int>(j);
                    labels->points[j].label = 1;
                }
                else
                {
                    //label_indices[0].indices[label_count[0]++] = static_cast<int>(j);
                    labels->points[j].label = 0;
                }
            }

            std::vector<bool> excluded_labels;
            excluded_labels.resize (2, false);
            excluded_labels[0] = true;

            euclidean_cluster_comparator->setInputCloud (scene_cloud_);
            euclidean_cluster_comparator->setLabels (labels);
            euclidean_cluster_comparator->setExcludeLabels (excluded_labels);
            euclidean_cluster_comparator->setDistanceThreshold (param_.cluster_tolerance_, true);
            euclidean_cluster_comparator->setAngularThreshold(0.017453 * 5.f); //5 degrees

            pcl::PointCloud<pcl::Label> euclidean_labels;
            std::vector<pcl::PointIndices> clusters;
            pcl::OrganizedConnectedComponentSegmentation<SceneT, pcl::Label> euclidean_segmentation (euclidean_cluster_comparator);
            euclidean_segmentation.setInputCloud (scene_cloud_);
            euclidean_segmentation.segment (euclidean_labels, clusters);

            //                std::cout << "Number of clusters:" << clusters.size() << std::endl;
            std::vector<bool> good_cluster(clusters.size(), false);
            for (size_t i = 0; i < clusters.size (); i++)
            {
                if (clusters[i].indices.size () >= 100)
                    good_cluster[i] = true;
            }

            clusters_cloud_->points.resize (scene_sampled_indices_.size ());
            clusters_cloud_->width = scene_sampled_indices_.size();
            clusters_cloud_->height = 1;

            clusters_cloud_rgb_->points.resize (scene_sampled_indices_.size ());
            clusters_cloud_rgb_->width = scene_sampled_indices_.size();
            clusters_cloud_rgb_->height = 1;

            pcl::PointCloud<pcl::PointXYZL>::Ptr clusters_cloud (new pcl::PointCloud<pcl::PointXYZL>);
            clusters_cloud->points.resize (scene_cloud_->points.size ());
            clusters_cloud->width = scene_cloud_->points.size();
            clusters_cloud->height = 1;

            for (size_t i = 0; i < scene_cloud_->points.size (); i++)
            {
                pcl::PointXYZL p;
                p.getVector3fMap () = scene_cloud_->points[i].getVector3fMap ();
                p.label = 0;
                clusters_cloud->points[i] = p;
                //clusters_cloud_rgb_->points[i].getVector3fMap() = p.getVector3fMap();
                //clusters_cloud_rgb_->points[i].r = clusters_cloud_rgb_->points[i].g = clusters_cloud_rgb_->points[i].b = 100;
            }

            uint32_t label = 1;
            for (size_t i = 0; i < clusters.size (); i++)
            {
                if(!good_cluster[i])
                    continue;

                for (size_t j = 0; j < clusters[i].indices.size (); j++)
                    clusters_cloud->points[clusters[i].indices[j]].label = label;

                label++;
            }

            std::vector<uint32_t> label_colors_;
            int max_label = label;
            label_colors_.reserve (max_label + 1);
            srand (static_cast<unsigned int> (time (0)));
            while (label_colors_.size () <= max_label )
            {
                uint8_t r = static_cast<uint8_t>( (rand () % 256));
                uint8_t g = static_cast<uint8_t>( (rand () % 256));
                uint8_t b = static_cast<uint8_t>( (rand () % 256));
                label_colors_.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
            }

            if(scene_cloud_downsampled_->points.size() != scene_sampled_indices_.size())
            {
                std::cout << scene_cloud_downsampled_->points.size() << " " << scene_sampled_indices_.size() << std::endl;
                assert(scene_cloud_downsampled_->points.size() == scene_sampled_indices_.size());
            }


            for(size_t i=0; i < scene_sampled_indices_.size(); i++)
            {
                clusters_cloud_->points[i] = clusters_cloud->points[scene_sampled_indices_[i]];
                clusters_cloud_rgb_->points[i].getVector3fMap() = clusters_cloud->points[scene_sampled_indices_[i]].getVector3fMap();

                if(clusters_cloud->points[scene_sampled_indices_[i]].label == 0)
                    clusters_cloud_rgb_->points[i].r = clusters_cloud_rgb_->points[i].g = clusters_cloud_rgb_->points[i].b = 100;
                else
                    clusters_cloud_rgb_->points[i].rgb = label_colors_[clusters_cloud->points[scene_sampled_indices_[i]].label];
            }
        }
        else
        {
            std::vector<pcl::PointIndices> clusters;
            extractEuclideanClustersSmooth<SceneT, pcl::Normal> (*scene_cloud_downsampled_, *scene_normals_, param_.cluster_tolerance_,
                                                                 scene_downsampled_tree_, clusters, param_.eps_angle_threshold_,
                                                                 param_.curvature_threshold_, param_.min_points_);

            clusters_cloud_->points.resize (scene_cloud_downsampled_->points.size ());
            clusters_cloud_->width = scene_cloud_downsampled_->width;
            clusters_cloud_->height = 1;

            clusters_cloud_rgb_->points.resize (scene_cloud_downsampled_->points.size ());
            clusters_cloud_rgb_->width = scene_cloud_downsampled_->width;
            clusters_cloud_rgb_->height = 1;

            for (size_t i = 0; i < scene_cloud_downsampled_->points.size (); i++)
            {
                pcl::PointXYZL p;
                p.getVector3fMap () = scene_cloud_downsampled_->points[i].getVector3fMap ();
                p.label = 0;
                clusters_cloud_->points[i] = p;
                clusters_cloud_rgb_->points[i].getVector3fMap() = p.getVector3fMap();
                clusters_cloud_rgb_->points[i].r = clusters_cloud_rgb_->points[i].g = clusters_cloud_rgb_->points[i].b = 100;
            }

            uint32_t label = 1;
            for (size_t i = 0; i < clusters.size (); i++)
            {
                for (size_t j = 0; j < clusters[i].indices.size (); j++)
                    clusters_cloud_->points[clusters[i].indices[j]].label = label;

                label++;
            }

            std::vector<uint32_t> label_colors_;
            int max_label = label;
            label_colors_.reserve (max_label + 1);
            srand (static_cast<unsigned int> (time (0)));
            while (label_colors_.size () <= max_label )
            {
                uint8_t r = static_cast<uint8_t>( (rand () % 256));
                uint8_t g = static_cast<uint8_t>( (rand () % 256));
                uint8_t b = static_cast<uint8_t>( (rand () % 256));
                label_colors_.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
            }

            for(size_t i=0; i < clusters_cloud_->points.size(); i++)
            {
                if (clusters_cloud_->points[i].label != 0)
                    clusters_cloud_rgb_->points[i].rgb = label_colors_[clusters_cloud_->points[i].label];
            }
        }
    }

    max_label_clusters_cloud_ = 0;
    for(size_t i=0; i < clusters_cloud_->points.size(); i++)
    {
        if (clusters_cloud_->points[i].label > max_label_clusters_cloud_)
            max_label_clusters_cloud_ = clusters_cloud_->points[i].label;
    }
}


template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::convertColor()
{

    //compute cloud LAB values for model visible points
    size_t num_color_channels = 0;
    switch (param_.color_space_)
    {
    case ColorSpace::LAB: case ColorSpace::RGB: num_color_channels = 3; break;
    case ColorSpace::GRAYSCALE: num_color_channels = 1; break;
    default: throw std::runtime_error("Color space not implemented!");
    }

    scene_color_channels_ = Eigen::MatrixXf::Zero ( scene_cloud_downsampled_->points.size(), num_color_channels);

#pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i < scene_cloud_downsampled_->points.size(); i++)
    {
        float rgb_s = 0.f;
        bool exists_s;
        pcl::for_each_type<FieldListS> (
                    pcl::CopyIfFieldExists<typename CloudS::PointType, float> (scene_cloud_downsampled_->points[i],
                                                                               "rgb", exists_s, rgb_s));
        if (exists_s)
        {
            uint32_t rgb = *reinterpret_cast<int*> (&rgb_s);
            unsigned char rs = (rgb >> 16) & 0x0000ff;
            unsigned char gs = (rgb >> 8) & 0x0000ff;
            unsigned char bs = (rgb) & 0x0000ff;
            float rsf,gsf,bsf;
            rsf = static_cast<float>(rs) / 255.f;
            gsf = static_cast<float>(gs) / 255.f;
            bsf = static_cast<float>(bs) / 255.f;


            switch (param_.color_space_)
            {
            case ColorSpace::LAB:
                float LRefs, aRefs, bRefs;
                color_transf_omp_.RGB2CIELAB_normalized(rs, gs, bs, LRefs, aRefs, bRefs);

                scene_color_channels_(i, 0) = LRefs;
                scene_color_channels_(i, 1) = aRefs;
                scene_color_channels_(i, 2) = bRefs;
                break;
            case ColorSpace::RGB:
                scene_color_channels_(i, 0) = rsf;
                scene_color_channels_(i, 1) = gsf;
                scene_color_channels_(i, 2) = bsf;
            case ColorSpace::GRAYSCALE:
                scene_color_channels_(i, 0) = .2126 * rsf + .7152 * gsf + .0722 * bsf;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename ModelT, typename SceneT>
bool
GHV<ModelT, SceneT>::initialize()
{
    //clear stuff
    unexplained_by_RM_neighboorhods_.clear ();
    explained_by_RM_distance_weighted_.clear ();
    previous_explained_by_RM_distance_weighted_.clear ();
    explained_by_RM_.clear ();
    explained_by_RM_model_.clear();
    complete_cloud_occupancy_by_RM_.clear ();
    mask_.clear ();
    mask_.resize (recognition_models_.size (), false);

    if(!scene_and_normals_set_from_outside_ || scene_cloud_downsampled_->points.size() != scene_normals_->points.size())
    {
        size_t kept = 0;
        for (size_t i = 0; i < scene_cloud_downsampled_->points.size (); ++i) {
            if ( pcl::isFinite( scene_cloud_downsampled_->points[i]) )
                scene_cloud_downsampled_->points[kept++] = scene_cloud_downsampled_->points[i];
        }

        scene_cloud_downsampled_->points.resize(kept);
        scene_cloud_downsampled_->width = kept;
        scene_cloud_downsampled_->height = 1;


        if(!scene_normals_)
            scene_normals_.reset (new pcl::PointCloud<pcl::Normal> ());
        computeNormals<SceneT>(scene_cloud_downsampled_, scene_normals_, param_.normal_method_);

        //check nans...
        kept = 0;
        for (size_t i = 0; i < scene_normals_->points.size (); ++i)
        {
            if ( pcl::isFinite( scene_normals_->points[i] ) )
            {
                scene_normals_->points[kept] = scene_normals_->points[i];
                scene_cloud_downsampled_->points[kept] = scene_cloud_downsampled_->points[i];
                scene_sampled_indices_[kept] = scene_sampled_indices_[i];
                kept++;
            }
        }
        scene_sampled_indices_.resize(kept);
        scene_normals_->points.resize (kept);
        scene_cloud_downsampled_->points.resize (kept);
        scene_cloud_downsampled_->width = scene_normals_->width = kept;
        scene_cloud_downsampled_->height = scene_normals_->height = 1;
    }
    else
    {
        scene_sampled_indices_.resize(scene_cloud_downsampled_->points.size());

        for(size_t k=0; k < scene_cloud_downsampled_->points.size(); k++)
            scene_sampled_indices_[k] = k;
    }

    explained_by_RM_.resize (scene_cloud_downsampled_->points.size (), 0);
    duplicates_by_RM_weighted_.resize (scene_cloud_downsampled_->points.size (), 0);
    explained_by_RM_model_.resize (scene_cloud_downsampled_->points.size (), -1);
    explained_by_RM_distance_weighted_.resize (scene_cloud_downsampled_->points.size (), 0);
    previous_explained_by_RM_distance_weighted_.resize (scene_cloud_downsampled_->points.size ());
    unexplained_by_RM_neighboorhods_.resize (scene_cloud_downsampled_->points.size (), 0.f);

    octree_scene_downsampled_.reset(new pcl::octree::OctreePointCloudSearch<SceneT>(0.01f));
    octree_scene_downsampled_->setInputCloud(scene_cloud_downsampled_);
    octree_scene_downsampled_->addPointsFromInputCloud();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (param_.detect_clutter_)
            {
                pcl::ScopeTime t("Smooth segmentation of the scene");
                //initialize kdtree for search
                scene_downsampled_tree_.reset (new pcl::search::KdTree<SceneT>);
                scene_downsampled_tree_->setInputCloud (scene_cloud_downsampled_);

                segmentScene();
            }
        }

        #pragma omp section
        {
            pcl::ScopeTime t("Converting scene color values");
            if(!param_.ignore_color_even_if_exists_)
                convertColor();
        }
    }

    // we need to know the color of scene points before we go on
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            pcl::ScopeTime t("Computing cues for recognition models");
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < recognition_models_.size (); i++)
                addModel(*recognition_models_[i]);
        }

        #pragma omp section
        {
            pcl::ScopeTime t("compute cloud occupancy by recognition models");
            computeSceneOccupancyGridByRM();
        }
    }

    {
        pcl::ScopeTime t("Compute clutter cue at once");
        computeClutterCueAtOnce();
    }

    // visualize cues
//    for (const auto & rm:recognition_models_)
//        visualizeGOCuesForModel(*rm);

    rm_ids_explaining_scene_pt_.clear ();
    rm_ids_explaining_scene_pt_.resize (scene_cloud_downsampled_->points.size ());
    for (size_t j = 0; j < recognition_models_.size (); j++)
    {
        HVRecognitionModel<ModelT> &rm = *recognition_models_[j];
        for (const auto &scene_pt_id : rm.explained_scene_indices_)
            rm_ids_explaining_scene_pt_[ scene_pt_id ].push_back (j);
    }

    return true;
}

template<typename ModelT, typename SceneT>
double
GHV<ModelT, SceneT>::getCurvWeight(double p_curvature) const
{

    if( param_.multiple_assignment_penalize_by_one_ == 2 )
        return 1.f;

    //return 1.f;

    /*if(p_curvature > duplicity_curvature_max)
        return 0.f;*/

    /*if(p_curvature < duplicity_curvature_max)
        return 1.f;*/

    return 1.f - std::min(1., p_curvature / param_.duplicity_curvature_max_);
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::updateExplainedVector (const std::vector<int> & vec, const std::vector<float> & vec_float, int sign, int model_id)
{
    double add_to_explained = 0.f;
    double add_to_duplicity_ = 0.f;

    for (size_t i = 0; i < vec.size (); i++)
    {
        bool prev_dup = explained_by_RM_[vec[i]] > 1;
        //bool prev_explained = explained_[vec[i]] == 1;
        int prev_explained = explained_by_RM_[vec[i]];
        double prev_explained_value = explained_by_RM_distance_weighted_[vec[i]];

        explained_by_RM_[vec[i]] += sign;
        //explained_by_RM_distance_weighted[vec[i]] += vec_float[i] * sign;

        if(sign > 0)
        {
            //adding, check that after adding the hypothesis, explained_by_RM_distance_weighted[vec[i]] is not higher than 1
            /*if(prev_explained_value + vec_float[i] > 1.f)
            {
                add_to_explained += std::max(0.0, 1 - prev_explained_value);
            }
            else
            {
                add_to_explained += vec_float[i];
            }*/

            if(prev_explained == 0)
            {
                //point was unexplained
                explained_by_RM_distance_weighted_[vec[i]] = vec_float[i];
                previous_explained_by_RM_distance_weighted_[vec[i]].push(std::make_pair(model_id, vec_float[i]));
            }
            else
            {
                //point was already explained
                if(vec_float[i] > prev_explained_value)
                {
                    previous_explained_by_RM_distance_weighted_[vec[i]].push(std::make_pair(model_id, vec_float[i]));
                    explained_by_RM_distance_weighted_[vec[i]] = (double)vec_float[i];
                }
                else
                {
                    //if it is smaller, we should keep the value in case the greater value gets removed
                    //need to sort the stack
                    if(previous_explained_by_RM_distance_weighted_[vec[i]].size() == 0)
                    {
                        previous_explained_by_RM_distance_weighted_[vec[i]].push(std::make_pair(model_id, vec_float[i]));
                    }
                    else
                    {
                        //sort and find the appropiate position

                        std::stack<std::pair<int, float>, std::vector<std::pair<int, float> > > kept;
                        while(previous_explained_by_RM_distance_weighted_[vec[i]].size() > 0)
                        {
                            std::pair<int, double> p = previous_explained_by_RM_distance_weighted_[vec[i]].top();
                            if(p.second < vec_float[i])
                            {
                                //should come here
                                break;
                            }

                            kept.push(p);
                            previous_explained_by_RM_distance_weighted_[vec[i]].pop();
                        }

                        previous_explained_by_RM_distance_weighted_[vec[i]].push(std::make_pair(model_id, vec_float[i]));

                        while(!kept.empty())
                        {
                            previous_explained_by_RM_distance_weighted_[vec[i]].push(kept.top());
                            kept.pop();
                        }
                    }
                }
            }
        }
        else
        {
            std::stack<std::pair<int, float>, std::vector<std::pair<int, float> > > kept;

            while(previous_explained_by_RM_distance_weighted_[vec[i]].size() > 0)
            {
                std::pair<int, double> p = previous_explained_by_RM_distance_weighted_[vec[i]].top();

                if(p.first == model_id)
                {
                    //found it
                }
                else
                {
                    kept.push(p);
                }

                previous_explained_by_RM_distance_weighted_[vec[i]].pop();
            }

            while(!kept.empty())
            {
                previous_explained_by_RM_distance_weighted_[vec[i]].push(kept.top());
                kept.pop();
            }

            if(prev_explained == 1)
            {
                //was only explained by this hypothesis
                explained_by_RM_distance_weighted_[vec[i]] = 0;
            }
            else
            {
                //there is at least another hypothesis explaining this point
                //assert(previous_explained_by_RM_distance_weighted[vec[i]].size() > 0);
                std::pair<int, double> p = previous_explained_by_RM_distance_weighted_[vec[i]].top();

                double previous = p.second;
                explained_by_RM_distance_weighted_[vec[i]] = previous;
            }

            //}

            /*if(prev_explained_value > 1.f)
            {
                if((prev_explained_value - vec_float[i]) < 1.f)
                {
                    add_to_explained -= 1.f - (prev_explained_value - vec_float[i]);
                }
            }
            else
            {
                add_to_explained -= vec_float[i];
            }*/
        }

        //add_to_explained += vec_float[i] * sign;
        //float curv_weight = std::min(duplicity_curvature_ - scene_curvature_[vec[i]], 0.f);
        float curv_weight = getCurvWeight( scene_normals_->points[ vec[i] ].curvature);
        /*if (explained_[vec[i]] == 1 && !prev_explained)
        {
            if (sign > 0)
            {
                add_to_explained += vec_float[i];
            }
            else
            {
                add_to_explained += explained_by_RM_distance_weighted[vec[i]];
            }
        }

        //hypotheses being removed, now the point is not explained anymore and was explained before by this hypothesis
        if ((sign < 0) && (explained_[vec[i]] == 0) && prev_explained)
        {
            //assert(prev_explained_value == vec_float[i]);
            add_to_explained -= prev_explained_value;
        }

      //this hypothesis was added and now the point is not explained anymore, remove previous value (it is a duplicate)
      if ((sign > 0) && (explained_[vec[i]] == 2) && prev_explained)
        add_to_explained -= prev_explained_value;*/

        if(param_.multiple_assignment_penalize_by_one_ == 1)
        {
            if ((explained_by_RM_[vec[i]] > 1) && prev_dup)
            { //its still a duplicate, do nothing

            }
            else if ((explained_by_RM_[vec[i]] == 1) && prev_dup)
            { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
                add_to_duplicity_ -= curv_weight;
            }
            else if ((explained_by_RM_[vec[i]] > 1) && !prev_dup)
            { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
                add_to_duplicity_ += curv_weight;
            }
        }
        else if( param_.multiple_assignment_penalize_by_one_ == 2)
        {
            if ((explained_by_RM_[vec[i]] > 1) && prev_dup)
            { //its still a duplicate, add or remove current explained value

                //float add_for_this_p = std::
                add_to_duplicity_ += curv_weight * vec_float[i] * sign;
                duplicates_by_RM_weighted_[vec[i]] += curv_weight * vec_float[i] * sign;
            }
            else if ((explained_by_RM_[vec[i]] == 1) && prev_dup)
            { //if was duplicate before, now its not, remove current explained weight and old one
                add_to_duplicity_ -= duplicates_by_RM_weighted_[vec[i]];
                duplicates_by_RM_weighted_[vec[i]] = 0;
            }
            else if ((explained_by_RM_[vec[i]] > 1) && !prev_dup)
            { //it was not a duplicate but it is now, add prev explained value + current explained weight
                add_to_duplicity_ += curv_weight * (prev_explained_value + vec_float[i]);
                duplicates_by_RM_weighted_[vec[i]] = curv_weight * (prev_explained_value + vec_float[i]);
            }
        }
        else
        {
            if ((explained_by_RM_[vec[i]] > 1) && prev_dup)
            { //its still a duplicate
                //add_to_duplicity_ += vec_float[i] * static_cast<int> (sign); //so, just add or remove one
                //add_to_duplicity_ += vec_float[i] * static_cast<int> (sign) * duplicy_weight_test_ * curv_weight; //so, just add or remove one
                add_to_duplicity_ += static_cast<int> (sign) * param_.duplicy_weight_test_ * curv_weight; //so, just add or remove one
            }
            else if ((explained_by_RM_[vec[i]] == 1) && prev_dup)
            { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
                //add_to_duplicity_ -= prev_explained_value; // / 2.f; //explained_by_RM_distance_weighted[vec[i]];
                //add_to_duplicity_ -= prev_explained_value * duplicy_weight_test_ * curv_weight;
                add_to_duplicity_ -= param_.duplicy_weight_test_ * curv_weight * 2;
            }
            else if ((explained_by_RM_[vec[i]] > 1) && !prev_dup)
            { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
                //add_to_duplicity_ += explained_by_RM_distance_weighted[vec[i]];
                //add_to_duplicity_ += explained_by_RM_distance_weighted[vec[i]] * duplicy_weight_test_ * curv_weight;
                add_to_duplicity_ += param_.duplicy_weight_test_ * curv_weight  * 2;
            }
        }

        add_to_explained += explained_by_RM_distance_weighted_[vec[i]] - prev_explained_value;
    }

    //update explained and duplicity values...
    previous_explained_value_ += add_to_explained;
    previous_duplicity_ += add_to_duplicity_;
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::updateCMDuplicity (const std::vector<int> & vec, int sign)
{
    int add_to_duplicity_ = 0;
    for (size_t i = 0; i < vec.size (); i++)
    {
        if( (vec[i] > complete_cloud_occupancy_by_RM_.size() ) || ( i > vec.size()))
        {
            std::cout << complete_cloud_occupancy_by_RM_.size() << " " << vec[i] << " " << vec.size() << " " << i << std::endl;
            throw std::runtime_error("something is wrong with the occupancy grid.");
        }

        bool prev_dup = complete_cloud_occupancy_by_RM_[vec[i]] > 1;
        complete_cloud_occupancy_by_RM_[vec[i]] += static_cast<int> (sign);
        if ((complete_cloud_occupancy_by_RM_[vec[i]] > 1) && prev_dup)
        { //its still a duplicate, we are adding
            add_to_duplicity_ += static_cast<int> (sign); //so, just add or remove one
        }
        else if ((complete_cloud_occupancy_by_RM_[vec[i]] == 1) && prev_dup)
        { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
            add_to_duplicity_ -= 2;
        }
        else if ((complete_cloud_occupancy_by_RM_[vec[i]] > 1) && !prev_dup)
        { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
            add_to_duplicity_ += 2;
        }
    }

    previous_duplicity_complete_models_ += add_to_duplicity_;
}

template<typename ModelT, typename SceneT>
double
GHV<ModelT, SceneT>::getTotalExplainedInformation (const std::vector<int> & explained, const std::vector<double> & explained_by_RM_distance_weighted, double &duplicity)
{
    double explained_info = 0;
    duplicity = 0;

    for (size_t i = 0; i < explained.size (); i++)
    {
        if (explained[i] > 0)
            explained_info += explained_by_RM_distance_weighted[i];

        if (explained[i] > 1)
        {
            float curv_weight = getCurvWeight( scene_normals_->points[i].curvature );

            if(param_.multiple_assignment_penalize_by_one_ == 1)
                duplicity += curv_weight;
            else if(param_.multiple_assignment_penalize_by_one_ == 2)
                duplicity += duplicates_by_RM_weighted_[i];
            else
                duplicity += param_.duplicy_weight_test_ * curv_weight * explained[i];
        }
    }

    return explained_info;
}

template<typename ModelT, typename SceneT>
double
GHV<ModelT, SceneT>::getExplainedByIndices(const std::vector<int> & indices, const std::vector<float> & explained_values,
                                           const std::vector<double> & explained_by_RM, std::vector<int> & indices_to_update_in_RM_local) const
{
    float v=0;
    int indices_to_update_count = 0;
    for(size_t k=0; k < indices.size(); k++)
    {
        if(explained_by_RM_[indices[k]] == 0)
        { //in X1, the point is not explained
            if(explained_by_RM[indices[k]] == 0)
            { //in X2, this is the single hypothesis explaining the point so far
                v += explained_values[k];
                indices_to_update_in_RM_local[indices_to_update_count] = k;
                indices_to_update_count++;
            }
            else
            {
                //in X2, there was a previous hypotheses explaining the point
                //if the previous hypothesis was better, then reject this hypothesis for this point
                if(explained_by_RM[indices[k]] >= explained_values[k])
                {

                }
                else
                {
                    //add the difference
                    v += explained_values[k] - explained_by_RM[indices[k]];
                    indices_to_update_in_RM_local[indices_to_update_count] = k;
                    indices_to_update_count++;
                }
            }
        }
    }

    indices_to_update_in_RM_local.resize(indices_to_update_count);
    return v;
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::fill_structures(const std::vector<bool> & initial_solution, GHVSAModel<ModelT, SceneT> & model)
{
    for (size_t j = 0; j < recognition_models_.size (); j++)
    {
        if(!initial_solution[j])
            continue;

        boost::shared_ptr<HVRecognitionModel<ModelT> > recog_model = recognition_models_[j];
        for (size_t i = 0; i < recog_model->explained_scene_indices_.size (); i++)
        {
            explained_by_RM_[recog_model->explained_scene_indices_[i]]++;
            //explained_by_RM_distance_weighted[recog_model->explained_[i]] += recog_model->explained_distances_[i];
            explained_by_RM_distance_weighted_[recog_model->explained_scene_indices_[i]] = std::max(explained_by_RM_distance_weighted_[recog_model->explained_scene_indices_[i]], (double)recog_model->distances_to_explained_scene_indices_[i]);
        }

        if (param_.detect_clutter_)
        {
            for (size_t i = 0; i < recog_model->unexplained_in_neighborhood.size (); i++)
                unexplained_by_RM_neighboorhods_[recog_model->unexplained_in_neighborhood[i]] += recog_model->unexplained_in_neighborhood_weights[i];
        }

        for (size_t i = 0; i < recog_model->complete_cloud_occupancy_indices_.size (); i++)
        {
            int idx = recog_model->complete_cloud_occupancy_indices_[i];
            complete_cloud_occupancy_by_RM_[idx]++;
        }
    }

    //another pass to update duplicates_by_RM_weighted_ (only if multiple_assignment_penalize_by_one_ == 2)
    for (size_t j = 0; j < recognition_models_.size (); j++)
    {
        if(!initial_solution[j])
            continue;

        boost::shared_ptr<HVRecognitionModel<ModelT> > recog_model = recognition_models_[j];
        for (size_t i = 0; i < recog_model->explained_scene_indices_.size (); i++)
        {
            if(explained_by_RM_[recog_model->explained_scene_indices_[i]] > 1)
            {
                float curv_weight = getCurvWeight( scene_normals_->points[ recog_model->explained_scene_indices_[i] ].curvature );
                duplicates_by_RM_weighted_[recog_model->explained_scene_indices_[i]] += curv_weight * (double)recog_model->distances_to_explained_scene_indices_[i];
            }
        }
    }

    int occupied_multiple = 0;
    for (size_t i = 0; i < complete_cloud_occupancy_by_RM_.size (); i++)
    {
        if (complete_cloud_occupancy_by_RM_[i] > 1)
            occupied_multiple += complete_cloud_occupancy_by_RM_[i];
    }

    //do optimization
    //Define model SAModel, initial solution is all models activated

    double duplicity;
    double good_information = getTotalExplainedInformation (explained_by_RM_, explained_by_RM_distance_weighted_, duplicity);
    double bad_information = 0;
    double unexplained_in_neighboorhod = 0;

    if(param_.detect_clutter_)
        unexplained_in_neighboorhod = getUnexplainedInformationInNeighborhood (unexplained_by_RM_neighboorhods_, explained_by_RM_);

    for (size_t i = 0; i < initial_solution.size (); i++)
    {
        if (initial_solution[i])
            bad_information += static_cast<double> (recognition_models_[i]->getOutliersWeight()) * recognition_models_[i]->outlier_indices_.size ();
    }

    previous_duplicity_complete_models_ = occupied_multiple;
    previous_explained_value_ = good_information;
    previous_duplicity_ = duplicity;
    previous_bad_info_ = bad_information;
    previous_unexplained_ = unexplained_in_neighboorhod;

    model.cost_ = static_cast<mets::gol_type> ((good_information - bad_information - static_cast<double> (duplicity)
                                                - static_cast<double> (occupied_multiple) * param_.w_occupied_multiple_cm_ -
                                                - unexplained_in_neighboorhod - countActiveHypotheses (initial_solution) - countPointsOnDifferentPlaneSides(initial_solution)) * -1.f);

    model.setSolution (initial_solution);
    model.setOptimizer (this);

    //    std::cout << "*****************************" << std::endl;
    //    std::cout << "Cost recomputing:" << model.cost_ << std::endl;

    //    //std::cout << countActiveHypotheses (initial_solution) << " points on diff plane sides:" << countPointsOnDifferentPlaneSides(initial_solution, false) << std::endl;
    //    std::cout << "*****************************" << std::endl;
    //    std::cout << std::endl;
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::clear_structures()
{
    size_t kk = complete_cloud_occupancy_by_RM_.size();
    explained_by_RM_.clear();
    explained_by_RM_distance_weighted_.clear();
    previous_explained_by_RM_distance_weighted_.clear();
    unexplained_by_RM_neighboorhods_.clear();
    complete_cloud_occupancy_by_RM_.clear();
    explained_by_RM_model_.clear();
    duplicates_by_RM_weighted_.clear();

    explained_by_RM_.resize (scene_cloud_downsampled_->points.size (), 0);
    duplicates_by_RM_weighted_.resize (scene_cloud_downsampled_->points.size (), 0);
    explained_by_RM_distance_weighted_.resize (scene_cloud_downsampled_->points.size (), 0);
    previous_explained_by_RM_distance_weighted_.resize (scene_cloud_downsampled_->points.size ());
    unexplained_by_RM_neighboorhods_.resize (scene_cloud_downsampled_->points.size (), 0.f);
    complete_cloud_occupancy_by_RM_.resize(kk, 0);
    explained_by_RM_model_.resize (scene_cloud_downsampled_->points.size (), -1);
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::SAOptimize (std::vector<int> & cc_indices, std::vector<bool> & initial_solution)
{
    //temporal copy of recogniton_models_
    std::vector<boost::shared_ptr<HVRecognitionModel<ModelT> > > recognition_models_copy = recognition_models_;

    recognition_models_.resize( cc_indices.size () );

    for (size_t j = 0; j < cc_indices.size (); j++)
        recognition_models_[j] = recognition_models_copy[ cc_indices[j] ];

    clear_structures();

    GHVSAModel<ModelT, SceneT> model;
    fill_structures(initial_solution, model);

    GHVSAModel<ModelT, SceneT> * best = new GHVSAModel<ModelT, SceneT> (model);

    GHVmove_manager<ModelT, SceneT> neigh (static_cast<int> (cc_indices.size ()), param_.use_replace_moves_);
    boost::shared_ptr<std::map< std::pair<int, int>, bool > > intersect_map;
    intersect_map.reset(new std::map< std::pair<int, int>, bool >);

    if(param_.use_replace_moves_)
    {
        pcl::ScopeTime t("compute intersection map...");

        std::vector<size_t> n_conflicts(recognition_models_.size() * recognition_models_.size(), 0);
        for (size_t k = 0; k < rm_ids_explaining_scene_pt_.size(); k++)
        {
            if (rm_ids_explaining_scene_pt_[k].size() > 1)
            {
                // this point could be a conflict
                for (size_t kk = 0; kk < rm_ids_explaining_scene_pt_[k].size (); kk++)
                {
                    for (size_t jj = kk+1; jj < rm_ids_explaining_scene_pt_[k].size (); jj++)
                    {
                        n_conflicts[rm_ids_explaining_scene_pt_[k][kk] * recognition_models_.size() + rm_ids_explaining_scene_pt_[k][jj]]++;
                        n_conflicts[rm_ids_explaining_scene_pt_[k][jj] * recognition_models_.size() + rm_ids_explaining_scene_pt_[k][kk]]++;
                    }
                }
            }
        }

        int num_conflicts = 0;
        for (size_t i = 0; i < recognition_models_.size (); i++)
        {
            for (size_t j = (i+1); j < recognition_models_.size (); j++)
            {
                //assert(n_conflicts[i * recognition_models_.size() + j] == n_conflicts[j * recognition_models_.size() + i]);
                //std::cout << n_conflicts[i * recognition_models_.size() + j] << std::endl;
                bool conflict = (n_conflicts[i * recognition_models_.size() + j] > 10);
                std::pair<int, int> p = std::make_pair<int, int> (static_cast<int> (i), static_cast<int> (j));
                (*intersect_map)[p] = conflict;
                if(conflict)
                    num_conflicts++;
            }
        }

        //#define VIS_PLANES

        if(param_.use_points_on_plane_side_)
        {
#ifdef VIS_PLANES
            pcl::visualization::PCLVisualizer vis("TEST");
#endif
            points_on_plane_sides_ = Eigen::MatrixXf::Zero(recognition_models_.size(), recognition_models_.size());

            for(size_t i=0; i < recognition_models_.size(); i++)
            {
                const HVRecognitionModel<ModelT> &rm_i = *recognition_models_[i];
                if( rm_i.is_planar_ )
                {
                    //is a plane, check how many points from other hypotheses are at each side of the plane
                    for(size_t j=0; j < recognition_models_.size(); j++)
                    {
                        const HVRecognitionModel<ModelT> &rm_j =  *recognition_models_[j];

                        if( rm_j.is_planar_ ) //both are planes, ignore
                            continue;

                        bool conflict = (n_conflicts[ i * recognition_models_.size() + j ] > 0);
                        if(!conflict)
                            continue;

                        //is not a plane and is in conflict, compute points on both sides
                        Eigen::Vector2f side_count = Eigen::Vector2f::Zero();
                        for(size_t k=0; k < rm_j.complete_cloud_->points.size(); k++)
                        {
                            const std::vector<float> &p = rm_i.plane_model_.coefficients_.values;
                            const Eigen::Vector3f &xyz_p = rm_j.complete_cloud_->points[k].getVector3fMap();
                            float val = xyz_p[0] * p[0] + xyz_p[1] * p[1] + xyz_p[2] * p[2] + p[3];

                            if(std::abs(val) <= param_.inliers_threshold_)
                                continue;

                            if(val < 0)
                                side_count[0]+= 1.f;
                            else
                                side_count[1]+= 1.f;
                        }

                        float min_side = std::min(side_count[0],side_count[1]);
                        float max_side = std::max(side_count[0],side_count[1]);
                        //float ratio = static_cast<float>(min_side) / static_cast<float>(max_side); //between 0 and 1
                        if(max_side != 0)
                        {
                            points_on_plane_sides_(i, j) = min_side;
#ifdef VIS_PLANES
                            vis.addPointCloud<SceneT>(scene_cloud_downsampled_, "scene");
                            vis.addPointCloud<ModelT>(recognition_models_[j]->complete_cloud_, "complete_cloud");
                            vis.addPolygonMesh(*(planar_models_[it1->second].convex_hull_), "polygon");
                            vis.spin();
                            vis.removeAllPointClouds();
                            vis.removeAllShapes();
#endif
                        }
                    }
                }
            }
        }
        std::cout << "num_conflicts:" << num_conflicts << " " << recognition_models_.size() * recognition_models_.size() << std::endl;
    }

    neigh.setExplainedPointIntersections(intersect_map);

    //mets::best_ever_solution best_recorder (best);
    cost_logger_.reset(new GHVCostFunctionLogger<ModelT, SceneT>(*best));
    mets::noimprove_termination_criteria noimprove (param_.max_iterations_);

    if(param_.visualize_go_cues_)
        cost_logger_->setVisualizeFunction(visualize_cues_during_logger_);

    switch( param_.opt_type_ )
    {
    case 0:
    {
        mets::local_search<GHVmove_manager<ModelT, SceneT> > local ( model, *(cost_logger_.get()), neigh, 0, false);
        {
            pcl::ScopeTime t ("local search...");
            local.search ();
        }
        break;
    }
    case 1:
    {
        //Tabu search
        //mets::simple_tabu_list tabu_list ( initial_solution.size() * sqrt ( 1.0*initial_solution.size() ) ) ;
        mets::simple_tabu_list tabu_list ( 5 * initial_solution.size()) ;
        mets::best_ever_criteria aspiration_criteria ;

        std::cout << "max iterations:" << param_.max_iterations_ << std::endl;
        mets::tabu_search<GHVmove_manager<ModelT, SceneT> > tabu_search(model,  *(cost_logger_.get()), neigh, tabu_list, aspiration_criteria, noimprove);
        //mets::tabu_search<move_manager> tabu_search(model, best_recorder, neigh, tabu_list, aspiration_criteria, noimprove);

        {
            pcl::ScopeTime t ("TABU search...");
            try {
                tabu_search.search ();
            } catch (mets::no_moves_error e) {
                //} catch (std::exception e) {

            }
        }
        break;
    }
    case 4:
    {
        GHVmove_manager<ModelT, SceneT> neigh4 (static_cast<int> (cc_indices.size ()), false);
        neigh4.setExplainedPointIntersections(intersect_map);

        mets::simple_tabu_list tabu_list ( initial_solution.size() * sqrt ( 1.0*initial_solution.size() ) ) ;
        mets::best_ever_criteria aspiration_criteria ;
        mets::tabu_search<GHVmove_manager<ModelT, SceneT> > tabu_search(model,  *(cost_logger_.get()), neigh4, tabu_list, aspiration_criteria, noimprove);
        //mets::tabu_search<move_manager> tabu_search(model, best_recorder, neigh, tabu_list, aspiration_criteria, noimprove);

        {
            pcl::ScopeTime t_tabu ("TABU search + LS (RM)...");
            try {
                tabu_search.search ();
            } catch (mets::no_moves_error e) {

            }

            std::cout << "Tabu search finished... starting LS with RM" << std::endl;

            //after TS, we do LS with RM
            GHVmove_manager<ModelT, SceneT> neigh4RM (static_cast<int> (cc_indices.size ()), true);
            neigh4RM.setExplainedPointIntersections(intersect_map);

            mets::local_search<GHVmove_manager<ModelT, SceneT> > local ( model, *(cost_logger_.get()), neigh4RM, 0, false);
            {
                pcl::ScopeTime t_local_search ("local search...");
                local.search ();
                (void)t_local_search;
            }
        }
        break;

    }
    default:
    {
        //Simulated Annealing
        //mets::linear_cooling linear_cooling;
        mets::exponential_cooling linear_cooling;
        mets::simulated_annealing<GHVmove_manager<ModelT, SceneT> > sa (model,  *(cost_logger_.get()), neigh, noimprove, linear_cooling, initial_temp_, 1e-7, 1);
        sa.setApplyAndEvaluate (true);

        {
            pcl::ScopeTime t ("SA search...");
            sa.search ();
        }
        break;
    }
    }

    best_seen_ = static_cast<const GHVSAModel<ModelT, SceneT>&> (cost_logger_->best_seen ());
    std::cout << "*****************************" << std::endl;
    std::cout << "Final cost:" << best_seen_.cost_;
    std::cout << " Number of ef evaluations:" << cost_logger_->getTimesEvaluated();
    std::cout << std::endl;
    std::cout << "Number of accepted moves:" << cost_logger_->getAcceptedMovesSize() << std::endl;
    std::cout << "*****************************" << std::endl;

    for (size_t i = 0; i < best_seen_.solution_.size (); i++) {
        initial_solution[i] = best_seen_.solution_[i];
    }

    //pcl::visualization::PCLVisualizer vis_ ("test histograms");

    for(size_t i = 0; i < initial_solution.size(); i++) {
        if(initial_solution[i]) {
            //            std::cout << "id: " << recognition_models_[i]->id_s_ << std::endl;
            //            /*std::cout << "Median:" << recognition_models_[i]->median_ << std::endl;
            //            std::cout << "Mean:" << recognition_models_[i]->mean_ << std::endl;
            //            std::cout << "Color similarity:" << recognition_models_[i]->color_similarity_ << std::endl;*/
            //            std::cout << "#outliers:" << recognition_models_[i]->outlier_indices_.size () << " " << recognition_models_[i]->outliers_weight_ << std::endl;
            //            //std::cout << "#under table:" << recognition_models_[i]->model_constraints_value_ << std::endl;
            //            std::cout << "#explained:" << recognition_models_[i]->explained_.size() << std::endl;
            //            std::cout << "normal entropy:" << recognition_models_[i]->normal_entropy_ << std::endl;
            //            std::cout << "color entropy:" << recognition_models_[i]->color_entropy_ << std::endl;
            //            std::cout << "hyp penalty:" << recognition_models_[i]->hyp_penalty_ << std::endl;
            //            std::cout << "color diff:" << recognition_models_[i]->color_diff_trhough_specification_ << std::endl;
            //            std::cout << "Mean:" << recognition_models_[i]->mean_ << std::endl;
        }
    }

    delete best;

    {
        //check results
        GHVSAModel<ModelT, SceneT> _model;
        clear_structures();
        fill_structures(initial_solution, _model);
    }

    recognition_models_ = recognition_models_copy;

}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::verify()
{
    {
        pcl::ScopeTime t_init("initialization");
        if (!initialize ())
            return;
    }

    if(param_.visualize_go_cues_)
        visualize_cues_during_logger_ = boost::bind(&GHV<ModelT, SceneT>::visualizeGOCues, this, _1, _2, _3);

    n_cc_ = 1;
    cc_.resize(1);
    cc_[0].resize(recognition_models_.size());
    for(size_t i=0; i < recognition_models_.size(); i++)
        cc_[0][i] = static_cast<int>(i);

    //for each connected component, find the optimal solution
    {
        pcl::ScopeTime t("Optimizing object hypotheses verification cost function");

        for (size_t c = 0; c < n_cc_; c++)
        {
            //TODO: Check for trivial case...
            //TODO: Check also the number of hypotheses and use exhaustive enumeration if smaller than 10
            std::vector<bool> subsolution (cc_[c].size (), param_.initial_status_);
            SAOptimize (cc_[c], subsolution);

            for (size_t i = 0; i < subsolution.size (); i++)
                mask_[cc_[c][i]] = subsolution[i];
        }
    }

    recognition_models_.clear();
}

inline void softBining(float val, int pos1, float bin_size, int max_pos, int & pos2, float & w1, float & w2) {
    float c1 = pos1 * bin_size + bin_size / 2;
    pos2 = 0;
    float c2 = 0;
    if(pos1 == 0)
    {
        pos2 = 1;
        c2 = pos2 * bin_size + bin_size / 2;
    }
    else if(pos1 == (max_pos-1)) {
        pos2 = max_pos-2;
        c2 = pos2 * bin_size + bin_size / 2;
    } else
    {
        if(val > c1)
            pos2 = pos1 + 1;
        else
            pos2 = pos1 - 1;

        c2 = pos2 * bin_size + bin_size / 2;
    }

    w1 = (val - c1) / (c2 - c1);
    w2 = (c2 - val) / (c2 - c1);
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::specifyHistograms (const std::vector<size_t> &src_hist, const std::vector<size_t> &dst_hist, std::vector<size_t> & lut)
{
    if(src_hist.size() != dst_hist.size())
        throw std::runtime_error ("Histograms do not have the same size!");

    // normalize histograms
    size_t sum_src = 0, sum_dst = 0;
#pragma omp parallel for reduction(+:sum_src)
    for(size_t i=0; i<src_hist.size(); i++)
        sum_src += src_hist[i];

#pragma omp parallel for reduction(+:sum_dst)
    for(size_t i=0; i<src_hist.size(); i++)
        sum_dst += dst_hist[i];


    std::vector<float> src_hist_normalized (src_hist.size());
    std::vector<float> dst_hist_normalized (dst_hist.size());

#pragma omp parallel for
    for(size_t i=0; i<src_hist.size(); i++) {
        src_hist_normalized[i] = static_cast<float>(src_hist[i]) / sum_src;
        dst_hist_normalized[i] = static_cast<float>(dst_hist[i]) / sum_dst;
    }


    // compute cumulative histogram
    std::vector<float> src_hist_cumulative (src_hist.size(), 0.f);
    std::vector<float> dst_hist_cumulative (dst_hist.size(), 0.f);

    if ( !src_hist.empty() ) {
        src_hist_cumulative[0] = src_hist_normalized[0];
        dst_hist_cumulative[0] = dst_hist_normalized[0];
    }

    for(size_t i=1; i<src_hist.size(); i++) {
        src_hist_cumulative[i] = src_hist_cumulative[i-1] + src_hist_normalized[i];
        dst_hist_cumulative[i] = dst_hist_cumulative[i-1] + dst_hist_normalized[i];
    }


    lut.clear();
    lut.resize(src_hist.size(), 0);

    int last = 0;
    for (size_t k = 0; k < dst_hist_cumulative.size(); k++)
    {
        for (size_t z = last; z < src_hist_cumulative.size(); z++)
        {
            if ( (src_hist_cumulative[z] - dst_hist_cumulative[k]) >= 0)
            {
                if (z > 0 && (dst_hist_cumulative[k] - src_hist_cumulative[z-1]) < (src_hist_cumulative[z] - dst_hist_cumulative[k]))
                    z--;

                lut[k] = z;
                last = z;
                break;
            }
        }
    }

    int min = 0;
    for (size_t k = 0; k < src_hist_cumulative.size(); k++)
    {
        if (lut[k] != 0)
        {
            min = lut[k];
            break;
        }
    }

    for (size_t k = 0; k < src_hist_cumulative.size(); k++)
    {
        if ( lut[k] == 0)
            lut[k] = min;
        else
            break;
    }

    //max mapping extension
    int max = 0;
    for (int k = src_hist_cumulative.size() - 1; k >= 0; k--)
    {
        if (lut[k] != 0)
        {
            max = lut[k];
            break;
        }
    }

    for (int k = src_hist_cumulative.size() - 1; k >= 0; k--)
    {
        if ( lut[k] == 0)
            lut[k] = max;
        else
            break;
    }
}

template<typename ModelT, typename SceneT>
bool
GHV<ModelT, SceneT>::removeNanNormals (HVRecognitionModel<ModelT> &rm)
{
    if(!rm.visible_cloud_normals_) {
        rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal>);
        computeNormals<ModelT>(rm.visible_cloud_, rm.visible_cloud_normals_, param_.normal_method_);
    }

    //check nans...
    size_t kept = 0;
    for (size_t idx = 0; idx < rm.visible_cloud_->points.size (); idx++)
    {
        if ( pcl::isFinite(rm.visible_cloud_->points[idx]) && pcl::isFinite(rm.visible_cloud_normals_->points[idx]) )
        {
            rm.visible_cloud_->points[kept] = rm.visible_cloud_->points[idx];
            rm.visible_cloud_normals_->points[kept] = rm.visible_cloud_normals_->points[idx];
            kept++;
        }
    }

    rm.visible_cloud_->points.resize (kept);
    rm.visible_cloud_normals_->points.resize (kept);
    rm.visible_cloud_->width = rm.visible_cloud_normals_->width = kept;
    rm.visible_cloud_->height = rm.visible_cloud_normals_->height = 1;

    return !rm.visible_cloud_->points.empty();
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::registerModelAndSceneColor(std::vector<size_t> & lookup, HVRecognitionModel<ModelT> & rm)
{
    std::vector< std::vector<int> > label_indices;
    std::vector< std::vector<int> > explained_scene_pts_per_label;

    if(rm.smooth_faces_)
    {
        //use visible indices to check which points are visible
        rm.visible_labels_.reset(new pcl::PointCloud<pcl::PointXYZL>);
        pcl::copyPointCloud(*rm.smooth_faces_, rm.visible_indices_, *rm.visible_labels_);

        //specify using the smooth faces
        int max_label = 0;
        for(size_t k=0; k < rm.visible_labels_->points.size(); k++)
        {
            if( rm.visible_labels_->points[k].label > max_label)
                max_label = rm.visible_labels_->points[k].label;
        }

        //1) group points based on label
        label_indices.resize(max_label + 1);
        for(size_t k=0; k < rm.visible_labels_->points.size(); k++)
            label_indices[ rm.visible_labels_->points[k].label ].push_back(k);


        //2) for each group, find corresponding scene points and push them into label_explained_indices_points
        std::vector<std::pair<int, float> > label_index_distances;
        label_index_distances.resize(scene_cloud_downsampled_->points.size(), std::make_pair(-1, std::numeric_limits<float>::infinity()));

        for(size_t j=0; j < label_indices.size(); j++)
        {
            for (size_t i = 0; i < label_indices[j].size (); i++)
            {
                std::vector<int> & nn_indices = rm.scene_inlier_indices_for_visible_pt_[label_indices[j][i]];
                std::vector<float> & nn_distances = rm.scene_inlier_distances_for_visible_pt_[label_indices[j][i]];

                for (size_t k = 0; k < nn_indices.size (); k++)
                {
                    if(label_index_distances[nn_indices[k]].first != static_cast<int>(j)) // notalready explained by the same label
                    {
                        //if different labels, then take the new label if distances is smaller
                        if(nn_distances[k] < label_index_distances[nn_indices[k]].second)
                        {
                            label_index_distances[nn_indices[k]].first = static_cast<int>(j);
                            label_index_distances[nn_indices[k]].second = nn_distances[k];
                        } //otherwise, ignore new label since the older one is closer
                    }
                }
            }
        }

        //3) set label_explained_indices_points
        explained_scene_pts_per_label.resize(max_label + 1);
        for (size_t i = 0; i < scene_cloud_downsampled_->points.size (); i++)
        {
            if(label_index_distances[i].first < 0)
                continue;

            explained_scene_pts_per_label[label_index_distances[i].first].push_back(i);
        }
    }
    else
    {
        label_indices.resize(1);
        label_indices[0].resize(rm.visible_cloud_->points.size());
        for(size_t k=0; k < rm.visible_cloud_->points.size(); k++)
            label_indices[0][k] = k;

        std::vector<bool> scene_pt_is_explained ( scene_cloud_downsampled_->points.size(), false );
        for (size_t i = 0; i < label_indices[0].size (); i++)
        {
            std::vector<int> & nn_indices = rm.scene_inlier_indices_for_visible_pt_[ label_indices[0][i] ];
            //            std::vector<float> & nn_distances = recog_model.inlier_distances_[ label_indices[0][i] ];

            for (size_t k = 0; k < nn_indices.size (); k++)
                scene_pt_is_explained[ nn_indices[k] ] = true;
        }

        std::vector<int> explained_scene_pts = createIndicesFromMask<int>(scene_pt_is_explained);

        explained_scene_pts_per_label.resize(1, std::vector<int>( explained_scene_pts.size() ) );
        for (size_t i = 0; i < explained_scene_pts.size (); i++)
            explained_scene_pts_per_label[0][i] = explained_scene_pts[i];
    }

    for(size_t j=0; j < label_indices.size(); j++)
    {
        std::vector<int> explained_scene_pts = explained_scene_pts_per_label[j];

        if(param_.color_space_ == 0 || param_.color_space_ == 3)
        {
            std::vector<float> model_L_values ( label_indices[j].size () );
            std::vector<float> scene_L_values ( explained_scene_pts.size() );

            float range_min = -1.0f, range_max = 1.0f;

            for (size_t i = 0; i < label_indices[j].size (); i++)
                model_L_values[i] = rm.pt_color_( label_indices[j][i], 0);

            for(size_t i=0; i<explained_scene_pts.size(); i++)
                scene_L_values[i] = scene_color_channels_(explained_scene_pts[i], 0);

            size_t hist_size = 100;
            std::vector<size_t> model_L_hist, scene_L_hist;
            computeHistogram(model_L_values, model_L_hist, hist_size, range_min, range_max);
            computeHistogram(scene_L_values, scene_L_hist, hist_size, range_min, range_max);
            specifyHistograms(scene_L_hist, model_L_hist, lookup);

            for (size_t i = 0; i < label_indices[j].size(); i++)
            {
                float LRefm = rm.pt_color_( label_indices[j][i], 0);
                int pos = std::floor (hist_size * (LRefm - range_min) / (range_max - range_min));

                if(pos > lookup.size())
                    throw std::runtime_error("Color not specified");

                LRefm = lookup[pos] * (range_max - range_min)/hist_size + range_min;
                rm.pt_color_( label_indices[j][i], 0) = LRefm;
                rm.cloud_indices_specified_.push_back(label_indices[j][i]);
            }
        }
        else
            throw std::runtime_error("Specified color space not implemented at the moment!");
        //        else if(param_.color_space_ == 1 || param_.color_space_ == 5)
        //        {

        //            std::vector<Eigen::Vector3f> model_gs_values, scene_gs_values;

        //            //compute RGB histogram for the model points
        //            for (size_t i = 0; i < label_indices[j].size (); i++)
        //            {
        //                model_gs_values.push_back(recog_model.cloud_RGB_[label_indices[j][i]] * 255.f);
        //            }

        //            //compute RGB histogram for the explained points
        //            std::set<int>::iterator it;
        //            for(it=explained_scene_pts.begin(); it != explained_scene_pts.end(); it++)
        //            {
        //                scene_gs_values.push_back(scene_RGB_values_[*it] * 255.f);
        //            }

        //            int dim = 3;
        //            Eigen::MatrixXf gs_model, gs_scene;
        //            computeRGBHistograms(model_gs_values, gs_model, dim);
        //            computeRGBHistograms(scene_gs_values, gs_scene, dim);

        //            //histogram specification, adapt model values to scene values
        //            specifyRGBHistograms(gs_scene, gs_model, lookup, dim);

        //            for (size_t ii = 0; ii < label_indices[j].size (); ii++)
        //            {
        //                for(int k=0; k < dim; k++)
        //                {
        //                    float color = recog_model.cloud_RGB_[label_indices[j][ii]][k] * 255.f;
        //                    int pos = std::floor (static_cast<float> (color) / 255.f * 256);
        //                    float specified = lookup(pos, k);
        //                    recog_model.cloud_RGB_[label_indices[j][ii]][k] = specified / 255.f;
        //                }
        //            }

        //            if(param_.color_space_ == 5)
        //            {
        //                //transform specified RGB to lab
        //                for(size_t jj=0; jj < recog_model.cloud_LAB_.size(); jj++)
        //                {
        //                    unsigned char rm = recog_model.cloud_RGB_[jj][0] * 255;
        //                    unsigned char gm = recog_model.cloud_RGB_[jj][1] * 255;
        //                    unsigned char bm = recog_model.cloud_RGB_[jj][2] * 255;

        //                    float LRefm, aRefm, bRefm;
        //                    color_transf_omp_.RGB2CIELAB_normalized (rm, gm, bm, LRefm, aRefm, bRefm);
        //                    recog_model.cloud_LAB_[jj] = Eigen::Vector3f(LRefm, aRefm, bRefm);
        //                }
        //            }
        //        }
        //        else if(param_.color_space_ == 2) //gray scale
        //        {
        //            std::vector<float> model_gs_values, scene_gs_values;

        //            //compute RGB histogram for the model points
        //            for (size_t ii = 0; ii < label_indices[j].size (); ii++)
        //            {
        //                model_gs_values.push_back(recog_model.cloud_GS_[label_indices[j][ii]] * 255.f);
        //            }

        //            //compute RGB histogram for the explained points
        //            std::set<int>::iterator it;
        //            for(it=explained_scene_pts.begin(); it != explained_scene_pts.end(); it++)
        //                scene_gs_values.push_back(scene_GS_values_[*it] * 255.f);

        //            Eigen::MatrixXf gs_model, gs_scene;
        //            computeHistogram(model_gs_values, gs_model);
        //            computeHistogram(scene_gs_values, gs_scene);

        //            //histogram specification, adapt model values to scene values
        //            specifyRGBHistograms(gs_scene, gs_model, lookup, 1);

        //            for (size_t ii = 0; ii < label_indices[j].size (); ii++)
        //            {
        //                float LRefm = recog_model.cloud_GS_[label_indices[j][ii]] * 255.f;
        //                int pos = std::floor (static_cast<float> (LRefm) / 255.f * 256);
        //                float gs_specified = lookup(pos, 0);
        //                LRefm = gs_specified / 255.f;
        //                recog_model.cloud_GS_[label_indices[j][ii]] = LRefm;
        //            }
        //        }
        //        else if(param_.color_space_ == 6)
        //        {
        //            std::vector<Eigen::Vector3f> model_gs_values, scene_gs_values;
        //            int dim = 3;

        //            //compute LAB histogram for the model points
        //            for (size_t ii = 0; ii < label_indices[j].size (); ii++)
        //            {
        //                Eigen::Vector3f lab = recog_model.cloud_LAB_[label_indices[j][ii]] * 255.f;
        //                lab[1] = (lab[1] + 255.f) / 2.f;
        //                lab[2] = (lab[2] + 255.f) / 2.f;

        //                for(int k=0; k < dim; k++)
        //                {
        //                    if(!(lab[k] >= 0 && lab[k] <= 255.f))
        //                    {
        //                        std::cout << lab[k] << " dim:" << dim << std::endl;
        //                        assert(lab[k] >= 0 && lab[k] <= 255.f);
        //                    }
        //                }
        //                model_gs_values.push_back(lab);
        //            }

        //            //compute LAB histogram for the explained points
        //            for(size_t p_id=0; p_id < explained_scene_pts.size(); p_id++)
        //            {
        //                Eigen::Vector3f lab = scene_LAB_values_[ explained_scene_pts[p_id] ] * 255.f;
        //                lab[1] = (lab[1] + 255.f) / 2.f;
        //                lab[2] = (lab[2] + 255.f) / 2.f;

        //                for(size_t k=0; k < dim; k++)
        //                {
        //                    assert(lab[k] >= 0 && lab[k] <= 255.f);
        //                }
        //                scene_gs_values.push_back(lab);
        //            }

        //            Eigen::MatrixXf gs_model, gs_scene;
        //            computeRGBHistograms(model_gs_values, gs_model, dim);
        //            computeRGBHistograms(scene_gs_values, gs_scene, dim);

        //            //histogram specification, adapt model values to scene values
        //            specifyRGBHistograms(gs_scene, gs_model, lookup, dim);

        //            for (size_t ii = 0; ii < label_indices[j].size (); ii++)
        //            {
        //                recog_model.cloud_indices_specified_.push_back(label_indices[j][ii]);

        //                for(int k=0; k < dim; k++)
        //                {
        //                    float LRefm = recog_model.cloud_LAB_[label_indices[j][ii]][k] * 255.f;
        //                    if(k > 0)
        //                        LRefm = (LRefm + 255.f) / 2.f;

        //                    int pos = std::floor (static_cast<float> (LRefm) / 255.f * 256);
        //                    assert(pos < lookup.rows());
        //                    float gs_specified = lookup(pos, k);

        //                    float diff = std::abs(LRefm - gs_specified);
        //                    assert(gs_specified >= 0 && gs_specified <= 255.f);
        //                    LRefm = gs_specified / 255.f;
        //                    assert(LRefm >= 0 && LRefm <= 1.f);
        //                    if(k > 0)
        //                    {
        //                        LRefm *= 2.f;
        //                        LRefm -= 1.f;

        //                        if(!(LRefm >= -1.f && LRefm <= 1.f))
        //                        {
        //                            std::cout << LRefm << " dim:" << k << " diff:" << diff << std::endl;
        //                            assert(LRefm >= -1.f && LRefm <= 1.f);
        //                        }
        //                    }
        //                    else
        //                    {
        //                        if(!(LRefm >= 0.f && LRefm <= 1.f))
        //                        {
        //                            std::cout << LRefm << " dim:" << k << std::endl;
        //                            assert(LRefm >= 0.f && LRefm <= 1.f);
        //                        }
        //                    }

        //                    recog_model.cloud_LAB_[label_indices[j][ii]][k] = LRefm;
        //                }
        //            }
        //        }
    }
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::computeSceneOccupancyGridByRM()
{
    ModelT min_pt_all, max_pt_all;
    min_pt_all.x = min_pt_all.y = min_pt_all.z = std::numeric_limits<float>::max ();
    max_pt_all.x = max_pt_all.y = max_pt_all.z = std::numeric_limits<float>::min ();

    for (size_t i = 0; i < recognition_models_.size (); i++)
    {
        HVRecognitionModel<ModelT> &rm = *recognition_models_[i];

        if(rm.is_planar_)
            continue;

        ModelT min_pt, max_pt;
        pcl::getMinMax3D (*rm.complete_cloud_, min_pt, max_pt);

        if (min_pt.x < min_pt_all.x)
            min_pt_all.x = min_pt.x;

        if (min_pt.y < min_pt_all.y)
            min_pt_all.y = min_pt.y;

        if (min_pt.z < min_pt_all.z)
            min_pt_all.z = min_pt.z;

        if (max_pt.x > max_pt_all.x)
            max_pt_all.x = max_pt.x;

        if (max_pt.y > max_pt_all.y)
            max_pt_all.y = max_pt.y;

        if (max_pt.z > max_pt_all.z)
            max_pt_all.z = max_pt.z;
    }

    size_t size_x = static_cast<size_t> ( (max_pt_all.x - min_pt_all.x) / param_.res_occupancy_grid_ + 1.5f);  // rounding up and add 1
    size_t size_y = static_cast<size_t> ( (max_pt_all.y - min_pt_all.y) / param_.res_occupancy_grid_ + 1.5f);
    size_t size_z = static_cast<size_t> ( (max_pt_all.z - min_pt_all.z) / param_.res_occupancy_grid_ + 1.5f);
    complete_cloud_occupancy_by_RM_.resize (size_x * size_y * size_z, 0);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < recognition_models_.size (); i++)
    {
        HVRecognitionModel<ModelT> &rm = *recognition_models_[i];

        if(rm.is_planar_)
            continue;

        std::vector<bool> voxel_is_occupied(size_x * size_y * size_z, false);

        rm.complete_cloud_occupancy_indices_.resize( rm.complete_cloud_->points.size ());

        size_t kept = 0;
        for (size_t j = 0; j < rm.complete_cloud_->points.size (); j++)
        {
            size_t pos_x, pos_y, pos_z;
            pos_x = static_cast<size_t>( (rm.complete_cloud_->points[j].x - min_pt_all.x) / param_.res_occupancy_grid_);
            pos_y = static_cast<size_t>( (rm.complete_cloud_->points[j].y - min_pt_all.y) / param_.res_occupancy_grid_);
            pos_z = static_cast<size_t>( (rm.complete_cloud_->points[j].z - min_pt_all.z) / param_.res_occupancy_grid_);

            size_t idx = pos_z * size_x * size_y + pos_y * size_x + pos_x;

            if ( !voxel_is_occupied[idx] )
            {
                rm.complete_cloud_occupancy_indices_[kept] = idx;
                voxel_is_occupied[idx] = true;
                kept++;
            }
        }
        rm.complete_cloud_occupancy_indices_.resize(kept);
    }
}

template<typename ModelT, typename SceneT>
bool
GHV<ModelT, SceneT>::addModel (HVRecognitionModel<ModelT> &rm)
{
    if( requires_normals_ )
        removeNanNormals(rm);

    std::vector<std::vector<std::pair<size_t, float> > > scene_pt_is_explained_by_model_pt (scene_cloud_downsampled_->points.size()); // stores information about which scene point (outer loop) is explained by which model pt (inner loop)
    std::vector<std::vector<std::pair<size_t, float> > > scene_pt_is_explained_by_model_pt_with_color (scene_cloud_downsampled_->points.size());

    rm.outliers_weight_.resize (rm.visible_cloud_->points.size ());
    rm.outlier_indices_.resize (rm.visible_cloud_->points.size ());
    rm.outlier_indices_3d_.resize (rm.visible_cloud_->points.size ());
    rm.outlier_indices_color_.resize (rm.visible_cloud_->points.size ());
    rm.scene_inlier_indices_for_visible_pt_.resize(rm.visible_cloud_->points.size ());
    rm.scene_inlier_distances_for_visible_pt_.resize(rm.visible_cloud_->points.size ());

    if(!rm.is_planar_ && !param_.ignore_color_even_if_exists_)
    {
        //compute cloud LAB values for model visible points
        size_t num_color_channels = 0;
        switch (param_.color_space_)
        {
        case ColorSpace::LAB: case ColorSpace::RGB: num_color_channels = 3; break;
        case ColorSpace::GRAYSCALE: num_color_channels = 1; break;
        default: throw std::runtime_error("Color space not implemented!");
        }

        rm.pt_color_ = Eigen::MatrixXf::Zero ( rm.visible_cloud_->points.size(), num_color_channels);

        #pragma omp parallel for schedule (dynamic)
        for(size_t j=0; j < rm.visible_cloud_->points.size(); j++)
        {
            bool exists_m;
            float rgb_m = 0.f;
            pcl::for_each_type<FieldListM> ( pcl::CopyIfFieldExists<typename CloudM::PointType, float> (
                                                 rm.visible_cloud_->points[j], "rgb", exists_m, rgb_m));

            if(!exists_m)
                throw std::runtime_error("Color verification was requested but point cloud does not have color information!");

            uint32_t rgb = *reinterpret_cast<int*> (&rgb_m);
            unsigned char rmc = (rgb >> 16) & 0x0000ff;
            unsigned char gmc = (rgb >> 8) & 0x0000ff;
            unsigned char bmc = (rgb) & 0x0000ff;
            float rmf = static_cast<float>(rmc) / 255.f;
            float gmf = static_cast<float>(gmc) / 255.f;
            float bmf = static_cast<float>(bmc) / 255.f;

            switch (param_.color_space_)
            {
            case ColorSpace::LAB:
                float LRefm, aRefm, bRefm;
                color_transf_omp_.RGB2CIELAB_normalized (rmc, gmc, bmc, LRefm, aRefm, bRefm);
                rm.pt_color_(j, 0) = LRefm;
                rm.pt_color_(j, 1) = aRefm;
                rm.pt_color_(j, 2) = bRefm;
                break;
            case ColorSpace::RGB:
                rm.pt_color_(j, 0) = rmf;
                rm.pt_color_(j, 1) = gmf;
                rm.pt_color_(j, 2) = bmf;
            case ColorSpace::GRAYSCALE:
                rm.pt_color_(j, 0) = .2126 * rmf + .7152 * gmf + .0722 * bmf;
            }
        }

        if(param_.use_histogram_specification_) {
            std::vector<size_t> lookup;
            registerModelAndSceneColor(lookup, rm);
        }
    }


    //Goes through the visible model points and finds scene points within a radius neighborhood
    //If in this neighborhood, there are no scene points, model point is considered outlier
    //If there are scene points, the model point is associated with the scene point, together with its distance
    //A scene point might end up being explained by multiple model points
    #pragma omp parallel for schedule(dynamic)
    for (size_t pt = 0; pt < rm.visible_cloud_->points.size (); pt++)
        octree_scene_downsampled_->radiusSearch (rm.visible_cloud_->points[pt], param_.inliers_threshold_,
                                                 rm.scene_inlier_indices_for_visible_pt_[pt], rm.scene_inlier_distances_for_visible_pt_[pt],
                                                 std::numeric_limits<int>::max ());

    float inliers_gaussian = 2 * param_.inliers_threshold_ * param_.inliers_threshold_;
    float inliers_gaussian_soft = 2 * (param_.inliers_threshold_ + param_.resolution_) * (param_.inliers_threshold_ + param_.resolution_);
    size_t outliers = 0, outliers_3d = 0, outliers_color = 0;
    float sigma = 2.f * param_.color_sigma_ab_ * param_.color_sigma_ab_;
    float sigma_y = 2.f * param_.color_sigma_l_ * param_.color_sigma_l_;

    for (size_t m_pt_id = 0; m_pt_id < rm.visible_cloud_->points.size (); m_pt_id++)
    {
        bool outlier = false;
        int outlier_type = OutlierType::DIST;

        const std::vector<int> & nn_indices = rm.scene_inlier_indices_for_visible_pt_[m_pt_id];
        const std::vector<float> & nn_distances = rm.scene_inlier_distances_for_visible_pt_[m_pt_id];

        if( nn_indices.empty() ) // if there is no scene point nearby, count it as an outlier
        {
            rm.outlier_indices_3d_[outliers_3d] = m_pt_id;
            outlier = true;
            outlier_type = OutlierType::DIST;
            outliers_3d++;
        }
        else    // check if it is an outlier due to color mismatch
        {
            bool is_color_outlier = true;
            std::vector<float> weights;
            if (!rm.is_planar_)
            {
                weights.resize( nn_distances.size () );
                float color_weight = 1.f;

                for (size_t k = 0; k < nn_distances.size(); k++)
                {
                    if (!param_.ignore_color_even_if_exists_ )
                    {
                        if(param_.color_space_ == ColorSpace::LAB || param_.color_space_ == 5 || param_.color_space_ == 6)
                        {
                            const Eigen::Vector3f &color_m = rm.pt_color_.row(m_pt_id);
                            const Eigen::Vector3f &color_s = scene_color_channels_.row( nn_indices[k] );

                            color_weight = std::exp ( -0.5f * ( (color_m[0] - color_s[0]) * (color_m[0] - color_s[0]) / sigma_y
                                    +( (color_m[1] - color_s[1]) * (color_m[1] - color_s[1]) + (color_m[2] - color_s[2]) * (color_m[2] - color_s[2]) ) / sigma ) );
                        }
                        else if(param_.color_space_ == ColorSpace::RGB)
                        {
                            const Eigen::Vector3f &color_m = rm.pt_color_.row(m_pt_id);
                            const Eigen::Vector3f &color_s = scene_color_channels_.row( nn_indices[k] );

                            color_weight = std::exp (-0.5f * (   (color_m[0] - color_s[0]) * (color_m[0] - color_s[0])
                                    + (color_m[1] - color_s[1]) * (color_m[1] - color_s[1]) + (color_m[2] - color_s[2]) * (color_m[2] - color_s[2])) / sigma);
                        }
                        else if(param_.color_space_ == ColorSpace::GRAYSCALE)
                        {
                            float yuvm = rm.pt_color_( m_pt_id, 0 );
                            float yuvs = scene_color_channels_( nn_indices[k], 0 );

                            color_weight = std::exp (-0.5f * (yuvm - yuvs) * (yuvm - yuvs) / sigma_y);
                        }
                        else if(param_.color_space_ == 3)
                        {
                            float yuvm = rm.pt_color_(m_pt_id,0);
                            float yuvs = scene_color_channels_( nn_indices[k], 0 );

                            color_weight = std::exp (-0.5f * (yuvm - yuvs) * (yuvm - yuvs) / sigma_y);
                        }
                    }

                    const float dist_weight = std::exp( -nn_distances[k] / inliers_gaussian_soft );
                    weights[k] = color_weight * dist_weight;

                    if (weights[k] > param_.best_color_weight_)
                    {
                        scene_pt_is_explained_by_model_pt_with_color[ nn_indices[k] ].push_back( std::pair<size_t, float>( m_pt_id, weights[k] ));
                        is_color_outlier = false;
                    }
                }

                if(is_color_outlier)
                {
                    rm.outlier_indices_color_[outliers_color] = m_pt_id;
                    outlier = true;
                    outliers_color++;
                    outlier_type = OutlierType::COLOR;
                }
            }

            // if it is not an outlier, then mark corresponding scene point(s) as explained
            for (size_t k = 0; k < nn_distances.size (); k++)
            {
                int scene_pt_id = nn_indices[k];
                float weight = nn_distances[k];

                if ( k >= weights.size() || weights[k] > param_.best_color_weight_ ) // if color check is enabled, only include if it passes the check
                    scene_pt_is_explained_by_model_pt[ scene_pt_id ].push_back( std::pair<size_t, float>( m_pt_id, weight) );
            }
        }

        if(outlier)
        {
            //weight outliers based on noise model
            //model points close to occlusion edges or with perpendicular normals
            float d_weight = 1.f;
            //std::cout << "is an outlier" << is_planar_model << " " << occ_edges_available_ << std::endl;

            if(!rm.is_planar_ && requires_normals_ && false)
            {
                //std::cout << "going to weight based on normals..." << std::endl;
                Eigen::Vector3f normal_p = rm.visible_cloud_normals_->points[m_pt_id].getNormalVector3fMap();
                Eigen::Vector3f normal_vp = Eigen::Vector3f::UnitZ() * -1.f;
                normal_p.normalize ();
                normal_vp.normalize ();

                float dot = normal_vp.dot(normal_p);
                float angle = pcl::rad2deg(acos(dot));
                if (angle > 60.f)
                {
                    if(outlier_type == OutlierType::COLOR)
                        outliers_color--;
                    else
                        outliers_3d--;

                    // [60,75) => 0.5
                    // [75,90) => 0.25
                    // >90 => 0

                    /*if(angle >= 90.f)
                        d_weight = 0.25f;
                    else*/
                    d_weight = param_.d_weight_for_bad_normals_;
                }
            }

            rm.outliers_weight_[outliers] = param_.regularizer_ * d_weight;
            rm.outlier_indices_[outliers] = m_pt_id;
            outliers++;
        }
    }

    rm.outliers_weight_.resize (outliers);
    rm.outlier_indices_.resize (outliers);
    rm.outlier_indices_3d_.resize (outliers_3d);
    rm.outlier_indices_color_.resize (outliers_color);

    std::vector<int> explained_indices ( scene_cloud_downsampled_->points.size() );
    std::vector<float> explained_indices_distances ( scene_cloud_downsampled_->points.size() );

    rm.scene_pt_is_explained_.clear();
    rm.scene_pt_is_explained_.resize(scene_cloud_downsampled_->points.size(), false);
    size_t kept=0;
    for(size_t s_pt_id=0; s_pt_id<scene_cloud_downsampled_->points.size(); s_pt_id++)
    {
        if ( scene_pt_is_explained_by_model_pt[s_pt_id].empty() )
            continue;

        Eigen::Vector3f scene_p_normal = scene_normals_->points[s_pt_id].getNormalVector3fMap ();
        scene_p_normal.normalize();

        size_t closest = 0;
        float min_d = std::numeric_limits<float>::max();
        for (size_t i = 0; i < scene_pt_is_explained_by_model_pt[ s_pt_id ].size(); i++)
        {
            std::pair<size_t, float> m_pt_and_weight_p = scene_pt_is_explained_by_model_pt[ s_pt_id ][i];
            if ( m_pt_and_weight_p.second < min_d )
            {
                min_d = m_pt_and_weight_p.second;
                closest = i;
            }
        }

        float d_weight = std::exp( -( scene_pt_is_explained_by_model_pt[ s_pt_id ][closest].second / inliers_gaussian));

        //using normals to weight inliers
        size_t m_pt = scene_pt_is_explained_by_model_pt[ s_pt_id ][closest].first;
        Eigen::Vector3f model_p_normal = rm.visible_cloud_normals_->points[ m_pt ].getNormalVector3fMap ();
        model_p_normal.normalize();

        bool use_dot = false;
        float dotp = scene_p_normal.dot (model_p_normal); //[-1,1] from antiparallel trough perpendicular to parallel

        if(use_dot)
        {
            if (dotp < 0.f)
                dotp = 0.f;
        }
        else
        {
            if(dotp < -1.f) dotp = -1.f;
            if(dotp > 1.f) dotp = 1.f;

            float angle = pcl::rad2deg(acos(dotp));

            if(angle > 90.f) //ATTENTION!
                dotp = 0;
            else
                dotp = (1.f - angle / 90.f);
        }


        rm.scene_pt_is_explained_[ s_pt_id ] = true; //this scene point is explained by this hypothesis

//        #pragma omp critical
        {
        explained_indices[kept] = s_pt_id;
        explained_indices_distances [kept] = d_weight * dotp * rm.extra_weight_;
        kept++;
        }
    }
    explained_indices.resize(kept);
    explained_indices_distances.resize(kept);

    //compute the amount of information for explained scene points (color)
//    float mean_distance = 0.f;
//    if( !explained_indices.empty())
//        mean_distance = std::accumulate(explained_indices_distances.begin(), explained_indices_distances.end(), 0.f) / static_cast<float>(explained_indices_distances.size());

//    rm.mean_ = mean_distance;

    //modify the explained weights for planar models if color is being used
    if(rm.is_planar_)
    {
        for(size_t k=0; k < explained_indices_distances.size(); k++)
        {
            explained_indices_distances[k] *= param_.best_color_weight_;

            if (!param_.ignore_color_even_if_exists_)
                explained_indices_distances[k] /= 2;
        }
    }

    rm.hyp_penalty_ = 0; //ATTENTION!
    rm.explained_scene_indices_ = explained_indices;
    rm.distances_to_explained_scene_indices_ = explained_indices_distances;

    return true;
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::computeClutterCueAtOnce ()
{
    //create mask which says which scene points are explained by any of the recognition models
    std::vector<bool> scene_pt_is_explained (scene_cloud_downsampled_->size(), false);

    for (size_t i = 0; i < recognition_models_.size (); i++)
    {
        const HVRecognitionModel<ModelT> &rm = *recognition_models_[i];

        for (size_t jj = 0; jj < rm.explained_scene_indices_.size (); jj++)
            scene_pt_is_explained [ rm.explained_scene_indices_[jj] ] = true;
    }

    std::vector<int> explained_points_vec = createIndicesFromMask<int>(scene_pt_is_explained);

    std::vector<int> scene_to_unique( scene_cloud_downsampled_->size(), -1 );
    for(size_t i=0; i < explained_points_vec.size(); i++)
        scene_to_unique[ explained_points_vec[i] ] = i;


    //find neighbours within clutter radius for all explained points
    std::vector<std::vector<int> > nn_indices_all_points(explained_points_vec.size());
    std::vector<std::vector<float> > nn_distances_all_points(explained_points_vec.size());

    #pragma omp parallel for schedule(dynamic)
    for(size_t k=0; k < explained_points_vec.size(); k++)
    {
        octree_scene_downsampled_->radiusSearch (scene_cloud_downsampled_->points[explained_points_vec[k]],
                param_.radius_neighborhood_clutter_, nn_indices_all_points[k],
                nn_distances_all_points[k], std::numeric_limits<int>::max ());
    }

    //    const float min_clutter_dist = std::pow(param_.inliers_threshold_ * 0.f, 2.f); // ??? why *0.0f? --> always 0 then

//    #pragma omp parallel for schedule(dynamic)
    for (size_t rm_id = 0; rm_id < recognition_models_.size (); rm_id++)
    {
        HVRecognitionModel<ModelT> &rm = *recognition_models_[rm_id];

        std::pair<int, float> def_value = std::make_pair(-1, std::numeric_limits<float>::max());
        std::vector< std::pair<int, float> > unexplained_points_per_model (scene_cloud_downsampled_->points.size(), def_value);

        for (size_t i = 0; i < rm.explained_scene_indices_.size (); i++)
        {
            int sidx = rm.explained_scene_indices_[i];
            int idx_to_unique = scene_to_unique[sidx];

            for (size_t k = 0; k < nn_indices_all_points[idx_to_unique].size (); k++)
            {
                int sidx_nn = nn_indices_all_points[idx_to_unique][k]; //in the neighborhood of an explained point (idx_to_ep)
                if(rm.scene_pt_is_explained_[sidx_nn])
                    continue;

                float d = ( scene_cloud_downsampled_->points[sidx].getVector3fMap()
                            - scene_cloud_downsampled_->points[sidx_nn].getVector3fMap() ).squaredNorm ();

                if( d < unexplained_points_per_model[sidx_nn].second ) //there is an explained point which is closer to this unexplained point
                {
                    unexplained_points_per_model[sidx_nn].first = sidx;
                    unexplained_points_per_model[sidx_nn].second = d;
                }
            }
        }

        rm.unexplained_in_neighborhood.resize (scene_cloud_downsampled_->points.size ());
        rm.unexplained_in_neighborhood_weights.resize (scene_cloud_downsampled_->points.size ());

        float sigma_clutter = 2 * param_.radius_neighborhood_clutter_ * param_.radius_neighborhood_clutter_;
        size_t kept=0;
        for(size_t i=0; i < unexplained_points_per_model.size(); i++)
        {
            int sidx = unexplained_points_per_model[i].first;
            if(sidx < 0)
                continue;

            //sidx is the closest explained point to the unexplained point

            if (!rm.scene_pt_is_explained_[sidx] || rm.scene_pt_is_explained_[i])
                throw std::runtime_error ("Something is wrong!");

            //point i is unexplained and in the neighborhood of sidx (explained point)
            rm.unexplained_in_neighborhood[kept] = i;

            float d = unexplained_points_per_model[i].second;
            float d_weight;
            if( param_.use_clutter_exp_ )
                d_weight = std::exp( -(d / sigma_clutter));
            else
                d_weight = 1 - d / (param_.radius_neighborhood_clutter_ * param_.radius_neighborhood_clutter_); //points that are close have a strong weight

            //using normals to weight clutter points
            const Eigen::Vector3f & scene_p_normal = scene_normals_->points[sidx].getNormalVector3fMap ();
            const Eigen::Vector3f & model_p_normal = scene_normals_->points[i].getNormalVector3fMap ();
            float dotp = scene_p_normal.dot (model_p_normal); //[-1,1] from antiparallel trough perpendicular to parallel

            if (dotp < 0)
                dotp = 0.f;

            float w = d_weight * dotp;

            float curvature = scene_normals_->points[ sidx ].curvature;

            if ( (clusters_cloud_->points[i].label != 0 || param_.use_super_voxels_) &&
                 (clusters_cloud_->points[i].label == clusters_cloud_->points[sidx].label) && !rm.is_planar_ && curvature < 0.015)
            {
                w = 1.f; //ATTENTION!
                rm.unexplained_in_neighborhood_weights[kept] = param_.clutter_regularizer_ * w;
            }
            else
                rm.unexplained_in_neighborhood_weights[kept] = w;

            kept++;
        }

        rm.unexplained_in_neighborhood_weights.resize (kept);
        rm.unexplained_in_neighborhood.resize (kept);
    }
}



//######### VISUALIZATION FUNCTIONS #####################

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::getOutliersForAcceptedModels(std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud) const
{
    for(size_t i=0; i < recognition_models_.size(); i++)
    {
        if(mask_[i])
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_points(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*(recognition_models_[i]->visible_cloud_), recognition_models_[i]->outlier_indices_, *outlier_points);
            outliers_cloud.push_back(outlier_points);
        }
    }
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::getOutliersForAcceptedModels(std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud_color,
                                                  std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud_3d) const
{
    for(size_t i=0; i < recognition_models_.size(); i++)
    {
        const HVRecognitionModel<ModelT> &rm = *recognition_models_[i];
        if(mask_[i])
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_points (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*(rm.visible_cloud_), rm.outlier_indices_color_, *outlier_points);
            outliers_cloud_color.push_back(outlier_points);
            pcl::copyPointCloud(*rm.visible_cloud_, rm.outlier_indices_3d_, *outlier_points);
            outliers_cloud_3d.push_back(outlier_points);
        }
    }
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::visualizeGOCuesForModel(const HVRecognitionModel<ModelT> &rm) const
{

    if(!rm_vis_) {
        rm_vis_.reset (new pcl::visualization::PCLVisualizer ("model cues"));
        rm_vis_->createViewPort(0   , 0   , 0.33,0.5 , rm_v1);
        rm_vis_->createViewPort(0.33, 0   , 0.66,0.5 , rm_v2);
        rm_vis_->createViewPort(0.66, 0   , 1   ,1   , rm_v3);
        rm_vis_->createViewPort(0   , 0.5 , 0.33,1   , rm_v4);
        rm_vis_->createViewPort(0.33, 0.5 , 0.66,1   , rm_v5);
        rm_vis_->createViewPort(0.66, 0.5 , 1   ,1   , rm_v6);
    }

    rm_vis_->removeAllPointClouds();
    rm_vis_->removeAllShapes();

    float max_weight=std::numeric_limits<float>::min();
    float min_weight=std::numeric_limits<float>::max();

    for(size_t i=0; i<rm.unexplained_in_neighborhood_weights.size(); i++) {
        if(rm.unexplained_in_neighborhood_weights[i] < min_weight)
            min_weight = rm.unexplained_in_neighborhood_weights[i];

        if(rm.unexplained_in_neighborhood_weights[i] > max_weight)
            max_weight = rm.unexplained_in_neighborhood_weights[i];
    }

    rm_vis_->addText("scene",10,10,10,0.5,0.5,0.5,"scene",rm_v1);
    std::stringstream text_outliers; text_outliers << "model outliers 3d (blue) and color (green). Weight: " << rm.getOutliersWeight() << std::endl;
    rm_vis_->addText(text_outliers.str(),10,10,10,0.5,0.5,0.5,"model outliers",rm_v2);
    rm_vis_->addText("scene pts explained (blue)",10,10,10,0.5,0.5,0.5,"scene pt explained",rm_v3);
    rm_vis_->addText("unexplained in neighborhood",10,10,10,0.5,0.5,0.5,"unexplained in neighborhood",rm_v4);
    rm_vis_->addText("smooth segmentation",10,10,10,0.5,0.5,0.5,"smooth segmentation",rm_v5);
    rm_vis_->addText("segments for label=0",10,10,10,0.5,0.5,0.5,"label 0",rm_v6);

    // scene and model
    rm_vis_->addPointCloud(scene_cloud_downsampled_, "scene1",rm_v1);
    rm_vis_->addPointCloud(rm.visible_cloud_, "model",rm_v1);

    // model outliers
    rm_vis_->addPointCloud(rm.visible_cloud_, "model2",rm_v2);
    pcl::PointCloud<pcl::PointXYZRGB> o_3d_p, o_color_p, scene_pts_explained;
    pcl::copyPointCloud(*rm.visible_cloud_, rm.outlier_indices_3d_, o_3d_p);
    pcl::copyPointCloud(*rm.visible_cloud_, rm.outlier_indices_color_, o_color_p);
    for(pcl::PointXYZRGB &p:o_3d_p.points)   { p.r = p.g = 0; p.b = 255.f;  }
    for(pcl::PointXYZRGB &p:o_color_p.points)   { p.r = p.g = 255.f; p.b = 0.f;  }
    rm_vis_->addPointCloud(o_3d_p.makeShared(), "outliers_3d", rm_v2);
    rm_vis_->addPointCloud(o_color_p.makeShared(), "outliers_color", rm_v2);

    // explained points
    rm_vis_->addPointCloud(scene_cloud_downsampled_, "scene4", rm_v3);
    pcl::copyPointCloud(*scene_cloud_downsampled_, rm.explained_scene_indices_, scene_pts_explained);
    for(pcl::PointXYZRGB &p:scene_pts_explained.points)   { p.r = p.g = 0; p.b = 255.f;  }
    rm_vis_->addPointCloud(scene_pts_explained.makeShared(), "scene_pts_explained", rm_v3);

    // unexplained points
    rm_vis_->addPointCloud(scene_cloud_downsampled_, "scene2",rm_v4);
    pcl::PointCloud<pcl::PointXYZRGB> scene_pts_unexplained;
    scene_pts_unexplained.points.resize(rm.unexplained_in_neighborhood.size());
    pcl::copyPointCloud(*scene_cloud_downsampled_, rm.unexplained_in_neighborhood, scene_pts_unexplained);

    for(size_t i=0; i < rm.unexplained_in_neighborhood.size(); i++) {
        int sidx = rm.unexplained_in_neighborhood[i];
        const SceneT &spt = scene_cloud_downsampled_->points[sidx];
        pcl::PointXYZRGB &pt = scene_pts_unexplained.points[i];
        pt.x = spt.x; pt.y = spt.y; pt.z = spt.z;
        pt.r = pt.g = 0;
        pt.b = 128 + (rm.unexplained_in_neighborhood_weights[i] + min_weight) / (max_weight - min_weight);
    }
    rm_vis_->addPointCloud(scene_pts_unexplained.makeShared(), "scene_pt_unexplained", rm_v4);

    // smooth segmentation
    rm_vis_->addPointCloud (clusters_cloud_rgb_, "smooth_segments" , rm_v5);

    // label 0
    pcl::PointCloud<pcl::PointXYZRGB> cloud_for_label_zero;
    pcl::copyPointCloud(*scene_cloud_downsampled_, cloud_for_label_zero);
    size_t kept=0;
    for(size_t i=0; i< scene_cloud_downsampled_->points.size(); i++) {
        if(clusters_cloud_->points[i].label == 0)
            cloud_for_label_zero.points[kept++] = cloud_for_label_zero.points[i];
    }
    cloud_for_label_zero.points.resize(kept);
    rm_vis_->addPointCloud (cloud_for_label_zero.makeShared(), "label_0" , rm_v6);

    rm_vis_->resetCamera();
    rm_vis_->spin();
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::visualizeGOCues (const std::vector<bool> & active_solution, float cost, int times_evaluated)
{
    if(!vis_go_cues_) {
        vis_go_cues_.reset(new pcl::visualization::PCLVisualizer("visualizeGOCues"));
        vis_go_cues_->createViewPort(0, 0, 0.5, 0.5, viewport_scene_cues_);
        vis_go_cues_->createViewPort(0.5, 0, 1, 0.5, viewport_model_cues_);
        vis_go_cues_->createViewPort(0.5, 0.5, 1, 1, viewport_smooth_seg_);
        vis_go_cues_->createViewPort(0, 0.5, 0.5, 1, viewport_scene_and_hypotheses_);
    }

    vis_go_cues_->removeAllPointClouds();
    vis_go_cues_->removeAllShapes();

    std::ostringstream out;
    out << "Cost: " << std::setprecision(2) << cost << " , #Evaluations: " << times_evaluated;

    bool for_paper_ = false;
    bool show_weights_with_color_fading_ = true;

    if(for_paper_)
    {
        vis_go_cues_->setBackgroundColor (1, 1, 1);
    }
    else
    {
        vis_go_cues_->setBackgroundColor (0, 0, 0);
        vis_go_cues_->addText (out.str(), 1, 30, 16, 1, 1, 1, "cost_text", viewport_scene_and_hypotheses_);
        vis_go_cues_->addText ("Model inliers & outliers", 1, 30, 16, 1, 1, 1, "inliers_outliers", viewport_model_cues_);
        vis_go_cues_->addText ("Smooth segmentation", 1, 30, 16, 1, 1, 1, "smooth", viewport_smooth_seg_);
        vis_go_cues_->addText ("Explained, multiple assignment & clutter", 1, 30, 16, 1, 1, 1, "scene_cues", viewport_scene_cues_);
    }

    //scene
    pcl::visualization::PointCloudColorHandlerCustom<SceneT> random_handler_scene (scene_cloud_downsampled_, 200, 0, 0);
    vis_go_cues_->addPointCloud<SceneT> (scene_cloud_downsampled_, random_handler_scene, "scene_cloud", viewport_scene_and_hypotheses_);

    //smooth segmentation
    if(clusters_cloud_rgb_)
    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> random_handler (clusters_cloud_rgb_);
        vis_go_cues_->addPointCloud<pcl::PointXYZRGBA> (clusters_cloud_rgb_, random_handler, "smooth_cloud", viewport_smooth_seg_);
    }

    //display active hypotheses
    for(size_t i=0; i < active_solution.size(); i++)
    {
        if(active_solution[i])
        {
            //complete models

            HVRecognitionModel<ModelT> &rm = *recognition_models_[i];

            std::stringstream m;
            m << "model_" << i;

            if(poses_ply_.size() == 0)
            {
                pcl::visualization::PointCloudColorHandlerCustom<ModelT> handler_model (rm.complete_cloud_, 0, 255, 0);
                vis_go_cues_->addPointCloud<ModelT> (rm.complete_cloud_, handler_model, m.str(), viewport_scene_and_hypotheses_);
            }
            else
            {
                if(!rm.is_planar_)
                    vis_go_cues_->addModelFromPLYFile (ply_paths_[i], poses_ply_[i], m.str (), viewport_scene_and_hypotheses_);
                else
                    vis_go_cues_->addPolygonMesh (*(rm.plane_model_.convex_hull_), m.str(), viewport_scene_and_hypotheses_);
            }

            //model inliers and outliers
            std::stringstream cluster_name;
            cluster_name << "visible" << i;

            typename pcl::PointCloud<ModelT>::Ptr outlier_points (new pcl::PointCloud<ModelT> ());
            for (size_t j = 0; j < rm.outlier_indices_.size (); j++)
            {
                ModelT c_point;
                c_point.getVector3fMap () = rm.visible_cloud_->points[ rm.outlier_indices_[j] ].getVector3fMap ();
                outlier_points->push_back (c_point);
            }

            pcl::visualization::PointCloudColorHandlerCustom<ModelT> random_handler (rm.visible_cloud_, 255, 90, 0);
            vis_go_cues_->addPointCloud<ModelT> ( rm.visible_cloud_, random_handler, cluster_name.str (), viewport_model_cues_);

            cluster_name << "_outliers";

            pcl::visualization::PointCloudColorHandlerCustom<ModelT> random_handler_out (outlier_points, 0, 94, 22);
            vis_go_cues_->addPointCloud<ModelT> (outlier_points, random_handler_out, cluster_name.str (), viewport_model_cues_);
        }
    }

    vis_go_cues_->setRepresentationToSurfaceForAllActors();

    //display scene cues (explained points, multiply explained, clutter (smooth and normal)
    vis_go_cues_->addPointCloud<SceneT> (scene_cloud_downsampled_, random_handler_scene, "scene_cloud_viewport", viewport_scene_cues_);
    vis_go_cues_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "scene_cloud_viewport");

    //clutter...
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clutter (new pcl::PointCloud<pcl::PointXYZRGB> ());
    typename pcl::PointCloud<SceneT>::Ptr clutter_smooth (new pcl::PointCloud<SceneT> ());
    for (size_t j = 0; j < unexplained_by_RM_neighboorhods_.size (); j++)
    {
        if(unexplained_by_RM_neighboorhods_[j] >= (param_.clutter_regularizer_ - 0.01f) && explained_by_RM_[j] == 0 && (clusters_cloud_->points[j].label != 0 || param_.use_super_voxels_))
        {
            SceneT c_point;
            c_point.getVector3fMap () = scene_cloud_downsampled_->points[j].getVector3fMap ();
            clutter_smooth->push_back (c_point);
        }
        else if (unexplained_by_RM_neighboorhods_[j] > 0 && explained_by_RM_[j] == 0)
        {
            pcl::PointXYZRGB c_point;
            c_point.getVector3fMap () = scene_cloud_downsampled_->points[j].getVector3fMap ();

            if(show_weights_with_color_fading_)
            {
                c_point.r = round(255.0 * unexplained_by_RM_neighboorhods_[j]);
                c_point.g = 40;
                c_point.b = round(255.0 * unexplained_by_RM_neighboorhods_[j]);
            }
            else
            {
                c_point.r = 255.0;
                c_point.g = 40;
                c_point.b = 255.0;
            }
            clutter->push_back (c_point);
        }
    }

    //explained
    typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr explained_points (new pcl::PointCloud<pcl::PointXYZRGB> ());
    //typename pcl::PointCloud<SceneT>::Ptr explained_points (new pcl::PointCloud<SceneT> ());
    for (size_t j = 0; j < explained_by_RM_.size (); j++)
    {
        if (explained_by_RM_[j] == 1)
        {
            pcl::PointXYZRGB c_point;

            //if(show_weights_with_color_fading_)
            //{
            c_point.getVector3fMap () = scene_cloud_downsampled_->points[j].getVector3fMap ();
            c_point.b = 100 + explained_by_RM_distance_weighted_[j] * 155;
            c_point.r = c_point.g = 0;
            //}
            //else
            //{
            //    c_point.getVector3fMap () = scene_cloud_downsampled_->points[j].getVector3fMap ();
            //    c_point.b = 255;
            //    c_point.r = c_point.g = 0;
            //}
            explained_points->push_back (c_point);
        }
    }

    //duplicity
    typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr duplicity_points (new pcl::PointCloud<pcl::PointXYZRGB> ());
    for (size_t j = 0; j < explained_by_RM_.size (); j++)
    {
        if (explained_by_RM_[j] > 1)
        {
            pcl::PointXYZRGB c_point;
            c_point.getVector3fMap () = scene_cloud_downsampled_->points[j].getVector3fMap ();
            float curv_weight = getCurvWeight( scene_normals_->points[j].curvature );

            if( param_.multiple_assignment_penalize_by_one_ == 1)
            {
                c_point.r = c_point.g = c_point.b = 0;
                c_point.g = curv_weight * param_.duplicy_weight_test_ * 255;
            }
            else if( param_.multiple_assignment_penalize_by_one_ == 2)
            {
                if(show_weights_with_color_fading_)
                {
                    c_point.r = c_point.g = c_point.b = 0;
                    c_point.g = std::min(duplicates_by_RM_weighted_[j],1.0) * 255;
                }
                else
                    c_point.r = c_point.g = c_point.b = 0;
            }
            else
            {
                c_point.r = c_point.g = c_point.b = 0;
                c_point.g = 255;
            }

            duplicity_points->push_back (c_point);
        }
    }

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> random_handler_clutter (clutter);
    vis_go_cues_->addPointCloud<pcl::PointXYZRGB> (clutter, random_handler_clutter, "clutter", viewport_scene_cues_);
    vis_go_cues_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "clutter");

    pcl::visualization::PointCloudColorHandlerCustom<SceneT> random_handler_clutter_smooth (clutter_smooth, 255, 255, 0);
    vis_go_cues_->addPointCloud<SceneT> (clutter_smooth, random_handler_clutter_smooth, "clutter_smooth", viewport_scene_cues_);
    vis_go_cues_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "clutter_smooth");

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> random_handler_explained (explained_points);
    vis_go_cues_->addPointCloud<pcl::PointXYZRGB> (explained_points, random_handler_explained, "explained", viewport_scene_cues_);
    vis_go_cues_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "explained");

    //pcl::visualization::PointCloudColorHandlerCustom<SceneT> random_handler_dup (duplicity_points, 200, 200, 200);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> random_handler_dup (duplicity_points);

    vis_go_cues_->addPointCloud<pcl::PointXYZRGB> (duplicity_points, random_handler_dup, "dup", viewport_scene_cues_);
    vis_go_cues_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "dup");
    vis_go_cues_->spin();
}

}
