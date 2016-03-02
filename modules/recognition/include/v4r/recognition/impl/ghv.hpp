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
    //boost::posix_time::ptime start_time (boost::posix_time::microsec_clock::local_time ());
    float sign = 1.f;
    //update explained_by_RM
    if (active[changed])
    {
        //it has been activated
        updateExplainedVector (recognition_models_[changed]->explained_, recognition_models_[changed]->explained_distances_, explained_by_RM_,
                               explained_by_RM_distance_weighted, 1.f, changed);

        if(param_.detect_clutter_) {
            updateUnexplainedVector (recognition_models_[changed]->unexplained_in_neighborhood,
                                     recognition_models_[changed]->unexplained_in_neighborhood_weights, unexplained_by_RM_neighboorhods,
                                     recognition_models_[changed]->explained_, explained_by_RM_, 1.f);
        }

        updateCMDuplicity (recognition_models_[changed]->complete_cloud_occupancy_indices_, complete_cloud_occupancy_by_RM_, 1.f);
    }
    else
    {
        //it has been deactivated
        updateExplainedVector (recognition_models_[changed]->explained_, recognition_models_[changed]->explained_distances_, explained_by_RM_,
                               explained_by_RM_distance_weighted, -1.f, changed);

        if(param_.detect_clutter_) {
            updateUnexplainedVector (recognition_models_[changed]->unexplained_in_neighborhood,
                                     recognition_models_[changed]->unexplained_in_neighborhood_weights, unexplained_by_RM_neighboorhods,
                                     recognition_models_[changed]->explained_, explained_by_RM_, -1.f);
        }
        updateCMDuplicity (recognition_models_[changed]->complete_cloud_occupancy_indices_, complete_cloud_occupancy_by_RM_, -1.f);
        sign = -1.f;
    }

    double duplicity = static_cast<double> (getDuplicity ());
    //duplicity = 0.f; //ATTENTION!!
    double good_info = getExplainedValue ();

    double unexplained_info = getPreviousUnexplainedValue ();
    if(!param_.detect_clutter_) {
        unexplained_info = 0;
    }

    double bad_info = static_cast<double> (getPreviousBadInfo ()) + (recognition_models_[changed]->outliers_weight_
                                                                     * static_cast<double> (recognition_models_[changed]->outlier_indices_.size ())) * sign;

    setPreviousBadInfo (bad_info);

    double duplicity_cm = static_cast<double> (getDuplicityCM ()) * param_.w_occupied_multiple_cm_;
    //float duplicity_cm = 0;

    //boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time ();
    //std::cout << (end_time - start_time).total_microseconds () << " microsecs" << std::endl;
    double cost = (good_info - bad_info - duplicity - unexplained_info - duplicity_cm - countActiveHypotheses (active) - countPointsOnDifferentPlaneSides(active)) * -1.f;

//    std::cout << "COST: " << cost << " (good info: " << good_info << ", bad _info: " << bad_info << ", duplicity:" << duplicity <<
//                 ", unexplained_info: " << unexplained_info << ", duplicity_cm: " << duplicity_cm <<
//                 ", ActiveHypotheses: " << countActiveHypotheses (active) <<
//                 ", PointsOnDifferentPlaneSides: " <<  countPointsOnDifferentPlaneSides(active) << ")" << std::endl;


    if(cost_logger_) {
        cost_logger_->increaseEvaluated();
        cost_logger_->addCostEachTimeEvaluated(cost);
    }

    //ntimes_evaluated_++;
    return static_cast<mets::gol_type> (cost); //return the dual to our max problem
}


template<typename ModelT, typename SceneT>
double
GHV<ModelT, SceneT>::countActiveHypotheses (const std::vector<bool> & sol)
{
    double c = 0;
    for (size_t i = 0; i < sol.size (); i++)
    {
        if (sol[i]) {
            //c++;
            //c += static_cast<double>(recognition_models_[i]->explained_.size()) * active_hyp_penalty_ + min_contribution_;
            c += static_cast<double>(recognition_models_[i]->explained_.size()) / 2.f * recognition_models_[i]->hyp_penalty_ + min_contribution_;
        }
    }

    return c;
    //return static_cast<float> (c) * active_hyp_penalty_;
}

template<typename ModelT, typename SceneT>
double
GHV<ModelT, SceneT>::
countPointsOnDifferentPlaneSides (const std::vector<bool> & sol,
                                  bool print)
{
    if(!param_.use_points_on_plane_side_)
        return 0;

    size_t recog_models = recognition_models_.size() - planar_models_.size();
    double c=0;
    for(size_t i=0; i < planar_models_.size(); i++)
    {
        if( (recog_models + i) >= recognition_models_.size() )
        {
            std::cout << "i:" << i << std::endl
                      << "recog models:" << recog_models << std::endl
                      << "recogition models:" << recognition_models_.size() << std::endl
                      << "solution size:" << sol.size() << std::endl
                      << "planar models size:" << planar_models_.size() << std::endl;
        }

        assert( (recog_models + i) < recognition_models_.size());
        assert( (recog_models + i) < sol.size());
        if(sol[recog_models + i])
        {
            std::map<size_t, size_t>::iterator it1;
            it1 = model_to_planar_model_.find(recog_models + i);

            if(print)
                std::cout << "plane is active:" << recognition_models_[recog_models + i]->id_s_ << std::endl;

            for(size_t j=0; j < points_one_plane_sides_[it1->second].size(); j++)
            {
                if(sol[j])
                {
                    c += points_one_plane_sides_[it1->second][j];
                    if(print)
                        std::cout << "Adding to c:" << points_one_plane_sides_[it1->second][j] << " " << recognition_models_[j]->id_s_ << " " << recognition_models_[j]->id_ << " plane_id:" << it1->second << std::endl;
                }
            }
        }
    }

    if(print)
    {
        for(size_t kk=0; kk < points_one_plane_sides_.size(); kk++)
        {
            for(size_t kkk=0; kkk < points_one_plane_sides_[kk].size(); kkk++)
            {
                std::cout << "i:" << kkk << " val:" << points_one_plane_sides_[kk][kkk] << " ";
            }

            std::cout << std::endl;
        }
    }
    return c;
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::addPlanarModels(std::vector<PlaneModel<ModelT> > & models)
{
    planar_models_ = models;
    model_to_planar_model_.clear();
    //iterate through the planar models and append them to complete_models_?

    size_t size_start = visible_models_.size();
    for(size_t i=0; i < planar_models_.size(); i++)
    {
        model_to_planar_model_[size_start + i] = i;
        typename pcl::PointCloud<SceneT>::Ptr plane_cloud = planar_models_[i].projectPlaneCloud();
        complete_models_.push_back(plane_cloud);

        ZBuffering<ModelT, SceneT> zbuffer_scene (param_.zbuffer_scene_resolution_, param_.zbuffer_scene_resolution_, 1.f);
        if (!occlusion_cloud_->isOrganized ())
            zbuffer_scene.computeDepthMap (*occlusion_cloud_, true);

        //self-occlusions
        typename pcl::PointCloud<ModelT>::Ptr filtered (new pcl::PointCloud<ModelT> (*plane_cloud));
        typename pcl::PointCloud<ModelT>::ConstPtr const_filtered(new pcl::PointCloud<ModelT> (*filtered));

        std::vector<int> indices_cloud_occlusion;

        if (occlusion_cloud_->isOrganized ())
            filtered = filter<ModelT,SceneT> (*occlusion_cloud_, *const_filtered, param_.focal_length_, param_.occlusion_thres_, indices_cloud_occlusion);
        else
            zbuffer_scene.filter (*const_filtered, *filtered, param_.occlusion_thres_);

        visible_models_.push_back (filtered);

        pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());
        normals->points.resize(filtered->points.size());
        Eigen::Vector3f plane_normal;
        plane_normal[0] = planar_models_[i].coefficients_.values[0];
        plane_normal[1] = planar_models_[i].coefficients_.values[1];
        plane_normal[2] = planar_models_[i].coefficients_.values[2];

        for(size_t k=0; k < filtered->points.size(); k++)
            normals->points[k].getNormalVector3fMap() = plane_normal;

        visible_normal_models_.push_back(normals);
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
                {
                    clusters_cloud->points[clusters[i].indices[j]].label = label;
                }
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
    scene_LAB_values_.resize(scene_cloud_downsampled_->points.size());
    scene_RGB_values_.resize(scene_cloud_downsampled_->points.size());
    scene_GS_values_.resize(scene_cloud_downsampled_->points.size());

#pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i < scene_cloud_downsampled_->points.size(); i++)
    {
        float rgb_s;
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

            float LRefs, aRefs, bRefs;
            color_transf_omp_.RGB2CIELAB_normalized(rs, gs, bs, LRefs, aRefs, bRefs);
            scene_LAB_values_[i] = Eigen::Vector3f(LRefs, aRefs, bRefs);

            float rsf,gsf,bsf;
            rsf = static_cast<float>(rs) / 255.f;
            gsf = static_cast<float>(gs) / 255.f;
            bsf = static_cast<float>(bs) / 255.f;
            scene_RGB_values_[i] = Eigen::Vector3f(rsf,gsf,bsf);
            scene_GS_values_[i] = (rsf + gsf + bsf) / 3.f;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename ModelT, typename SceneT>
bool
GHV<ModelT, SceneT>::initialize()
{
    //clear stuff
    recognition_models_.clear ();
    unexplained_by_RM_neighboorhods.clear ();
    explained_by_RM_distance_weighted.clear ();
    previous_explained_by_RM_distance_weighted.clear ();
    explained_by_RM_.clear ();
    explained_by_RM_model.clear();
    complete_cloud_occupancy_by_RM_.clear ();
    mask_.clear ();
    mask_.resize (complete_models_.size (), false);

    if(!scene_and_normals_set_from_outside_ || scene_cloud_downsampled_->points.size() != scene_normals_->points.size())
    {
        if(!scene_normals_)
            scene_normals_.reset (new pcl::PointCloud<pcl::Normal> ());

        size_t kept = 0;
        for (size_t i = 0; i < scene_cloud_downsampled_->points.size (); ++i) {
            if ( pcl::isFinite( scene_cloud_downsampled_->points[i]) )
                scene_cloud_downsampled_->points[kept++] = scene_cloud_downsampled_->points[i];
        }

        scene_cloud_downsampled_->points.resize(kept);
        scene_cloud_downsampled_->width = kept;
        scene_cloud_downsampled_->height = 1;

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

    scene_curvature_.resize (scene_cloud_downsampled_->points.size ());

    for(size_t k=0; k < scene_normals_->points.size(); k++)
        scene_curvature_[k] = scene_normals_->points[k].curvature;

    explained_by_RM_.resize (scene_cloud_downsampled_->points.size (), 0);
    duplicates_by_RM_weighted_.resize (scene_cloud_downsampled_->points.size (), 0);
    explained_by_RM_model.resize (scene_cloud_downsampled_->points.size (), -1);
    explained_by_RM_distance_weighted.resize (scene_cloud_downsampled_->points.size (), 0);
    previous_explained_by_RM_distance_weighted.resize (scene_cloud_downsampled_->points.size ());
    unexplained_by_RM_neighboorhods.resize (scene_cloud_downsampled_->points.size (), 0.f);

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
        //compute scene LAB values
        if(!param_.ignore_color_even_if_exists_)
            convertColor();
        }
    }

    valid_model_.resize(complete_models_.size ());
    {
    pcl::ScopeTime t("Computing cues");
    recognition_models_.resize (complete_models_.size ());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < complete_models_.size (); i++)
    {
        recognition_models_[i].reset (new GHVRecognitionModel<ModelT> ());
        valid_model_[i] = addModel(i, *recognition_models_[i]); // check if model is valid
    }
    }

    // check if any valid model exists, otherwise there is nothing to verify
    bool valid_model_exists = false;
    for (size_t i = 0; i < complete_models_.size (); i++)
    {
        if(valid_model_[i]){
            valid_model_exists = true;
            break;
        }
    }

    if (!valid_model_exists) {
        std::cout << "No valid model exists. " << std::endl;
        return false;
    }

    //compute the bounding boxes for the models to create an occupancy grid
    {
        pcl::ScopeTime t_cloud_occupancy ("complete_cloud_occupancy_by_RM_");
        ModelT min_pt_all, max_pt_all;
        min_pt_all.x = min_pt_all.y = min_pt_all.z = std::numeric_limits<float>::max ();
        max_pt_all.x = max_pt_all.y = max_pt_all.z = std::numeric_limits<float>::min ();

        for (size_t i = 0; i < recognition_models_.size (); i++)
        {
            if(!valid_model_[i])
                continue;

            ModelT min_pt, max_pt;
            pcl::getMinMax3D (*complete_models_[i], min_pt, max_pt);

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

        size_t size_x, size_y, size_z;
        size_x = static_cast<size_t> (std::abs (max_pt_all.x - min_pt_all.x) / param_.res_occupancy_grid_ + 1.5f);  // rounding up and add 1
        size_y = static_cast<size_t> (std::abs (max_pt_all.y - min_pt_all.y) / param_.res_occupancy_grid_ + 1.5f);
        size_z = static_cast<size_t> (std::abs (max_pt_all.z - min_pt_all.z) / param_.res_occupancy_grid_ + 1.5f);

        complete_cloud_occupancy_by_RM_.resize (size_x * size_y * size_z, 0);

        for (size_t i = 0; i < recognition_models_.size (); i++)
        {
            if(!valid_model_[i])
                continue;

            std::vector<bool> banned_vector(size_x * size_y * size_z, false);   // what's the purpose of this?

            recognition_models_[i]->complete_cloud_occupancy_indices_.resize(complete_models_[i]->points.size ());
            size_t used = 0;

            for (size_t j = 0; j < complete_models_[i]->points.size (); j++)
            {
                size_t pos_x, pos_y, pos_z;
                pos_x = static_cast<size_t>( (complete_models_[i]->points[j].x - min_pt_all.x) / param_.res_occupancy_grid_);
                pos_y = static_cast<size_t>( (complete_models_[i]->points[j].y - min_pt_all.y) / param_.res_occupancy_grid_);
                pos_z = static_cast<size_t>( (complete_models_[i]->points[j].z - min_pt_all.z) / param_.res_occupancy_grid_);

                size_t idx = pos_z * size_x * size_y + pos_y * size_x + pos_x;

                if ( !banned_vector[idx] )
                {
                    recognition_models_[i]->complete_cloud_occupancy_indices_[used] = idx;
                    banned_vector[idx] = true;
                    used++;
                }
            }

            recognition_models_[i]->complete_cloud_occupancy_indices_.resize(used);
        }
    }

    {
        pcl::ScopeTime t_clutter("Compute clutter cue at once");
        computeClutterCueAtOnce();
    }

    points_explained_by_rm_.clear ();
    points_explained_by_rm_.resize (scene_cloud_downsampled_->points.size ());
    for (size_t j = 0; j < recognition_models_.size (); j++)
    {
        boost::shared_ptr<GHVRecognitionModel<ModelT> > recog_model = recognition_models_[j];

        for (size_t i = 0; i < recog_model->explained_.size (); i++)
            points_explained_by_rm_[ recog_model->explained_[i] ].push_back (recog_model);
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

    double w = 1.f - std::min(1., p_curvature / param_.duplicity_curvature_max_);
    return w;
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::updateExplainedVector (const std::vector<int> & vec,
                                            const std::vector<float> & vec_float,
                                            std::vector<int> & explained,
                                            std::vector<double> & explained_by_RM_distance_weighted__not_used,
                                            float sign,
                                            int model_id)
{
    double add_to_explained = 0;
    double add_to_duplicity_ = 0;
    const int id_model_ = recognition_models_[model_id]->id_;

    for (size_t i = 0; i < vec.size (); i++)
    {
        bool prev_dup = explained[vec[i]] > 1;
        //bool prev_explained = explained_[vec[i]] == 1;
        int prev_explained = explained[vec[i]];
        double prev_explained_value = explained_by_RM_distance_weighted[vec[i]];

        explained[vec[i]] += static_cast<int> (sign);
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
                explained_by_RM_distance_weighted[vec[i]] = vec_float[i];
                previous_explained_by_RM_distance_weighted[vec[i]].push(std::make_pair(id_model_, vec_float[i]));
            }
            else
            {
                //point was already explained
                if(vec_float[i] > prev_explained_value)
                {
                    previous_explained_by_RM_distance_weighted[vec[i]].push(std::make_pair(id_model_, vec_float[i]));
                    explained_by_RM_distance_weighted[vec[i]] = (double)vec_float[i];
                }
                else
                {

                    //size_t anfang = previous_explained_by_RM_distance_weighted[vec[i]].size();

                    //if it is smaller, we should keep the value in case the greater value gets removed
                    //need to sort the stack
                    if(previous_explained_by_RM_distance_weighted[vec[i]].size() == 0)
                    {
                        previous_explained_by_RM_distance_weighted[vec[i]].push(std::make_pair(id_model_, vec_float[i]));
                    }
                    else
                    {
                        //sort and find the appropiate position

                        std::stack<std::pair<int, float>, std::vector<std::pair<int, float> > > kept;
                        while(previous_explained_by_RM_distance_weighted[vec[i]].size() > 0)
                        {
                            std::pair<int, double> p = previous_explained_by_RM_distance_weighted[vec[i]].top();
                            if(p.second < vec_float[i])
                            {
                                //should come here
                                break;
                            }

                            kept.push(p);
                            previous_explained_by_RM_distance_weighted[vec[i]].pop();
                        }

                        previous_explained_by_RM_distance_weighted[vec[i]].push(std::make_pair(id_model_, vec_float[i]));

                        while(!kept.empty())
                        {
                            previous_explained_by_RM_distance_weighted[vec[i]].push(kept.top());
                            kept.pop();
                        }
                    }
                }
            }
        }
        else
        {
            std::stack<std::pair<int, float>, std::vector<std::pair<int, float> > > kept;

            while(previous_explained_by_RM_distance_weighted[vec[i]].size() > 0)
            {
                std::pair<int, double> p = previous_explained_by_RM_distance_weighted[vec[i]].top();

                if(p.first == id_model_)
                {
                    //found it
                }
                else
                {
                    kept.push(p);
                }

                previous_explained_by_RM_distance_weighted[vec[i]].pop();
            }

            while(!kept.empty())
            {
                previous_explained_by_RM_distance_weighted[vec[i]].push(kept.top());
                kept.pop();
            }

            if(prev_explained == 1)
            {
                //was only explained by this hypothesis
                explained_by_RM_distance_weighted[vec[i]] = 0;
            }
            else
            {
                //there is at least another hypothesis explaining this point
                //assert(previous_explained_by_RM_distance_weighted[vec[i]].size() > 0);
                std::pair<int, double> p = previous_explained_by_RM_distance_weighted[vec[i]].top();

                double previous = p.second;
                explained_by_RM_distance_weighted[vec[i]] = previous;
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
        float curv_weight = getCurvWeight(scene_curvature_[vec[i]]);
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
            if ((explained[vec[i]] > 1) && prev_dup)
            { //its still a duplicate, do nothing

            }
            else if ((explained[vec[i]] == 1) && prev_dup)
            { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
                add_to_duplicity_ -= curv_weight;
            }
            else if ((explained[vec[i]] > 1) && !prev_dup)
            { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
                add_to_duplicity_ += curv_weight;
            }
        }
        else if( param_.multiple_assignment_penalize_by_one_ == 2)
        {
            if ((explained[vec[i]] > 1) && prev_dup)
            { //its still a duplicate, add or remove current explained value

                //float add_for_this_p = std::
                add_to_duplicity_ += curv_weight * vec_float[i] * sign;
                duplicates_by_RM_weighted_[vec[i]] += curv_weight * vec_float[i] * sign;
            }
            else if ((explained[vec[i]] == 1) && prev_dup)
            { //if was duplicate before, now its not, remove current explained weight and old one
                add_to_duplicity_ -= duplicates_by_RM_weighted_[vec[i]];
                duplicates_by_RM_weighted_[vec[i]] = 0;
            }
            else if ((explained[vec[i]] > 1) && !prev_dup)
            { //it was not a duplicate but it is now, add prev explained value + current explained weight
                add_to_duplicity_ += curv_weight * (prev_explained_value + vec_float[i]);
                duplicates_by_RM_weighted_[vec[i]] = curv_weight * (prev_explained_value + vec_float[i]);
            }
        }
        else
        {
            if ((explained[vec[i]] > 1) && prev_dup)
            { //its still a duplicate
                //add_to_duplicity_ += vec_float[i] * static_cast<int> (sign); //so, just add or remove one
                //add_to_duplicity_ += vec_float[i] * static_cast<int> (sign) * duplicy_weight_test_ * curv_weight; //so, just add or remove one
                add_to_duplicity_ += static_cast<int> (sign) * param_.duplicy_weight_test_ * curv_weight; //so, just add or remove one
            }
            else if ((explained[vec[i]] == 1) && prev_dup)
            { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
                //add_to_duplicity_ -= prev_explained_value; // / 2.f; //explained_by_RM_distance_weighted[vec[i]];
                //add_to_duplicity_ -= prev_explained_value * duplicy_weight_test_ * curv_weight;
                add_to_duplicity_ -= param_.duplicy_weight_test_ * curv_weight * 2;
            }
            else if ((explained[vec[i]] > 1) && !prev_dup)
            { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
                //add_to_duplicity_ += explained_by_RM_distance_weighted[vec[i]];
                //add_to_duplicity_ += explained_by_RM_distance_weighted[vec[i]] * duplicy_weight_test_ * curv_weight;
                add_to_duplicity_ += param_.duplicy_weight_test_ * curv_weight  * 2;
            }
        }

        add_to_explained += explained_by_RM_distance_weighted[vec[i]] - prev_explained_value;
    }

    //update explained and duplicity values...
    previous_explained_value += add_to_explained;
    previous_duplicity_ += add_to_duplicity_;
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::updateCMDuplicity (const std::vector<int> & vec, std::vector<int> & occupancy_vec, float sign)
{
    int add_to_duplicity_ = 0;
    for (size_t i = 0; i < vec.size (); i++)
    {
        if( (vec[i] > occupancy_vec.size() ) || ( i > vec.size()))
        {
            std::cout << occupancy_vec.size() << " " << vec[i] << " " << vec.size() << " " << i << std::endl;
            throw std::runtime_error("something is wrong with the occupancy grid.");
        }

        bool prev_dup = occupancy_vec[vec[i]] > 1;
        occupancy_vec[vec[i]] += static_cast<int> (sign);
        if ((occupancy_vec[vec[i]] > 1) && prev_dup)
        { //its still a duplicate, we are adding
            add_to_duplicity_ += static_cast<int> (sign); //so, just add or remove one
        }
        else if ((occupancy_vec[vec[i]] == 1) && prev_dup)
        { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
            add_to_duplicity_ -= 2;
        }
        else if ((occupancy_vec[vec[i]] > 1) && !prev_dup)
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
            //if (explained_[i] == 1) //only counts points that are explained once
        {
            //explained_info += std::min(explained_by_RM_distance_weighted[i], 1.0);
            explained_info += explained_by_RM_distance_weighted[i];
        }

        if (explained[i] > 1)
        {
            //duplicity += explained_by_RM_distance_weighted[i];
            //float curv_weight = std::min(duplicity_curvature_ - scene_curvature_[i], 0.f);

            float curv_weight = getCurvWeight(scene_curvature_[i]);

            if(param_.multiple_assignment_penalize_by_one_ == 1)
            {
                duplicity += curv_weight;
            }
            else if(param_.multiple_assignment_penalize_by_one_ == 2)
            {
                duplicity += duplicates_by_RM_weighted_[i];
                /*if(duplicates_by_RM_weighted_[i] > 1)
                {
                    PCL_WARN("duplicates_by_RM_weighted_[i] higher than one %f\n", duplicates_by_RM_weighted_[i]);
                }*/
            }
            else
            {
                //curv_weight = 1.f;
                //duplicity += explained_by_RM_distance_weighted[i] * duplicy_weight_test_ * curv_weight;
                duplicity += param_.duplicy_weight_test_ * curv_weight * explained[i];
            }
        }
    }

    return explained_info;
}

template<typename ModelT, typename SceneT>
double
GHV<ModelT, SceneT>::getExplainedByIndices(const std::vector<int> & indices, const std::vector<float> & explained_values,
                                                     const std::vector<double> & explained_by_RM, std::vector<int> & indices_to_update_in_RM_local)
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
GHV<ModelT, SceneT>::fill_structures(std::vector<int> & cc_indices, std::vector<bool> & initial_solution, GHVSAModel<ModelT, SceneT> & model)
{
    for (size_t j = 0; j < recognition_models_.size (); j++)
    {
        if(!initial_solution[j])
            continue;

        boost::shared_ptr<GHVRecognitionModel<ModelT> > recog_model = recognition_models_[j];
        for (size_t i = 0; i < recog_model->explained_.size (); i++)
        {
            explained_by_RM_[recog_model->explained_[i]]++;
            //explained_by_RM_distance_weighted[recog_model->explained_[i]] += recog_model->explained_distances_[i];
            explained_by_RM_distance_weighted[recog_model->explained_[i]] = std::max(explained_by_RM_distance_weighted[recog_model->explained_[i]], (double)recog_model->explained_distances_[i]);
        }

        if (param_.detect_clutter_)
        {
            for (size_t i = 0; i < recog_model->unexplained_in_neighborhood.size (); i++)
            {
                unexplained_by_RM_neighboorhods[recog_model->unexplained_in_neighborhood[i]] += recog_model->unexplained_in_neighborhood_weights[i];
            }
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

        boost::shared_ptr<GHVRecognitionModel<ModelT> > recog_model = recognition_models_[j];
        for (size_t i = 0; i < recog_model->explained_.size (); i++)
        {
            if(explained_by_RM_[recog_model->explained_[i]] > 1)
            {
                float curv_weight = getCurvWeight(scene_curvature_[recog_model->explained_[i]]);
                duplicates_by_RM_weighted_[recog_model->explained_[i]] += curv_weight * (double)recog_model->explained_distances_[i];
            }
        }
    }

    int occupied_multiple = 0;
    for (size_t i = 0; i < complete_cloud_occupancy_by_RM_.size (); i++)
    {
        if (complete_cloud_occupancy_by_RM_[i] > 1)
        {
            occupied_multiple += complete_cloud_occupancy_by_RM_[i];
        }
    }

    //do optimization
    //Define model SAModel, initial solution is all models activated

    double duplicity;
    double good_information = getTotalExplainedInformation (explained_by_RM_, explained_by_RM_distance_weighted, duplicity);
    double bad_information = 0;
    double unexplained_in_neighboorhod = 0;

    if(param_.detect_clutter_)
        unexplained_in_neighboorhod = getUnexplainedInformationInNeighborhood (unexplained_by_RM_neighboorhods, explained_by_RM_);

    for (size_t i = 0; i < initial_solution.size (); i++)
    {
        if (initial_solution[i])
            bad_information += static_cast<double> (recognition_models_[i]->outliers_weight_) * recognition_models_[i]->outlier_indices_.size ();
    }

    setPreviousDuplicityCM (occupied_multiple);
    setPreviousExplainedValue (good_information);
    setPreviousDuplicity (duplicity);
    setPreviousBadInfo (bad_information);
    setPreviousUnexplainedValue (unexplained_in_neighboorhod);

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
    explained_by_RM_distance_weighted.clear();
    previous_explained_by_RM_distance_weighted.clear();
    unexplained_by_RM_neighboorhods.clear();
    complete_cloud_occupancy_by_RM_.clear();
    explained_by_RM_model.clear();
    duplicates_by_RM_weighted_.clear();

    explained_by_RM_.resize (scene_cloud_downsampled_->points.size (), 0);
    duplicates_by_RM_weighted_.resize (scene_cloud_downsampled_->points.size (), 0);
    explained_by_RM_distance_weighted.resize (scene_cloud_downsampled_->points.size (), 0);
    previous_explained_by_RM_distance_weighted.resize (scene_cloud_downsampled_->points.size ());
    unexplained_by_RM_neighboorhods.resize (scene_cloud_downsampled_->points.size (), 0.f);
    complete_cloud_occupancy_by_RM_.resize(kk, 0);
    explained_by_RM_model.resize (scene_cloud_downsampled_->points.size (), -1);
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::SAOptimize (std::vector<int> & cc_indices, std::vector<bool> & initial_solution)
{

    //temporal copy of recogniton_models_
    std::vector<boost::shared_ptr<GHVRecognitionModel<ModelT> > > recognition_models_copy;
    recognition_models_copy = recognition_models_;

    recognition_models_.clear ();

    for (size_t j = 0; j < cc_indices.size (); j++)
        recognition_models_.push_back (recognition_models_copy[cc_indices[j]]);

    clear_structures();

    GHVSAModel<ModelT, SceneT> model;
    fill_structures(cc_indices, initial_solution, model);

    GHVSAModel<ModelT, SceneT> * best = new GHVSAModel<ModelT, SceneT> (model);

    GHVmove_manager<ModelT, SceneT> neigh (static_cast<int> (cc_indices.size ()), param_.use_replace_moves_);
    boost::shared_ptr<std::map< std::pair<int, int>, bool > > intersect_map;
    intersect_map.reset(new std::map< std::pair<int, int>, bool >);

    if(param_.use_replace_moves_ || (planar_models_.size() > 0))
    {
        pcl::ScopeTime t("compute intersection map...");

        std::vector<int> n_conflicts(recognition_models_.size() * recognition_models_.size(), 0);
        for (size_t k = 0; k < points_explained_by_rm_.size (); k++)
        {
            if (points_explained_by_rm_[k].size() > 1)
            {
                // this point could be a conflict
                for (size_t kk = 0; (kk < points_explained_by_rm_[k].size ()); kk++)
                {
                    for (size_t jj = (kk+1); (jj < points_explained_by_rm_[k].size ()); jj++)
                    {
                        //std::cout << points_explained_by_rm_[k][kk]->id_ << " " << points_explained_by_rm_[k][jj]->id_ << " " << n_conflicts.size() << std::endl;
                        //conflict, THIS MIGHT CAUSE A SEG FAULT AT SOME POINT!! ATTENTION! //todo
                        //i will probably need a vector going from id_ to recognition_models indices
                        assert(points_explained_by_rm_[k][kk]->id_ * recognition_models_.size() + points_explained_by_rm_[k][jj]->id_ < n_conflicts.size());
                        assert(points_explained_by_rm_[k][jj]->id_ * recognition_models_.size() + points_explained_by_rm_[k][kk]->id_ < n_conflicts.size());
                        n_conflicts[points_explained_by_rm_[k][kk]->id_ * recognition_models_.size() + points_explained_by_rm_[k][jj]->id_]++;
                        n_conflicts[points_explained_by_rm_[k][jj]->id_ * recognition_models_.size() + points_explained_by_rm_[k][kk]->id_]++;
                    }
                }
            }
        }

        int num_conflicts = 0;
        for (size_t i = 0; i < recognition_models_.size (); i++)
        {
            //std::cout << "id:" << recognition_models_[i]->id_ << std::endl;
            for (size_t j = (i+1); j < recognition_models_.size (); j++)
            {
                //assert(n_conflicts[i * recognition_models_.size() + j] == n_conflicts[j * recognition_models_.size() + i]);
                //std::cout << n_conflicts[i * recognition_models_.size() + j] << std::endl;
                bool conflict = (n_conflicts[i * recognition_models_.size() + j] > 10);
                std::pair<int, int> p = std::make_pair<int, int> (static_cast<int> (i), static_cast<int> (j));
                (*intersect_map)[p] = conflict;
                if(conflict)
                {
                    num_conflicts++;
                }
            }
        }

        //#define VIS_PLANES

        if(planar_models_.size() > 0 && param_.use_points_on_plane_side_)
        {
            //compute for each planar model, how many points for the other hypotheses (if in conflict) are on each side of the plane
            //std::cout << "recognition_models size:" << recognition_models_.size() << std::endl;

#ifdef VIS_PLANES
            pcl::visualization::PCLVisualizer vis("TEST");
#endif
            points_one_plane_sides_.clear();
            points_one_plane_sides_.resize(planar_models_.size());

            for(size_t i=0; i < recognition_models_.size(); i++)
            {
                std::map<size_t, size_t>::iterator it1, it2;
                it1 = model_to_planar_model_.find(i);
                if(it1 != model_to_planar_model_.end())
                {

                    points_one_plane_sides_[it1->second].resize(recognition_models_.size() - planar_models_.size() + 1, 0.f);

                    //is a plane, check how many points from other hypotheses are at each side of the plane
                    for(size_t j=0; j < recognition_models_.size(); j++)
                    {

                        if(i == j)
                            continue;

                        it2 = model_to_planar_model_.find(j);
                        if(it2 != model_to_planar_model_.end())
                        {
                            //both are planes, ignore
                            //std::cout << recognition_models_[i]->id_s_ << " " << recognition_models_[j]->id_s_ << std::endl;
                            continue;
                        }

                        assert(recognition_models_[j]->id_ < (recognition_models_.size() - planar_models_.size() + 1));

                        bool conflict = (n_conflicts[recognition_models_[i]->id_ * recognition_models_.size() + recognition_models_[j]->id_] > 0);
                        if(!conflict)
                            continue;

                        //is not a plane and is in conflict, compute points on both sides
                        Eigen::Vector2f side_count = Eigen::Vector2f::Zero();
                        for(size_t k=0; k < complete_models_[j]->points.size(); k++)
                        {
                            std::vector<float> p = planar_models_[it1->second].coefficients_.values;
                            Eigen::Vector3f xyz_p = complete_models_[j]->points[k].getVector3fMap();
                            float val = xyz_p[0] * p[0] + xyz_p[1] * p[1] + xyz_p[2] * p[2] + p[3];

                            if(std::abs(val) <= param_.inliers_threshold_)
                            {
                                /*if(val < 0)
                                    side_count[0]+= 0.1f;
                                else
                                    side_count[1]+= 0.1f;*/

                                continue;
                            }

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
                            assert(recognition_models_[j]->id_ < points_one_plane_sides_[it1->second].size());
                            points_one_plane_sides_[it1->second][recognition_models_[j]->id_] = min_side;
                            //std::cout << recognition_models_[i]->id_s_ << " " << recognition_models_[j]->id_s_ << std::endl;
                            //std::cout << min_side << " " << max_side << std::endl;

#ifdef VIS_PLANES
                            vis.addPointCloud<SceneT>(scene_cloud_downsampled_, "scene");
                            vis.addPointCloud<ModelT>(recognition_models_[j]->complete_cloud_, "complete_cloud");
                            vis.addPolygonMesh(*(planar_models_[it1->second].convex_hull_), "polygon");

                            vis.spin();
                            vis.removeAllPointClouds();
                            vis.removeAllShapes();
#endif
                        }
                        else
                            points_one_plane_sides_[it1->second][recognition_models_[j]->id_] = 0;
                    }
                }
            }

            /*std::cout << "recognition_models size:" << recognition_models_.size() << std::endl;
            std::cout << "complete models size:" << complete_models_.size() << std::endl;

            for(size_t kk=0; kk < points_one_plane_sides_.size(); kk++)
            {
                for(size_t kkk=0; kkk < points_one_plane_sides_[kk].size(); kkk++)
                {
                    std::cout << points_one_plane_sides_[kk][kkk] << " ";
                }

                std::cout << std::endl;
            }*/
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
        mets::local_search<GHVmove_manager<ModelT, SceneT> > local ( model, *(cost_logger_.get()), neigh, 0, LS_short_circuit_);
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

            typename boost::shared_ptr<GHVRecognitionModel<ModelT> > recog_model = recognition_models_[i];

            //visualize
            if( !param_.ignore_color_even_if_exists_ && visualize_accepted_)
            {

                std::map<size_t, size_t>::iterator it1;
                it1 = model_to_planar_model_.find(i);
                if(it1 != model_to_planar_model_.end())
                {
                    continue;
                }

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud_gs(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::copyPointCloud(*recog_model->visible_cloud_, *model_cloud_gs);

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud_gs_specified(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

                if(param_.color_space_ == 1 || param_.color_space_ == 5)
                {
                    pcl::copyPointCloud(*recog_model->visible_cloud_, *model_cloud_gs_specified);
                    for(size_t k=0; k < model_cloud_gs_specified->points.size(); k++)
                    {
                        model_cloud_gs_specified->points[k].r = recog_model->cloud_RGB_[k][0] * 255;
                        model_cloud_gs_specified->points[k].g = recog_model->cloud_RGB_[k][1] * 255;
                        model_cloud_gs_specified->points[k].b = recog_model->cloud_RGB_[k][2] * 255;
                    }

                    pcl::copyPointCloud(*scene_cloud_downsampled_, *scene_cloud);
                    //pcl::copyPointCloud(*scene_cloud, recognition_models_[i]->explained_, *scene_cloud);

                }
                else if(param_.color_space_ == 0 || param_.color_space_ == 6)
                {
                    pcl::copyPointCloud(*recog_model->visible_cloud_, *model_cloud_gs_specified);
                    for(size_t k=0; k < model_cloud_gs_specified->points.size(); k++)
                    {
                        model_cloud_gs_specified->points[k].r = recog_model->cloud_LAB_[k][0] * 255;
                        model_cloud_gs_specified->points[k].g = (recog_model->cloud_LAB_[k][1] + 1.f) / 2.f * 255;
                        model_cloud_gs_specified->points[k].b = (recog_model->cloud_LAB_[k][2] + 1.f) / 2.f * 255;
                    }

                    for(size_t k=0; k < model_cloud_gs->points.size(); k++)
                    {
                        //float gs = (scene_cloud->points[k].r + scene_cloud->points[k].g + scene_cloud->points[k].b) / 3.f;
                        uint32_t rgb = *reinterpret_cast<int*> (&model_cloud_gs->points[k].rgb);
                        uint8_t rs = (rgb >> 16) & 0x0000ff;
                        uint8_t gs = (rgb >> 8) & 0x0000ff;
                        uint8_t bs = (rgb) & 0x0000ff;

                        float LRefs, aRefs, bRefs;
                        color_transf_omp_.RGB2CIELAB_normalized (rs, gs, bs, LRefs, aRefs, bRefs);

                        model_cloud_gs->points[k].r = LRefs * 255.f;
                        model_cloud_gs->points[k].g = (aRefs + 1.f) / 2.f * 255;
                        model_cloud_gs->points[k].b = (bRefs + 1.f) / 2.f * 255;
                    }

                    pcl::copyPointCloud(*scene_cloud_downsampled_, *scene_cloud);

                    for(size_t k=0; k < scene_cloud->points.size(); k++)
                    {
                        scene_cloud->points[k].r = scene_LAB_values_[k][0] * 255;
                        scene_cloud->points[k].g = (scene_LAB_values_[k][1] + 1.f) / 2.f * 255;
                        scene_cloud->points[k].b = (scene_LAB_values_[k][2] + 1.f) / 2.f * 255;
                    }

                    //pcl::copyPointCloud(*scene_cloud, recognition_models_[i]->explained_, *scene_cloud);
                }
                else if(param_.color_space_ == 2)
                {
                    pcl::copyPointCloud(*recog_model->visible_cloud_, *model_cloud_gs_specified);
                    for(size_t k=0; k < model_cloud_gs_specified->points.size(); k++)
                    {
                        unsigned char c = (recog_model->cloud_GS_[k]) * 255;
                        model_cloud_gs_specified->points[k].r =
                                model_cloud_gs_specified->points[k].g =
                                model_cloud_gs_specified->points[k].b = c;
                    }

                    for(size_t k=0; k < model_cloud_gs->points.size(); k++)
                    {
                        //float gs = (scene_cloud->points[k].r + scene_cloud->points[k].g + scene_cloud->points[k].b) / 3.f;
                        uint32_t rgb = *reinterpret_cast<int*> (&model_cloud_gs->points[k].rgb);
                        uint8_t rs = (rgb >> 16) & 0x0000ff;
                        uint8_t gs = (rgb >> 8) & 0x0000ff;
                        uint8_t bs = (rgb) & 0x0000ff;

                        unsigned int c = (rs + gs + bs) / 3;

                        model_cloud_gs->points[k].r = model_cloud_gs->points[k].g = model_cloud_gs->points[k].b = c;
                    }

                    pcl::copyPointCloud(*scene_cloud_downsampled_, *scene_cloud);

                    for(size_t k=0; k < scene_cloud->points.size(); k++)
                    {
                        unsigned char c = scene_GS_values_[k] * 255;
                        scene_cloud->points[k].r = scene_cloud->points[k].g = scene_cloud->points[k].b = c;
                    }

                    //pcl::copyPointCloud(*scene_cloud, recognition_models_[i]->explained_, *scene_cloud);
                }
                else if(param_.color_space_ == 3)
                {
                    pcl::copyPointCloud(*recog_model->visible_cloud_, *model_cloud_gs_specified);
                    for(size_t k=0; k < model_cloud_gs_specified->points.size(); k++)
                    {
                        model_cloud_gs_specified->points[k].r =
                                model_cloud_gs_specified->points[k].g =
                                model_cloud_gs_specified->points[k].b = recog_model->cloud_LAB_[k][0] * 255;
                    }

                    for(size_t k=0; k < model_cloud_gs->points.size(); k++)
                    {
                        //float gs = (scene_cloud->points[k].r + scene_cloud->points[k].g + scene_cloud->points[k].b) / 3.f;
                        uint32_t rgb = *reinterpret_cast<int*> (&model_cloud_gs->points[k].rgb);
                        uint8_t rs = (rgb >> 16) & 0x0000ff;
                        uint8_t gs = (rgb >> 8) & 0x0000ff;
                        uint8_t bs = (rgb) & 0x0000ff;

                        float LRefs, aRefs, bRefs;
                        color_transf_omp_.RGB2CIELAB_normalized (rs, gs, bs, LRefs, aRefs, bRefs);
                        model_cloud_gs->points[k].r = model_cloud_gs->points[k].g = model_cloud_gs->points[k].b = LRefs * 255.f;
                    }

                    pcl::copyPointCloud(*scene_cloud_downsampled_, *scene_cloud);

                    for(size_t k=0; k < scene_cloud->points.size(); k++)
                    {
                        scene_cloud->points[k].r =
                                scene_cloud->points[k].g =
                                scene_cloud->points[k].b = scene_LAB_values_[k][0] * 255;
                    }
                }


                pcl::visualization::PCLVisualizer vis("TEST");
                int v1,v2, v3, v4;
                vis.createViewPort(0,0,0.25,1,v1);
                vis.createViewPort(0.25,0,0.5,1,v2);
                vis.createViewPort(0.5,0,0.75,1,v3);
                vis.createViewPort(0.75,0,1,1,v4);
                vis.addPointCloud<pcl::PointXYZRGB>(scene_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(scene_cloud), "scene", v1);
                vis.addPointCloud<pcl::PointXYZRGB>(model_cloud_gs, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(model_cloud_gs), "model", v2);
                vis.addPointCloud<pcl::PointXYZRGB>(model_cloud_gs_specified, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(model_cloud_gs_specified), "model_specified", v3);


                if(models_smooth_faces_.size() > i)
                {
                    pcl::PointCloud<pcl::PointXYZL>::Ptr supervoxels_labels_cloud = recognition_models_[i]->visible_labels_;
                    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> handler_labels(supervoxels_labels_cloud, "label");
                    vis.addPointCloud(supervoxels_labels_cloud, handler_labels, "labels_", v4);
                }

                vis.setBackgroundColor(1,1,1);
                vis.spin();
            }
        }
    }

    delete best;

    {
        //check results
        GHVSAModel<ModelT, SceneT> _model;
        clear_structures();
        fill_structures(cc_indices, initial_solution, _model);
    }

    recognition_models_ = recognition_models_copy;

}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename ModelT, typename SceneT>
void GHV<ModelT, SceneT>::verify()
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
    {
        cc_[0][i] = static_cast<int>(i);
    }

    //compute number of visible points
    number_of_visible_points_ = 0;
    for(size_t i=0; i < recognition_models_.size(); i++)
        number_of_visible_points_ += recognition_models_[i]->visible_cloud_->points.size();

    //for each connected component, find the optimal solution
    {
        pcl::ScopeTime t("Optimizing object hypotheses verification cost function");

        for (int c = 0; c < n_cc_; c++)
        {
            //TODO: Check for trivial case...
            //TODO: Check also the number of hypotheses and use exhaustive enumeration if smaller than 10
            std::vector<bool> subsolution (cc_[c].size (), param_.initial_status_);

            SAOptimize (cc_[c], subsolution);


            //ATTENTION: just for the paper to visualize cues!!
            /*if(visualize_go_cues_)
            {

                std::vector<bool> opt_subsolution = subsolution;
                //deactivate two hypotheses and randomly activate two others
                for(size_t k=0; k < 200; k++)
                {

                    srand((unsigned)time(NULL));
                    boost::mt19937 generator(time(0));
                    boost::uniform_01<boost::mt19937> gen(generator);

                    int deactivated = 0;
                    int max_deact = 2;
                    for(size_t i=0; i < subsolution.size(); i++)
                    {
                        if(deactivated >= max_deact)
                            break;

                        if(subsolution[i])
                        {
                            float r = gen();
                            std::cout << "randon number:" << r << std::endl;
                            if(r > 0.9f)
                            {
                                subsolution[i] = false;
                                deactivated++;
                            }
                        }
                    }

                    int act = 0;
                    int max_act = 2;

                    for (int i = 0; (i < max_act); i++)
                    {
                        int to_act = std::floor(gen() * subsolution.size());
                        subsolution[to_act] = true;
                    }

                    //check results
                    GHVSAModel<ModelT, SceneT> model;
                    clear_structures();
                    fill_structures(cc_[c], subsolution, model);

                    visualizeGOCues(subsolution, 0, 0);

                    subsolution = opt_subsolution;
                }
            }*/

            for (size_t i = 0; i < subsolution.size (); i++)
            {
                //mask_[indices_[cc_[c][i]]] = (subsolution[i]);
                mask_[cc_[c][i]] = subsolution[i];
            }
        }
    }
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
GHV<ModelT, SceneT>::computeRGBHistograms (const std::vector<Eigen::Vector3f> & rgb_values, Eigen::MatrixXf & rgb, int dim, float min, float max)
{
    int hist_size = max - min + 1;
    //float size_bin = 1.f / hist_size;
    rgb = Eigen::MatrixXf (hist_size, dim);
    rgb.setZero ();
    for (size_t i = 0; i < dim; i++)
    {
        for (size_t j = 0; j < rgb_values.size (); j++)
        {
            int pos = std::floor (static_cast<float> (rgb_values[j][i] - min) / (max - min) * hist_size);
            if(pos < 0)
                pos = 0;

            if(pos > hist_size)
                pos = hist_size - 1;
            rgb (pos, i)++;
        }
    }
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::specifyHistograms (const std::vector<size_t> &src_hist, const std::vector<size_t> &dst_hist, std::vector<float> & lut)
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
        sum_dst += src_hist[i];


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
    for (size_t k = 0; k < src_hist_cumulative.size(); k++)
    {
        for (int z = last; z < src_hist_cumulative.size(); z++)
        {
            if (src_hist_cumulative[z] - dst_hist_cumulative[k] >= 0)
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
GHV<ModelT, SceneT>::handlingNormals (GHVRecognitionModel<ModelT> &recog_model, size_t i, size_t object_models_size)
{
    if(visible_normal_models_.size() == object_models_size && !param_.use_normals_from_visible_/*&& !is_planar_model*/)
    {
        pcl::PointCloud<pcl::Normal>::ConstPtr model_normals = visible_normal_models_[i];

        recog_model.normals_.reset (new pcl::PointCloud<pcl::Normal> ());
        recog_model.normals_->points.resize(recog_model.visible_cloud_->points.size ());

        //check nans...
        size_t kept = 0;
        for (size_t idx = 0; idx < recog_model.visible_cloud_->points.size (); ++idx)
        {
            if ( pcl::isFinite(recog_model.visible_cloud_->points[idx]) && pcl::isFinite(model_normals->points[idx]) )
            {
                recog_model.visible_cloud_->points[kept] = recog_model.visible_cloud_->points[idx];
                recog_model.normals_->points[kept] = model_normals->points[idx];
                kept++;
            }
        }

        recog_model.visible_cloud_->points.resize (kept);
        recog_model.visible_cloud_->width = kept;
        recog_model.visible_cloud_->height = 1;

        recog_model.normals_->points.resize (kept);
        recog_model.normals_->width = kept;
        recog_model.normals_->height = 1;

        if (recog_model.visible_cloud_->points.empty())
            return false;
    }
    else
    {
        //pcl::ScopeTime t("Computing normals and checking nans");

        size_t kept = 0;
        for (size_t idx = 0; idx < recog_model.visible_cloud_->points.size (); ++idx)
        {
            if ( pcl::isFinite(recog_model.visible_cloud_->points[idx]) )
            {
                recog_model.visible_cloud_->points[kept] = recog_model.visible_cloud_->points[idx];
                kept++;
            }
        }

        recog_model.visible_cloud_->points.resize (kept);
        recog_model.visible_cloud_->width = kept;
        recog_model.visible_cloud_->height = 1;

        if (recog_model.visible_cloud_->points.empty())
        {
            PCL_WARN("The model cloud has no points..\n");
            return false;
        }
            //compute normals unless given (now do it always...)


        computeNormals<ModelT>(recog_model.visible_cloud_, recog_model.normals_, param_.normal_method_);

        kept = 0;
        for (size_t pt = 0; pt < recog_model.normals_->points.size (); pt++)
        {
            if ( pcl::isFinite(recog_model.normals_->points[pt]) )
            {
                recog_model.normals_->points[kept] = recog_model.normals_->points[pt];
                recog_model.visible_cloud_->points[kept] = recog_model.visible_cloud_->points[pt];
                kept++;
            }
        }

        recog_model.normals_->points.resize (kept);
        recog_model.visible_cloud_->points.resize (kept);
        recog_model.visible_cloud_->width = recog_model.normals_->width = kept;
        recog_model.visible_cloud_->height = recog_model.normals_->height = 1;
    }

    return true;
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::specifyColor(size_t id, std::vector<float> & lookup, GHVRecognitionModel<ModelT> & recog_model)
{
    std::vector< std::vector<int> > label_indices;
    std::vector< std::vector<int> > explained_scene_pts_per_label;

    if(models_smooth_faces_.size() > id)
    {
        //use visible indices to check which points are visible
        recog_model.visible_labels_.reset(new pcl::PointCloud<pcl::PointXYZL>);
        pcl::copyPointCloud(*models_smooth_faces_[id], visible_indices_[id], *recog_model.visible_labels_);

        //specify using the smooth faces
        int max_label = 0;
        for(size_t k=0; k < recog_model.visible_labels_->points.size(); k++)
        {
            if( recog_model.visible_labels_->points[k].label > max_label)
                max_label = recog_model.visible_labels_->points[k].label;
        }

        //1) group points based on label
        label_indices.resize(max_label + 1);
        for(size_t k=0; k < recog_model.visible_labels_->points.size(); k++)
            label_indices[ recog_model.visible_labels_->points[k].label ].push_back(k);


        //2) for each group, find corresponding scene points and push them into label_explained_indices_points
        std::vector<std::pair<int, float> > label_index_distances;
        label_index_distances.resize(scene_cloud_downsampled_->points.size(), std::make_pair(-1, std::numeric_limits<float>::infinity()));

        for(size_t j=0; j < label_indices.size(); j++)
        {
            for (size_t i = 0; i < label_indices[j].size (); i++)
            {
                std::vector<int> & nn_indices = recog_model.inlier_indices_[label_indices[j][i]];
                std::vector<float> & nn_distances = recog_model.inlier_distances_[label_indices[j][i]];

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
        label_indices[0].resize(recog_model.visible_cloud_->points.size());
        for(size_t k=0; k < recog_model.visible_cloud_->points.size(); k++)
            label_indices[0][k] = k;

        std::vector<bool> scene_pt_is_explained ( scene_cloud_downsampled_->points.size(), false );
        for (size_t i = 0; i < label_indices[0].size (); i++)
        {
            std::vector<int> & nn_indices = recog_model.inlier_indices_[ label_indices[0][i] ];
//            std::vector<float> & nn_distances = recog_model.inlier_distances_[ label_indices[0][i] ];

            for (size_t k = 0; k < nn_indices.size (); k++)
                scene_pt_is_explained[ nn_indices[k] ] = true;
        }

        std::vector<int> explained_scene_pts = createIndicesFromMask<int>(scene_pt_is_explained);

        explained_scene_pts_per_label.resize(1, std::vector<int>( explained_scene_pts.size() ) );
        for (size_t i = 0; i < explained_scene_pts.size (); i++)
            explained_scene_pts_per_label[0][i] = explained_scene_pts[i];
    }

    recog_model.cloud_LAB_original_ = recog_model.cloud_LAB_;

    for(size_t j=0; j < label_indices.size(); j++)
    {
        std::vector<int> explained_scene_pts = explained_scene_pts_per_label[j];

        if(param_.color_space_ == 0 || param_.color_space_ == 3)
        {
            std::vector<float> model_L_values ( label_indices[j].size () );
            std::vector<float> scene_L_values ( explained_scene_pts.size() );

            for (size_t i = 0; i < label_indices[j].size (); i++)
                model_L_values[i] = recog_model.cloud_LAB_[ label_indices[j][i] ][0];

            for(size_t i=0; i<explained_scene_pts.size(); i++)
                scene_L_values[i] = scene_LAB_values_[ explained_scene_pts[i] ][0];

            size_t hist_size = 100;
            std::vector<size_t> model_L_hist, scene_L_hist;
            computeHistogram(model_L_values, model_L_hist, hist_size, -1.f, 1.f);
            computeHistogram(scene_L_values, scene_L_hist, hist_size, -1.f, 1.f);
            specifyHistograms(scene_L_hist, model_L_hist, lookup);

            for (size_t i = 0; i < label_indices[j].size(); i++)
            {
                float LRefm = recog_model.cloud_LAB_[ label_indices[j][i] ][0];
                int pos = std::floor (hist_size * static_cast<float> (LRefm + 1.f) / (1.f - -1.f));

                if(pos < lookup.size())
                    throw std::runtime_error("Color not specified");

                LRefm = lookup[pos] * (1.f - -1.f)/hist_size - 1.f;
                recog_model.cloud_LAB_[ label_indices[j][i] ][0] = LRefm;
                recog_model.cloud_indices_specified_.push_back(label_indices[j][i]);
            }
        }
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
bool
GHV<ModelT, SceneT>::addModel (size_t model_id, GHVRecognitionModel<ModelT> &recog_model)
{
    bool is_planar_model = false;
    std::map<size_t, size_t>::iterator it1;
    it1 = model_to_planar_model_.find( model_id );
    if(it1 != model_to_planar_model_.end())
        is_planar_model = true;


    recog_model.visible_cloud_.reset (new pcl::PointCloud<ModelT> (*visible_models_[model_id]));
    recog_model.complete_cloud_.reset (new pcl::PointCloud<ModelT> (*complete_models_[model_id]));

//    recog_model.visible_cloud_ = visible_models_[model_id];
//    recog_model.complete_cloud_ = complete_models_[model_id];

    size_t object_models_size = complete_models_.size() - planar_models_.size();
    float extra_weight = 1.f;
    if( extra_weights_.size() == object_models_size)
        extra_weight = extra_weights_[model_id];

    if( object_ids_.size() == complete_models_.size() )
        recog_model.id_s_ = object_ids_[model_id];

    if( ! handlingNormals(recog_model, model_id, complete_models_.size() ) )
    {
        PCL_WARN("Handling normals returned false\n");
        return false;
    }

    //pcl::ScopeTime tt_nn("Computing outliers and explained points...");
    std::vector<int> explained_indices;
    std::vector<float> outliers_weight;
    std::vector<float> explained_indices_distances;

    //which point first from the scene is explained by a point j_k with dist d_k from the model
    std::map<int, boost::shared_ptr<std::vector<std::pair<int, float> > > > model_explains_scene_points;
    std::map<int, boost::shared_ptr<std::vector<std::pair<int, float> > > > model_explains_scene_points_color_weight;
    std::map<int, boost::shared_ptr<std::vector<std::pair<int, float> > > >::iterator it;

    outliers_weight.resize (recog_model.visible_cloud_->points.size ());
    recog_model.outlier_indices_.resize (recog_model.visible_cloud_->points.size ());
    recog_model.outliers_3d_indices_.resize (recog_model.visible_cloud_->points.size ());
    recog_model.color_outliers_indices_.resize (recog_model.visible_cloud_->points.size ());
    recog_model.scene_point_explained_by_hypothesis_.resize(scene_cloud_downsampled_->points.size(), false);

    if(!is_planar_model && !param_.ignore_color_even_if_exists_)
    {
        //compute cloud LAB values for model visible points
        recog_model.cloud_LAB_.resize(recog_model.visible_cloud_->points.size());
        recog_model.cloud_RGB_.resize(recog_model.visible_cloud_->points.size());
        recog_model.cloud_GS_.resize(recog_model.visible_cloud_->points.size());

        #pragma omp parallel for schedule (dynamic)
        for(size_t j=0; j < recog_model.cloud_LAB_.size(); j++)
        {
            bool exists_m;
            float rgb_m;
            pcl::for_each_type<FieldListM> (
                        pcl::CopyIfFieldExists<typename CloudM::PointType, float> (
                            recog_model.visible_cloud_->points[j],
                            "rgb", exists_m, rgb_m));

            uint32_t rgb = *reinterpret_cast<int*> (&rgb_m);
            unsigned char rm = (rgb >> 16) & 0x0000ff;
            unsigned char gm = (rgb >> 8) & 0x0000ff;
            unsigned char bm = (rgb) & 0x0000ff;

            float LRefm, aRefm, bRefm;
            color_transf_omp_.RGB2CIELAB_normalized (rm, gm, bm, LRefm, aRefm, bRefm);

            float rmf,gmf,bmf;
            rmf = static_cast<float>(rm) / 255.f;
            gmf = static_cast<float>(gm) / 255.f;
            bmf = static_cast<float>(bm) / 255.f;
            recog_model.cloud_LAB_[j] = Eigen::Vector3f(LRefm, aRefm, bRefm);
            recog_model.cloud_RGB_[j] = Eigen::Vector3f(rmf, gmf, bmf);
            recog_model.cloud_GS_[j] = (rmf + gmf + bmf) / 3.f;
        }
    }

    recog_model.inlier_indices_.resize(recog_model.visible_cloud_->points.size ());
    recog_model.inlier_distances_.resize(recog_model.visible_cloud_->points.size ());

#pragma omp parallel for schedule(dynamic)
    for (size_t pt = 0; pt < recog_model.visible_cloud_->points.size (); pt++)
        octree_scene_downsampled_->radiusSearch (recog_model.visible_cloud_->points[pt], param_.inliers_threshold_,
                                                 recog_model.inlier_indices_[pt], recog_model.inlier_distances_[pt],
                                                 std::numeric_limits<int>::max ());

    std::vector<float> lookup;
    if(!is_planar_model && !param_.ignore_color_even_if_exists_ && param_.use_histogram_specification_)
        specifyColor(model_id, lookup, recog_model);

    float inliers_gaussian = 2 * param_.inliers_threshold_ * param_.inliers_threshold_;
    float inliers_gaussian_soft = 2 * (param_.inliers_threshold_ + param_.resolution_) * (param_.inliers_threshold_ + param_.resolution_);

    size_t o = 0;
    size_t o_color = 0;
    size_t o_3d = 0;

    //Goes through the visible model points and finds scene points within a radius neighborhood
    //If in this neighborhood, there are no scene points, model point is considered outlier
    //If there are scene points, the model point is associated with the scene point, together with its distance
    //A scene point might end up being explained by multiple model points

    size_t bad_normals = 0;
    float sigma = 2.f * param_.color_sigma_ab_ * param_.color_sigma_ab_;
    float sigma_y = 2.f * param_.color_sigma_l_ * param_.color_sigma_l_;
    Eigen::Vector3f color_m, color_s;

    for (size_t i = 0; i < recog_model.visible_cloud_->points.size (); i++)
    {
        bool outlier = false;
        int outlier_type = 0;

        std::vector<int> & nn_indices = recog_model.inlier_indices_[i];
        std::vector<float> & nn_distances = recog_model.inlier_distances_[i];

        if( nn_indices.empty() ) // if there is no scene point nearby , count it as an outlier
        {
            recog_model.outliers_3d_indices_[o_3d] = i;
            outlier = true;
            o_3d++;
            outlier_type = 0;
        }
        else
        {
            std::vector<float> weights;
            if (!is_planar_model)
            {
                weights.resize( nn_distances.size () );
                float color_weight = 1.f;

                for (size_t k = 0; k < nn_distances.size () && !is_planar_model; k++)
                {
                    if (!param_.ignore_color_even_if_exists_ )
                    {
                        if(param_.color_space_ == 0 || param_.color_space_ == 5 || param_.color_space_ == 6)
                        {
                            color_m = recog_model.cloud_LAB_[i];
                            color_s = scene_LAB_values_[nn_indices[k]];

                            color_weight = std::exp ( -0.5f * (    (color_m[0] - color_s[0]) * (color_m[0] - color_s[0]) / sigma_y
                                                               +(  (color_m[1] - color_s[1]) * (color_m[1] - color_s[1])
                                                                  +(color_m[2] - color_s[2]) * (color_m[2] - color_s[2]) ) / sigma ) );
                        }
                        else if(param_.color_space_ == 1)
                        {
                            color_m = recog_model.cloud_RGB_[i];
                            color_s = scene_RGB_values_[nn_indices[k]];

                            color_weight = std::exp (-0.5f * (   (color_m[0] - color_s[0]) * (color_m[0] - color_s[0])
                                                               + (color_m[1] - color_s[1]) * (color_m[1] - color_s[1])
                                                               + (color_m[2] - color_s[2]) * (color_m[2] - color_s[2])) / sigma);
                        }
                        else if(param_.color_space_ == 2)
                        {
                            float yuvm = recog_model.cloud_GS_[i];
                            float yuvs = scene_GS_values_[nn_indices[k]];

                            color_weight = std::exp (-0.5f * (yuvm - yuvs) * (yuvm - yuvs) / sigma_y);
                            //color_weight = 1.f - (std::abs(yuvm - yuvs));
                        }
                        else if(param_.color_space_ == 3)
                        {
                            float yuvm = recog_model.cloud_LAB_[i][0];
                            float yuvs = scene_LAB_values_[nn_indices[k]][0];

                            color_weight = std::exp (-0.5f * (yuvm - yuvs) * (yuvm - yuvs) / sigma_y);
                            //color_weight = 1.f - (std::abs(yuvm - yuvs));

                        }
                    }

                    const float dist_weight = std::exp( -nn_distances[k] / inliers_gaussian_soft );

                    //float d_weight = std::exp(-( d / (inliers_threshold_ * 3.f)));
                    /*float d = nn_distances[k];
                    float d_weight = -(d / (inliers_threshold_)) + 1;
                    color_weight *= d_weight;*/

                    //scene_LAB_values.push_back(Eigen::Vector3f(yuvs));

                    //weights.push_back(color_weight);
                    weights[k] = color_weight * dist_weight;
                }
            }

            std::sort(weights.begin(), weights.end(), std::greater<float>());

            if(is_planar_model || param_.ignore_color_even_if_exists_ || weights[0] > param_.best_color_weight_) //best weight is not an outlier
            {
                for (size_t k = 0; k < nn_distances.size (); k++)
                {
                    std::pair<int, float> pair = std::make_pair (i, nn_distances[k]); //i is a index to a model point and then distance
                    //nn_distances is squared!!

                    it = model_explains_scene_points.find (nn_indices[k]);
                    if (it == model_explains_scene_points.end ())
                    {
                        boost::shared_ptr<std::vector<std::pair<int, float> > > vec (new std::vector<std::pair<int, float> > ());
                        vec->push_back (pair);
                        model_explains_scene_points[nn_indices[k]] = vec;
                    }
                    else
                    {
                        it->second->push_back (pair);
                    }

                    if(!is_planar_model && !param_.ignore_color_even_if_exists_)
                    {
                        //std::pair<int, float> pair_color = std::make_pair (i, weights_not_sorted[k]);
                        std::pair<int, float> pair_color = std::make_pair (i, weights[0]);
                        it = model_explains_scene_points_color_weight.find (nn_indices[k]);
                        if (it == model_explains_scene_points_color_weight.end ())
                        {
                            boost::shared_ptr<std::vector<std::pair<int, float> > > vec (new std::vector<std::pair<int, float> > ());
                            vec->push_back (pair_color);
                            model_explains_scene_points_color_weight[nn_indices[k]] = vec;
                        }
                        else
                        {
                            it->second->push_back (pair_color);
                        }
                    }
                }
            }
            else
            {
                recog_model.color_outliers_indices_[o_color] = i;
                outlier = true;
                o_color++;
                outlier_type = 1;
            }
        }

        if(outlier)
        {
            //weight outliers based on noise model
            //model points close to occlusion edges or with perpendicular normals
            float d_weight = 1.f;
            //std::cout << "is an outlier" << is_planar_model << " " << occ_edges_available_ << std::endl;

            if(!is_planar_model)
            {
                //std::cout << "going to weight based on normals..." << std::endl;
                Eigen::Vector3f normal_p = recog_model.normals_->points[i].getNormalVector3fMap();
                Eigen::Vector3f normal_vp = Eigen::Vector3f::UnitZ() * -1.f;
                normal_p.normalize ();
                normal_vp.normalize ();

                float dot = normal_vp.dot(normal_p);
                float angle = pcl::rad2deg(acos(dot));
                if (angle > 60.f)
                {
                    if(outlier_type == 1)
                        o_color--;
                    else
                        o_3d--;

                    // [60,75) => 0.5
                    // [75,90) => 0.25
                    // >90 => 0

                    /*if(angle >= 90.f)
                        d_weight = 0.25f;
                    else*/
                    d_weight = param_.d_weight_for_bad_normals_;

                    bad_normals++;  // is this due to noise of the normal estimation for points that "look away" from the camera?
                }
            }

            outliers_weight[o] = param_.regularizer_ * d_weight;
            recog_model.outlier_indices_[o] = i;
            o++;
        }
    }

    outliers_weight.resize (o);
    recog_model.outlier_indices_.resize (o);
    recog_model.outliers_3d_indices_.resize (o_3d);
    recog_model.color_outliers_indices_.resize (o_color);
    //std::cout << "outliers with bad normals for sensor:" << bad_normals << std::endl;

    if( o == 0 )
        recog_model.outliers_weight_ = 1.f;
    else  {
        if (param_.outliers_weight_computation_method_ == 0) {   // use mean
            recog_model.outliers_weight_ = (std::accumulate (outliers_weight.begin (), outliers_weight.end (), 0.f) / static_cast<float> (outliers_weight.size ()));
        }
        else { // use median
            std::sort(outliers_weight.begin(), outliers_weight.end());
            recog_model.outliers_weight_ = outliers_weight[outliers_weight.size() / 2.f];
        }
    }

    //float inliers_gaussian = 2 * std::pow(inliers_threshold_ + resolution_, 2);
    for (it = model_explains_scene_points.begin (); it != model_explains_scene_points.end (); ++it)
    {
        //ATTENTION, TODO => use normal information to select closest!

        Eigen::Vector3f scene_p_normal = scene_normals_->points[it->first].getNormalVector3fMap ();
        scene_p_normal.normalize();

        bool use_normal_info_for_closest = false;
        size_t closest = 0;
        float min_d = std::numeric_limits<float>::infinity ();
        for (size_t i = 0; i < it->second->size (); i++)
        {
            if(use_normal_info_for_closest)
            {
                Eigen::Vector3f model_p_normal;
                if(param_.use_normals_from_visible_ && (visible_normal_models_.size() == complete_models_.size()))
                {
                    model_p_normal = recog_model.normals_from_visible_->points[it->second->at (i).first].getNormalVector3fMap ();
                }
                else
                {
                    model_p_normal = recog_model.normals_->points[it->second->at (i).first].getNormalVector3fMap ();
                }
                model_p_normal.normalize();

//                float d = it->second->at (i).second;
                //float d_weight = std::exp( -(d / inliers_gaussian));
                //float dotp = scene_p_normal.dot (model_p_normal); //[-1,1] from antiparallel trough perpendicular to parallel
                //float w = d_weight * dotp;

            }
            else
            {
                if (it->second->at (i).second < min_d)
                {
                    min_d = it->second->at (i).second;
                    closest = i;
                }
            }
        }

        float d = it->second->at (closest).second;
        float d_weight = std::exp( -(d / inliers_gaussian));

        //it->first is index to scene point
        //using normals to weight inliers
        Eigen::Vector3f model_p_normal;

        if(param_.use_normals_from_visible_ && (visible_normal_models_.size() == complete_models_.size()))
        {
            model_p_normal = recog_model.normals_from_visible_->points[it->second->at (closest).first].getNormalVector3fMap ();
        }
        else
        {
            model_p_normal = recog_model.normals_->points[it->second->at (closest).first].getNormalVector3fMap ();
        }
        model_p_normal.normalize();

        bool use_dot_ = false;
        float dotp = scene_p_normal.dot (model_p_normal); //[-1,1] from antiparallel trough perpendicular to parallel

        if(use_dot_)
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

        //else
        //dotp = 1.f; //ATTENTION: Deactivated normal weight!

        /*if(model_p_normal.norm() > 1)
        {
            std::cout << "model_p normal:" << model_p_normal.norm() << std::endl;
        }
        assert(model_p_normal.norm() <= 1);
        assert(scene_p_normal.norm() <= 1);
        assert(std::abs(dotp) <= 1);
        assert(d_weight <= 1);
        assert(extra_weight <= 1);*/

        if (!is_planar_model && !param_.ignore_color_even_if_exists_)
        {
            std::map<int, boost::shared_ptr<std::vector<std::pair<int, float> > > >::iterator it_color;
            it_color = model_explains_scene_points_color_weight.find(it->first);
            if(it != model_explains_scene_points_color_weight.end())
                d_weight *= it_color->second->at(closest).second;


            /*float rgb_s;
            bool exists_s;

            typedef pcl::PointCloud<SceneT> CloudS;
            typedef typename pcl::traits::fieldList<typename CloudS::PointType>::type FieldListS;

            pcl::for_each_type<FieldListS> (
                        pcl::CopyIfFieldExists<typename CloudS::PointType, float> (scene_cloud_downsampled_->points[it->first],
                                                                                   "rgb", exists_s, rgb_s));

            if(exists_s)
            {
                uint32_t rgb = *reinterpret_cast<int*> (&rgb_s);
                unsigned char rs = (rgb >> 16) & 0x0000ff;
                unsigned char gs = (rgb >> 8) & 0x0000ff;
                unsigned char bs = (rgb) & 0x0000ff;

                float LRefs, aRefs, bRefs;
                RGB2CIELAB (rs, gs, bs, LRefs, aRefs, bRefs);
                LRefs /= 100.0f; aRefs /= 120.0f; bRefs /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

                scene_LAB_values.push_back(Eigen::Vector3f( LRefs, (aRefs + 1) / 2.f, (bRefs + 1) / 2.f));
            }*/
        }

        if( (d_weight * dotp * extra_weight > 1.f) || pcl_isnan(d_weight * dotp * extra_weight) || pcl_isinf(d_weight * dotp * extra_weight))
        {
            std::cout << d_weight * dotp * extra_weight << std::endl;
        }

        assert((d_weight * dotp * extra_weight) <= 1.0001f);
        explained_indices.push_back (it->first);
        explained_indices_distances.push_back (d_weight * dotp * extra_weight);
        recog_model.scene_point_explained_by_hypothesis_[it->first] = true; //this scene point is explained by this hypothesis
    }

//    recog_model.bad_information_ =  static_cast<int> (recog_model.outlier_indices_.size ());

    //compute the amount of information for explained scene points (color)
    float mean = std::accumulate(explained_indices_distances.begin(), explained_indices_distances.end(), 0.f) / static_cast<float>(explained_indices_distances.size());
    if(explained_indices.size() == 0)
    {
        mean = 0.f;
    }

    assert(mean <= 1.f);
    recog_model.mean_ = mean;

    //modify the explained weights for planar models if color is being used
    //ATTENTION: check this... looks weird!
    it1 = model_to_planar_model_.find( model_id );
    if(it1 != model_to_planar_model_.end())
    {
         //Plane found... decrease weight
        for(size_t k=0; k < explained_indices_distances.size(); k++)
        {
            explained_indices_distances[k] *= param_.best_color_weight_;

            if (!param_.ignore_color_even_if_exists_)
                explained_indices_distances[k] /= 2;
        }
    }

    recog_model.hyp_penalty_ = 0; //ATTENTION!

    recog_model.explained_ = explained_indices;
    recog_model.explained_distances_ = explained_indices_distances;
    recog_model.id_ = model_id;
    //std::cout << "Model:" << recog_model.complete_cloud_->points.size() << " " << recog_model.visible_cloud_->points.size() << std::endl;
    return true;
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::computeClutterCueAtOnce ()
{
    //compute all scene points that are explained by the hypothesis
    std::vector<bool> scene_pt_is_explained (scene_cloud_downsampled_->size(), false);

    for (size_t j = 0; j < recognition_models_.size (); j++)
    {
        for (size_t i = 0; i < recognition_models_[j]->explained_.size (); i++)
            scene_pt_is_explained [ recognition_models_[j]->explained_[i] ] = true;
    }

    std::vector<int> explained_points_vec = createIndicesFromMask<int>(scene_pt_is_explained);

    std::vector<int> scene_to_unique( scene_cloud_downsampled_->size(), -1 );
    for(size_t i=0; i < explained_points_vec.size(); i++)
        scene_to_unique[explained_points_vec[i]] = i;

    float rn_sqr = param_.radius_neighborhood_clutter_ * param_.radius_neighborhood_clutter_;

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

    #pragma omp parallel for schedule(dynamic)
    for (size_t j = 0; j < recognition_models_.size (); j++)
    {
        std::vector< std::pair<int, float> > unexplained_points_per_model;
        std::pair<int, float> def_value = std::make_pair(-1, std::numeric_limits<float>::infinity());
        unexplained_points_per_model.resize(scene_cloud_downsampled_->points.size(), def_value);

        boost::shared_ptr<GHVRecognitionModel<ModelT> > recog_model = recognition_models_[j];

        const size_t model_id = recog_model->id_;
        bool is_planar_model = false;
        std::map<size_t, size_t>::iterator it1;
        it1 = model_to_planar_model_.find(model_id);
        if( it1 != model_to_planar_model_.end() )
            is_planar_model = true;

        for (size_t i = 0; i < recog_model->explained_.size (); i++)
        {
            const int s_id_exp = recog_model->explained_[i];
            const int idx_to_unique = scene_to_unique[s_id_exp];

            for (size_t k = 0; k < nn_indices_all_points[idx_to_unique].size (); k++)
            {
                const int sidx = nn_indices_all_points[idx_to_unique][k]; //in the neighborhood of an explained point (idx_to_ep)
                if(recog_model->scene_point_explained_by_hypothesis_[sidx])
                    continue;

                assert(recog_model->scene_point_explained_by_hypothesis_[recog_model->explained_[i]]);
                assert(sidx != recog_model->explained_[i]);

                const float d = (scene_cloud_downsampled_->points[recog_model->explained_[i]].getVector3fMap ()
                           - scene_cloud_downsampled_->points[sidx].getVector3fMap ()).squaredNorm ();

                //float curvature = scene_curvature_[s_id_exp];
                //std::cout << "curvature:" << curvature << std::endl;

                if( d < unexplained_points_per_model[sidx].second )
                {
                    //there is an explained point which is closer to this unexplained point
                    unexplained_points_per_model[sidx].second = d;
                    unexplained_points_per_model[sidx].first = s_id_exp;
                }
            }
        }

        recog_model->unexplained_in_neighborhood.resize (scene_cloud_downsampled_->points.size ());
        recog_model->unexplained_in_neighborhood_weights.resize (scene_cloud_downsampled_->points.size ());

        const float clutter_gaussian = 2 * rn_sqr;
        size_t p=0;
        for(size_t i=0; i < unexplained_points_per_model.size(); i++)
        {
            int sidx = unexplained_points_per_model[i].first;
            if(sidx < 0)
                continue;

            //sidx is the closest explained point to the unexplained point

            assert(recog_model->scene_point_explained_by_hypothesis_[sidx]);
            assert(!recog_model->scene_point_explained_by_hypothesis_[i]);

            //point i is unexplained and in the neighborhood of sidx (explained point)
            recog_model->unexplained_in_neighborhood[p] = i;

            const float d = unexplained_points_per_model[i].second;
            float d_weight;
            if( param_.use_clutter_exp_ )
                d_weight = std::exp( -(d / clutter_gaussian));
            else
                d_weight = -(d / rn_sqr) + 1; //points that are close have a strong weight

            //using normals to weight clutter points
            const Eigen::Vector3f & scene_p_normal = scene_normals_->points[sidx].getNormalVector3fMap ();
            const Eigen::Vector3f & model_p_normal = scene_normals_->points[i].getNormalVector3fMap ();
            float dotp = scene_p_normal.dot (model_p_normal); //[-1,1] from antiparallel trough perpendicular to parallel

            if (dotp < 0)
                dotp = 0.f;

            float w = d_weight * dotp;

            float curvature = scene_curvature_[sidx];

            if ( (clusters_cloud_->points[i].label != 0 || param_.use_super_voxels_) &&
                 (clusters_cloud_->points[i].label == clusters_cloud_->points[sidx].label)
                 && !is_planar_model
                 && curvature < 0.015)
                 /*&& (curvature < 0.01f))*/
                //&& penalize_with_smooth_segmentation[clusters_cloud_->points[i].label]) //ATTENTION!
            {
                w = 1.f; //ATTENTION!
                assert(clusters_cloud_->points[i].label != 0 || param_.use_super_voxels_);
                recog_model->unexplained_in_neighborhood_weights[p] = param_.clutter_regularizer_ * w;
            }
            else
                recog_model->unexplained_in_neighborhood_weights[p] = w;

            p++;
        }

        recog_model->unexplained_in_neighborhood_weights.resize (p);
        recog_model->unexplained_in_neighborhood.resize (p);
    }
}


//######### VISUALIZATION FUNCTIONS #####################

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::getOutliersForAcceptedModels(std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud)
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
                                                            std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud_3d)
{
    for(size_t i=0; i < recognition_models_.size(); i++)
    {
        if(mask_[i])
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_points (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*(recognition_models_[i]->visible_cloud_), recognition_models_[i]->color_outliers_indices_, *outlier_points);
            outliers_cloud_color.push_back(outlier_points);
            pcl::copyPointCloud(*(recognition_models_[i]->visible_cloud_), recognition_models_[i]->outliers_3d_indices_, *outlier_points);
            outliers_cloud_3d.push_back(outlier_points);
        }
    }
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::visualizeGOCues (const std::vector<bool> & active_solution, float cost, int times_evaluated) const
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
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr smooth_cloud_ =  getSmoothClustersRGBCloud();
    if(smooth_cloud_)
    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> random_handler (smooth_cloud_);
        vis_go_cues_->addPointCloud<pcl::PointXYZRGBA> (smooth_cloud_, random_handler, "smooth_cloud", viewport_smooth_seg_);
    }

    //display active hypotheses
    for(size_t i=0; i < active_solution.size(); i++)
    {
        if(active_solution[i])
        {
            //complete models

            std::stringstream m;
            m << "model_" << i;

            if(poses_ply_.size() == 0)
            {
                pcl::visualization::PointCloudColorHandlerCustom<ModelT> handler_model (complete_models_[i], 0, 255, 0);
                vis_go_cues_->addPointCloud<ModelT> (complete_models_[i], handler_model, m.str(), viewport_scene_and_hypotheses_);
            }
            else
            {
                size_t model_id = i;
                bool is_planar_model = false;
                std::map<size_t, size_t>::const_iterator it1;
                it1 = model_to_planar_model_.find(model_id);
                if(it1 != model_to_planar_model_.end())
                    is_planar_model = true;

                if(!is_planar_model)
                    vis_go_cues_->addModelFromPLYFile (ply_paths_[i], poses_ply_[i], m.str (), viewport_scene_and_hypotheses_);
                else
                    vis_go_cues_->addPolygonMesh (*(planar_models_[it1->second].convex_hull_), m.str(), viewport_scene_and_hypotheses_);
            }

            //model inliers and outliers
            std::stringstream cluster_name;
            cluster_name << "visible" << i;

            typename pcl::PointCloud<ModelT>::Ptr outlier_points (new pcl::PointCloud<ModelT> ());
            for (size_t j = 0; j < recognition_models_[i]->outlier_indices_.size (); j++)
            {
                ModelT c_point;
                c_point.getVector3fMap () = recognition_models_[i]->visible_cloud_->points[recognition_models_[i]->outlier_indices_[j]].getVector3fMap ();
                outlier_points->push_back (c_point);
            }

            pcl::visualization::PointCloudColorHandlerCustom<ModelT> random_handler (recognition_models_[i]->visible_cloud_, 255, 90, 0);
            vis_go_cues_->addPointCloud<ModelT> (recognition_models_[i]->visible_cloud_, random_handler, cluster_name.str (), viewport_model_cues_);

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
    for (size_t j = 0; j < unexplained_by_RM_neighboorhods.size (); j++)
    {
        if(unexplained_by_RM_neighboorhods[j] >= (param_.clutter_regularizer_ - 0.01f) && explained_by_RM_[j] == 0 && (clusters_cloud_->points[j].label != 0 || param_.use_super_voxels_))
        {
            SceneT c_point;
            c_point.getVector3fMap () = scene_cloud_downsampled_->points[j].getVector3fMap ();
            clutter_smooth->push_back (c_point);
        }
        else if (unexplained_by_RM_neighboorhods[j] > 0 && explained_by_RM_[j] == 0)
        {
            pcl::PointXYZRGB c_point;
            c_point.getVector3fMap () = scene_cloud_downsampled_->points[j].getVector3fMap ();

            if(show_weights_with_color_fading_)
            {
                c_point.r = round(255.0 * unexplained_by_RM_neighboorhods[j]);
                c_point.g = 40;
                c_point.b = round(255.0 * unexplained_by_RM_neighboorhods[j]);
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
            c_point.b = 100 + explained_by_RM_distance_weighted[j] * 155;
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
            float curv_weight = getCurvWeight(scene_curvature_[j]);

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
                {
                    c_point.r = 0;
                    c_point.g = 0;
                    c_point.b = 0;
                }
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
