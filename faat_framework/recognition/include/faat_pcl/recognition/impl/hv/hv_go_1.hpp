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

#include <pcl/visualization/pcl_visualizer.h>
#include <faat_pcl/recognition/hv/hv_go_1.h>
//#include <faat_pcl/recognition/hv/cuda/hv_go_1_cuda_wrapper.h>
#include <functional>
#include <numeric>
#include <pcl/common/time.h>
#include <boost/graph/connected_components.hpp>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/common/angles.h>

template<typename ModelT, typename SceneT>
  mets::gol_type
  faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::evaluateSolution (const std::vector<bool> & active, int changed)
  {
    boost::posix_time::ptime start_time (boost::posix_time::microsec_clock::local_time ());
    float sign = 1.f;
    //update explained_by_RM
    if (active[changed])
    {
      //it has been activated
      updateExplainedVector (recognition_models_[changed]->explained_, recognition_models_[changed]->explained_distances_, explained_by_RM_,
                             explained_by_RM_distance_weighted, 1.f);

      if(detect_clutter_) {
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
                             explained_by_RM_distance_weighted, -1.f);

      if(detect_clutter_) {
        updateUnexplainedVector (recognition_models_[changed]->unexplained_in_neighborhood,
                                 recognition_models_[changed]->unexplained_in_neighborhood_weights, unexplained_by_RM_neighboorhods,
                                 recognition_models_[changed]->explained_, explained_by_RM_, -1.f);
      }
      updateCMDuplicity (recognition_models_[changed]->complete_cloud_occupancy_indices_, complete_cloud_occupancy_by_RM_, -1.f);
      sign = -1.f;
    }

    float duplicity = static_cast<float> (getDuplicity ());
    //duplicity = 0.f; //ATTENTION!!
    float good_info = getExplainedValue ();

    float unexplained_info = getPreviousUnexplainedValue ();
    if(!detect_clutter_) {
      unexplained_info = 0;
    }

    float bad_info = static_cast<float> (getPreviousBadInfo ()) + (recognition_models_[changed]->outliers_weight_
        * static_cast<float> (recognition_models_[changed]->bad_information_)) * sign;

    setPreviousBadInfo (bad_info);

    float duplicity_cm = static_cast<float> (getDuplicityCM ()) * w_occupied_multiple_cm_;
    //float duplicity_cm = 0;

    float under_table = 0;
    if (model_constraints_.size () > 0)
    {
      under_table = getModelConstraintsValueForActiveSolution (active);
    }

    boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time ();
    //std::cout << (end_time - start_time).total_microseconds () << " microsecs" << std::endl;
    float cost = (good_info - bad_info - duplicity - unexplained_info - under_table - duplicity_cm - countActiveHypotheses (active)) * -1.f;
    if(cost_logger_) {
      cost_logger_->increaseEvaluated();
      cost_logger_->addCostEachTimeEvaluated(cost);
    }

    //ntimes_evaluated_++;
    return static_cast<mets::gol_type> (cost); //return the dual to our max problem
  }


template<typename ModelT, typename SceneT>
  float
  faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::countActiveHypotheses (const std::vector<bool> & sol)
{
  float c = 0;
  for (size_t i = 0; i < sol.size (); i++)
  {
    if (sol[i]) {
      //c++;
      c += static_cast<float>(recognition_models_[i]->explained_.size()) * active_hyp_penalty_ + min_contribution_;
    }
  }

  return c;
  //return static_cast<float> (c) * active_hyp_penalty_;
}

template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::addPlanarModels(std::vector<faat_pcl::PlaneModel<ModelT> > & models)
{
  planar_models_ = models;
  model_to_planar_model_.clear();
  //iterate through the planar models and append them to complete_models_?
  for(size_t i=0; i < planar_models_.size(); i++)
  {
    model_to_planar_model_[static_cast<int>(visible_models_.size())] = static_cast<int>(i);
    complete_models_.push_back(planar_models_[i].plane_cloud_);
    visible_models_.push_back(planar_models_[i].plane_cloud_);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename ModelT, typename SceneT>
  void
  faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::initialize ()
  {
    //clear stuff
    recognition_models_.clear ();
    unexplained_by_RM_neighboorhods.clear ();
    explained_by_RM_distance_weighted.clear ();
    explained_by_RM_.clear ();
    mask_.clear ();
    //indices_.clear ();
    complete_cloud_occupancy_by_RM_.clear ();

    // initialize mask to false
    mask_.resize (complete_models_.size ());
    for (size_t i = 0; i < complete_models_.size (); i++)
      mask_[i] = false;

    //indices_.resize (complete_models_.size ());

    {
        pcl::ScopeTime t("compute scene normals");
        NormalEstimator_ n3d;
        scene_normals_.reset (new pcl::PointCloud<pcl::Normal> ());

        int j = 0;
        for (size_t i = 0; i < scene_cloud_downsampled_->points.size (); ++i) {
          if (!pcl_isfinite (scene_cloud_downsampled_->points[i].x) || !pcl_isfinite (scene_cloud_downsampled_->points[i].y)
              || !pcl_isfinite (scene_cloud_downsampled_->points[i].z)) {
            //std::cout << "Not finite..." << std::endl;
            continue;
          }

          scene_cloud_downsampled_->points[j] = scene_cloud_downsampled_->points[i];

          j++;
        }

        scene_cloud_downsampled_->points.resize(j);
        scene_cloud_downsampled_->width = j;
        scene_cloud_downsampled_->height = 1;

        std::cout << "scene points after removing NaNs:" << scene_cloud_downsampled_->points.size() << std::endl;

        typename pcl::search::KdTree<SceneT>::Ptr normals_tree (new pcl::search::KdTree<SceneT>);
        normals_tree->setInputCloud (scene_cloud_downsampled_);

        n3d.setRadiusSearch (radius_normals_);
        n3d.setSearchMethod (normals_tree);
        n3d.setInputCloud (scene_cloud_downsampled_);
        n3d.compute (*scene_normals_);

        //check nans...
        j = 0;
        for (size_t i = 0; i < scene_normals_->points.size (); ++i)
        {
          if (!pcl_isfinite (scene_normals_->points[i].normal_x) || !pcl_isfinite (scene_normals_->points[i].normal_y)
              || !pcl_isfinite (scene_normals_->points[i].normal_z))
            continue;

          scene_normals_->points[j] = scene_normals_->points[i];
          scene_cloud_downsampled_->points[j] = scene_cloud_downsampled_->points[i];

          j++;
        }

        scene_normals_->points.resize (j);
        scene_normals_->width = j;
        scene_normals_->height = 1;

        scene_cloud_downsampled_->points.resize (j);
        scene_cloud_downsampled_->width = j;
        scene_cloud_downsampled_->height = 1;
    }

    explained_by_RM_.resize (scene_cloud_downsampled_->points.size (), 0);
    explained_by_RM_distance_weighted.resize (scene_cloud_downsampled_->points.size (), 0.f);
    unexplained_by_RM_neighboorhods.resize (scene_cloud_downsampled_->points.size (), 0.f);

    octree_scene_downsampled_.reset(new pcl::octree::OctreePointCloudSearch<SceneT>(0.01f));
    octree_scene_downsampled_->setInputCloud(scene_cloud_downsampled_);
    octree_scene_downsampled_->addPointsFromInputCloud();

    if(occ_edges_available_)
    {
        octree_occ_edges_.reset(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(0.01f));
        octree_occ_edges_->setInputCloud (occ_edges_);
        octree_occ_edges_->addPointsFromInputCloud ();
    }

    //compute segmentation of the scene if detect_clutter_
    if (detect_clutter_)
    {
      pcl::ScopeTime t("Smooth segmentation of the scene");
      //initialize kdtree for search

      scene_downsampled_tree_.reset (new pcl::search::KdTree<SceneT>);
      scene_downsampled_tree_->setInputCloud (scene_cloud_downsampled_);

      if(use_super_voxels_)
      {
        float voxel_resolution = 0.005f;
        float seed_resolution = radius_neighborhood_GO_;
        typename pcl::SupervoxelClustering<SceneT> super (voxel_resolution, seed_resolution, false);
        super.setInputCloud (scene_cloud_downsampled_);
        super.setColorImportance (0.f);
        super.setSpatialImportance (1.f);
        super.setNormalImportance (1.f);
        std::map <uint32_t, typename pcl::Supervoxel<SceneT>::Ptr > supervoxel_clusters;
        pcl::console::print_highlight ("Extracting supervoxels!\n");
        super.extract (supervoxel_clusters);
        pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());

        pcl::PointCloud<pcl::PointXYZL>::Ptr supervoxels_labels_cloud = super.getLabeledCloud();
        std::cout << scene_cloud_downsampled_->points.size () << " " << supervoxels_labels_cloud->points.size () << std::endl;

        clusters_cloud_rgb_= super.getColoredCloud();

        /*pcl::visualization::PCLVisualizer vis("supervoxels");
        int v1,v2;
        vis.createViewPort(0,0,0.5,1.0, v1);
        vis.createViewPort(0.5,0,1,1.0, v2);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> handler(colored_cloud);
        vis.addPointCloud<pcl::PointXYZRGBA>(colored_cloud, handler, "labels", v1);
        vis.addPointCloud<SceneT>(scene_cloud_downsampled_, "scene_cloud", v2);
        vis.spin();*/

        clusters_cloud_.reset (new pcl::PointCloud<pcl::PointXYZL>(*supervoxels_labels_cloud));
      }
      else
      {

        clusters_cloud_.reset (new pcl::PointCloud<pcl::PointXYZL>);
        clusters_cloud_rgb_.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);

        if(scene_cloud_->isOrganized())
        {
            PCL_WARN("scene cloud is organized, filter points with high curvature and cluster the rest in smooth patches\n");

            typename pcl::EuclideanClusterComparator<SceneT, pcl::Normal, pcl::Label>::Ptr
                    euclidean_cluster_comparator (new pcl::EuclideanClusterComparator<SceneT, pcl::Normal, pcl::Label> ());

            //create two labels, 1 one for points to be smoothly clustered, another one for the rest
            /*std::vector<pcl::PointIndices> label_indices;
            label_indices.resize (2);
            label_indices[0].indices.resize(scene_cloud_->points.size ());
            label_indices[1].indices.resize(scene_cloud_->points.size ());
            std::vector<int> label_count(2,0);*/
            pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
            labels->points.resize(scene_cloud_->points.size());
            labels->width = scene_cloud_->width;
            labels->height = scene_cloud_->height;
            labels->is_dense = scene_cloud_->is_dense;

            for (size_t j = 0; j < scene_cloud_->points.size (); j++)
            {
              const Eigen::Vector3f& xyz_p = scene_cloud_->points[j].getVector3fMap ();
              if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
              {
                //label_indices[0].indices[label_count[0]++] = static_cast<int>(j);
                labels->points[j].label = 0;
                continue;
              }

              //check normal
              const Eigen::Vector3f& normal = scene_normals_for_clutter_term_->points[j].getNormalVector3fMap ();
              if (!pcl_isfinite (normal[0]) || !pcl_isfinite (normal[1]) || !pcl_isfinite (normal[2]))
              {
                //label_indices[0].indices[label_count[0]++] = static_cast<int>(j);
                labels->points[j].label = 0;
                continue;
              }

              //check curvature
              float curvature = scene_normals_for_clutter_term_->points[j].curvature;
              if(curvature > (curvature_threshold_ * (std::min(1.f,scene_cloud_->points[j].z))))
              {
                  //label_indices[0].indices[label_count[0]++] = static_cast<int>(j);
                  labels->points[j].label = 0;
                  continue;
              }

              //label_indices[1].indices[label_count[1]++] = static_cast<int>(j);
              labels->points[j].label = 1;
            }

            /*label_indices[0].indices.resize(label_count[0]);
            label_indices[1].indices.resize(label_count[1]);
            std::cout << "rejected:" << label_indices[0].indices.size() << std::endl;
            std::cout << "to be clustered:" << label_indices[1].indices.size() << std::endl;*/

            std::vector<bool> excluded_labels;
            excluded_labels.resize (2, false);
            excluded_labels[0] = true;

            euclidean_cluster_comparator->setInputCloud (scene_cloud_);
            euclidean_cluster_comparator->setLabels (labels);
            euclidean_cluster_comparator->setExcludeLabels (excluded_labels);
            euclidean_cluster_comparator->setDistanceThreshold (cluster_tolerance_, true);
            euclidean_cluster_comparator->setAngularThreshold(0.017453 * 5.f); //5 degrees

            pcl::PointCloud<pcl::Label> euclidean_labels;
            std::vector<pcl::PointIndices> clusters;
            pcl::OrganizedConnectedComponentSegmentation<SceneT, pcl::Label> euclidean_segmentation (euclidean_cluster_comparator);
            euclidean_segmentation.setInputCloud (scene_cloud_);
            euclidean_segmentation.segment (euclidean_labels, clusters);

            std::cout << "Number of clusters:" << clusters.size() << std::endl;
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

            int label = 1;
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

            for(size_t i=0; i < scene_sampled_indices_.size(); i++)
            {
                clusters_cloud_->points[i] = clusters_cloud->points[scene_sampled_indices_[i]];
                clusters_cloud_rgb_->points[i].getVector3fMap() = clusters_cloud->points[scene_sampled_indices_[i]].getVector3fMap();

                if(clusters_cloud->points[scene_sampled_indices_[i]].label == 0)
                {
                    clusters_cloud_rgb_->points[i].r = clusters_cloud_rgb_->points[i].g = clusters_cloud_rgb_->points[i].b = 100;
                }
                else
                {
                    clusters_cloud_rgb_->points[i].rgb = label_colors_[clusters_cloud->points[scene_sampled_indices_[i]].label];
                }
            }

            /*{
              int label = 0;
              for (size_t i = 0; i < clusters.size (); i++)
              {
                if(!good_cluster[i])
                    continue;

                for (size_t j = 0; j < clusters[i].indices.size (); j++)
                  clusters_cloud_rgb_->points[clusters[i].indices[j]].rgb = label_colors_[label];

                label++;
              }
            }*/

            /*{
                std::vector<pcl::PointIndices> clusters;
                extractEuclideanClustersSmooth<SceneT, pcl::Normal> (*scene_cloud_downsampled_, *scene_normals_, cluster_tolerance_,
                                                                     scene_downsampled_tree_, clusters, eps_angle_threshold_, curvature_threshold_, min_points_);

                clusters_cloud_->points.resize (scene_cloud_downsampled_->points.size ());
                clusters_cloud_->width = scene_cloud_downsampled_->width;
                clusters_cloud_->height = 1;
                clusters_cloud_->is_dense = scene_cloud_downsampled_->is_dense;

                for (size_t i = 0; i < scene_cloud_downsampled_->points.size (); i++)
                {
                  pcl::PointXYZL p;
                  p.getVector3fMap () = scene_cloud_downsampled_->points[i].getVector3fMap ();
                  p.label = 0;
                  clusters_cloud_->points[i] = p;
                }

                int label = 0;
                for (size_t i = 0; i < clusters.size (); i++)
                {
                  for (size_t j = 0; j < clusters[i].indices.size (); j++)
                    clusters_cloud_->points[clusters[i].indices[j]].label = label;
                  label++;
                }
            }*/
        }
        else
        {

            /*pcl::visualization::PCLVisualizer vis("scene and normals");

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::copyPointCloud(*scene_cloud_downsampled_, *scene_rgb);

            {
                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (scene_rgb);
                vis.addPointCloud (scene_rgb, handler, "scene");
            }

            std::cout << scene_normals_->points.size() << " " << scene_rgb->points.size() << std::endl;
            {
                vis.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (scene_rgb, scene_normals_, 1000, 0.05, "normals_big");
            }

            vis.spin();*/

            std::vector<pcl::PointIndices> clusters;
            extractEuclideanClustersSmooth<SceneT, pcl::Normal> (*scene_cloud_downsampled_, *scene_normals_, cluster_tolerance_,
                                                                 scene_downsampled_tree_, clusters, eps_angle_threshold_, curvature_threshold_, min_points_);

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

            int label = 0;
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

            {
              int label = 0;
              for (size_t i = 0; i < clusters.size (); i++)
              {
                for (size_t j = 0; j < clusters[i].indices.size (); j++)
                  clusters_cloud_rgb_->points[clusters[i].indices[j]].rgb = label_colors_[label];

                label++;
              }
            }
        }
      }

    }

    //compute cues
    {
      //std::vector<bool> valid_model(complete_models_.size (), true);

      valid_model_.resize(complete_models_.size (), true);
      {
        pcl::ScopeTime tcues ("Computing cues");
        recognition_models_.resize (complete_models_.size ());
  #pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_num_procs())
        for (int i = 0; i < static_cast<int> (complete_models_.size ()); i++)
        {
          //create recognition model
          recognition_models_[i].reset (new RecognitionModel<ModelT> ());
          if(!addModel(i, recognition_models_[i])) {
            valid_model_[i] = false;
            PCL_WARN("Model is not valid\n");
          }
        }
      }

      /*{
        pcl::ScopeTime tcues ("go through valid model vector");
        //go through valid model vector
        int valid = 0;
        for (int i = 0; i < static_cast<int> (valid_model.size ()); i++) {
          if(valid_model[i]) {
            recognition_models_[valid] = recognition_models_[i];
            //indices_[valid] = i;
            valid++;
          }
        }

        recognition_models_.resize (valid);
        //indices_.resize (valid);
      }*/

      //compute the bounding boxes for the models
      {
        pcl::ScopeTime tcues ("complete_cloud_occupancy_by_RM_");
        ModelT min_pt_all, max_pt_all;
        min_pt_all.x = min_pt_all.y = min_pt_all.z = std::numeric_limits<float>::max ();
        max_pt_all.x = max_pt_all.y = max_pt_all.z = (std::numeric_limits<float>::max () - 0.001f) * -1;

        for (size_t i = 0; i < recognition_models_.size (); i++)
        {
          if(!valid_model_[i])
            continue;

          ModelT min_pt, max_pt;
          //pcl::getMinMax3D (*complete_models_[indices_[i]], min_pt, max_pt);
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

        int size_x, size_y, size_z;
        size_x = static_cast<int> (std::ceil (std::abs (max_pt_all.x - min_pt_all.x) / res_occupancy_grid_)) + 1;
        size_y = static_cast<int> (std::ceil (std::abs (max_pt_all.y - min_pt_all.y) / res_occupancy_grid_)) + 1;
        size_z = static_cast<int> (std::ceil (std::abs (max_pt_all.z - min_pt_all.z) / res_occupancy_grid_)) + 1;

        complete_cloud_occupancy_by_RM_.resize (size_x * size_y * size_z, 0);

        for (size_t i = 0; i < recognition_models_.size (); i++)
        {
          if(!valid_model_[i])
            continue;

          //std::map<int, bool> banned;
          //std::map<int, bool>::iterator banned_it;
          std::vector<bool> banned_vector(size_x * size_y * size_z, false);
          //recognition_models_[i]->complete_cloud_occupancy_indices_.resize(complete_models_[indices_[i]]->points.size ());
          recognition_models_[i]->complete_cloud_occupancy_indices_.resize(complete_models_[i]->points.size ());
          int used = 0;
          //for (size_t j = 0; j < complete_models_[indices_[i]]->points.size (); j++)
          for (size_t j = 0; j < complete_models_[i]->points.size (); j++)
          {
            int pos_x, pos_y, pos_z;
            //pos_x = static_cast<int> (std::floor ((complete_models_[indices_[i]]->points[j].x - min_pt_all.x) / res_occupancy_grid_));
            //pos_y = static_cast<int> (std::floor ((complete_models_[indices_[i]]->points[j].y - min_pt_all.y) / res_occupancy_grid_));
            //pos_z = static_cast<int> (std::floor ((complete_models_[indices_[i]]->points[j].z - min_pt_all.z) / res_occupancy_grid_));
            pos_x = static_cast<int> (std::floor ((complete_models_[i]->points[j].x - min_pt_all.x) / res_occupancy_grid_));
            pos_y = static_cast<int> (std::floor ((complete_models_[i]->points[j].y - min_pt_all.y) / res_occupancy_grid_));
            pos_z = static_cast<int> (std::floor ((complete_models_[i]->points[j].z - min_pt_all.z) / res_occupancy_grid_));

            int idx = pos_z * size_x * size_y + pos_y * size_x + pos_x;
            /*banned_it = banned.find (idx);
            if (banned_it == banned.end ())
            {
              //complete_cloud_occupancy_by_RM_[idx]++;
              //recognition_models_[i]->complete_cloud_occupancy_indices_.push_back (idx);
              recognition_models_[i]->complete_cloud_occupancy_indices_[used] = idx;
              banned[idx] = true;
              used++;
            }*/

            assert(banned_vector.size() > idx);
            if (!banned_vector[idx])
            {
              //complete_cloud_occupancy_by_RM_[idx]++;
              //recognition_models_[i]->complete_cloud_occupancy_indices_.push_back (idx);
              recognition_models_[i]->complete_cloud_occupancy_indices_[used] = idx;
              banned_vector[idx] = true;
              used++;
            }
          }

          recognition_models_[i]->complete_cloud_occupancy_indices_.resize(used);
        }
      }
    }

    {

#ifdef FAAT_PCL_RECOGNITION_USE_GPU
      computeClutterCueGPU();
#else
      {
          pcl::ScopeTime tcues ("Computing clutter cues");
          #pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_num_procs())
              for (int j = 0; j < static_cast<int> (recognition_models_.size ()); j++)
                computeClutterCue (recognition_models_[j]);
      }
#endif
    }

    points_explained_by_rm_.clear ();
    points_explained_by_rm_.resize (scene_cloud_downsampled_->points.size ());
    for (size_t j = 0; j < recognition_models_.size (); j++)
    {
      boost::shared_ptr<RecognitionModel<ModelT> > recog_model = recognition_models_[j];
      for (size_t i = 0; i < recog_model->explained_.size (); i++)
      {
        points_explained_by_rm_[recog_model->explained_[i]].push_back (recog_model);
      }
    }

    /*if (use_conflict_graph_)
    {
      graph_id_model_map_.clear ();
      conflict_graph_.clear ();
      cc_.clear ();

      // create vertices for the graph
      for (size_t i = 0; i < (recognition_models_.size ()); i++)
      {
        const typename Graph::vertex_descriptor v = boost::add_vertex (recognition_models_[i], conflict_graph_);
      }

      // iterate over the remaining models and check for each one if there is a conflict with another one
      for (size_t i = 0; i < recognition_models_.size (); i++)
      {
        for (size_t j = i; j < recognition_models_.size (); j++)
        {
          if (i != j)
          {
            bool add_conflict = false;

            // count scene points explained by both models
            for (size_t k = 0; k < points_explained_by_rm_.size () && !add_conflict; k++)
            {
              if (points_explained_by_rm_[k].size () > 1)
              {
                // this point could be a conflict
                bool i_found = false;
                bool j_found = false;
                bool both_found = false;
                for (size_t kk = 0; (kk < points_explained_by_rm_[k].size ()) && !both_found; kk++)
                {
                  if (points_explained_by_rm_[k][kk]->id_ == recognition_models_[i]->id_)
                    i_found = true;

                  if (points_explained_by_rm_[k][kk]->id_ == recognition_models_[j]->id_)
                    j_found = true;

                  if (i_found && j_found)
                    both_found = true;
                }

                if (both_found)
                  add_conflict = true;
              }
            }

            if (add_conflict)
            {
              boost::add_edge (i, j, conflict_graph_);
            }
          }
        }
      }

      boost::vector_property_map<int> components (boost::num_vertices (conflict_graph_));
      n_cc_ = static_cast<int> (boost::connected_components (conflict_graph_, &components[0]));

      //std::cout << "Number of connected components:" << n_cc_ << std::endl;
      cc_.resize (n_cc_);
      for (unsigned int i = 0; i < boost::num_vertices (conflict_graph_); i++)
        cc_[components[i]].push_back (i);

      for (int i = 0; i < n_cc_; i++)
      {
        //std::cout << "Component " << i << " : " << cc_[i].size () << std::endl;
        for (size_t kk = 0; kk < cc_[i].size (); kk++)
        {
          std::cout << "\t" << cc_[i][kk] << std::endl;
        }
      }
    }
    else
    {
      cc_.clear ();
      n_cc_ = 1;
      cc_.resize (n_cc_);
      for (size_t i = 0; i < recognition_models_.size (); i++)
        cc_[0].push_back (static_cast<int> (i));
    }*/

    cc_.clear ();
    n_cc_ = 1;
    cc_.resize (n_cc_);
    for (size_t i = 0; i < recognition_models_.size (); i++)
    {
      if(!valid_model_[i])
        continue;

      cc_[0].push_back (static_cast<int> (i));
    }
  }

  template<typename ModelT, typename SceneT>
  void
  faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::updateExplainedVector (std::vector<int> & vec,
                         std::vector<float> & vec_float, std::vector<int> & explained_,
                         std::vector<double> & explained_by_RM_distance_weighted, float sign)
  {
    float add_to_explained = 0.f;
    float add_to_duplicity_ = 0;

    for (size_t i = 0; i < vec.size (); i++)
    {
      bool prev_dup = explained_[vec[i]] > 1;
      bool prev_explained = explained_[vec[i]] == 1;
      float prev_explained_value = explained_by_RM_distance_weighted[vec[i]];

      explained_[vec[i]] += static_cast<int> (sign);
      explained_by_RM_distance_weighted[vec[i]] += vec_float[i] * sign;

      //add_to_explained += vec_float[i] * sign;
      if (explained_[vec[i]] == 1 && !prev_explained)
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
        add_to_explained -= prev_explained_value;

      if ((explained_[vec[i]] > 1) && prev_dup)
      { //its still a duplicate
        add_to_duplicity_ += vec_float[i] * static_cast<int> (sign) / 2.f; //so, just add or remove one
      }
      else if ((explained_[vec[i]] == 1) && prev_dup)
      { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
        add_to_duplicity_ -= prev_explained_value / 2.f; //explained_by_RM_distance_weighted[vec[i]];
      }
      else if ((explained_[vec[i]] > 1) && !prev_dup)
      { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
        add_to_duplicity_ += explained_by_RM_distance_weighted[vec[i]]  / 2.f;
      }
    }

    //update explained and duplicity values...
    previous_explained_value += add_to_explained;
    previous_duplicity_ += add_to_duplicity_;
  }

  template<typename ModelT, typename SceneT>
  void
  faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::updateCMDuplicity (std::vector<int> & vec, std::vector<int> & occupancy_vec, float sign)
  {
    int add_to_duplicity_ = 0;
    for (size_t i = 0; i < vec.size (); i++)
    {
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
  float
  faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::getTotalExplainedInformation (std::vector<int> & explained_, std::vector<double> & explained_by_RM_distance_weighted, float * duplicity_)
  {
    float explained_info = 0;
    float duplicity = 0;

    for (size_t i = 0; i < explained_.size (); i++)
    {
      //if (explained_[i] > 0)
      if (explained_[i] == 1) //only counts points that are explained once
      {
        //explained_info += explained_by_RM_distance_weighted[i] / 2.f; //what is the magic division by 2?
        explained_info += explained_by_RM_distance_weighted[i];
      }
      if (explained_[i] > 1)
      {
        //duplicity += explained_by_RM_distance_weighted[i];
        duplicity += explained_by_RM_distance_weighted[i] / 2.f;
      }
    }

    *duplicity_ = duplicity;
    PCL_WARN("fixed error\n");
    std::cout << explained_info << " " << duplicity << std::endl;
    return explained_info;
  }

template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::fill_structures(std::vector<int> & cc_indices, std::vector<bool> & initial_solution, SAModel<ModelT, SceneT> & model)
{
  for (size_t j = 0; j < recognition_models_.size (); j++)
  {
    if(!initial_solution[j])
      continue;

    boost::shared_ptr<RecognitionModel<ModelT> > recog_model = recognition_models_[j];
    for (size_t i = 0; i < recog_model->explained_.size (); i++)
    {
      explained_by_RM_[recog_model->explained_[i]]++;
      explained_by_RM_distance_weighted[recog_model->explained_[i]] += recog_model->explained_distances_[i];
    }

    if (detect_clutter_)
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

  int occupied_multiple = 0;
  for (size_t i = 0; i < complete_cloud_occupancy_by_RM_.size (); i++)
  {
    if (complete_cloud_occupancy_by_RM_[i] > 1)
    {
      occupied_multiple += complete_cloud_occupancy_by_RM_[i];
    }
  }

  setPreviousDuplicityCM (occupied_multiple);
  //do optimization
  //Define model SAModel, initial solution is all models activated

  float duplicity;
  float good_information_ = getTotalExplainedInformation (explained_by_RM_, explained_by_RM_distance_weighted, &duplicity);
  float bad_information_ = 0;

  float unexplained_in_neighboorhod;
  if(detect_clutter_) {
    unexplained_in_neighboorhod = getUnexplainedInformationInNeighborhood (unexplained_by_RM_neighboorhods, explained_by_RM_);
  } else {
    unexplained_in_neighboorhod = 0;
  }

  for (size_t i = 0; i < initial_solution.size (); i++)
  {
    if (initial_solution[i])
      bad_information_ += recognition_models_[i]->outliers_weight_ * static_cast<float> (recognition_models_[i]->bad_information_);
  }

  setPreviousExplainedValue (good_information_);
  setPreviousDuplicity (duplicity);
  setPreviousBadInfo (bad_information_);
  setPreviousUnexplainedValue (unexplained_in_neighboorhod);

  float under_table = 0;
  if (model_constraints_.size () > 0)
  {
    under_table = getModelConstraintsValueForActiveSolution (initial_solution);
  }

  model.cost_ = static_cast<mets::gol_type> ((good_information_ - bad_information_ - static_cast<float> (duplicity)
      - static_cast<float> (occupied_multiple) * w_occupied_multiple_cm_ -
      - unexplained_in_neighboorhod - under_table - countActiveHypotheses (initial_solution)) * -1.f);

  model.setSolution (initial_solution);
  model.setOptimizer (this);

  std::cout << "*****************************" << std::endl;
  std::cout << "Cost recomputing:" <<
            static_cast<mets::gol_type> ((good_information_ - bad_information_ - static_cast<float> (duplicity)
          - static_cast<float> (occupied_multiple) * w_occupied_multiple_cm_
          - unexplained_in_neighboorhod - under_table - countActiveHypotheses (initial_solution)) * -1.f) << std::endl;
  std::cout << "*****************************" << std::endl;

}

template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::clear_structures()
{
  size_t kk = complete_cloud_occupancy_by_RM_.size();
  explained_by_RM_.clear();
  explained_by_RM_distance_weighted.clear();
  unexplained_by_RM_neighboorhods.clear();
  complete_cloud_occupancy_by_RM_.clear();

  explained_by_RM_.resize (scene_cloud_downsampled_->points.size (), 0);
  explained_by_RM_distance_weighted.resize (scene_cloud_downsampled_->points.size (), 0.f);
  unexplained_by_RM_neighboorhods.resize (scene_cloud_downsampled_->points.size (), 0.f);
  complete_cloud_occupancy_by_RM_.resize(kk, 0);
}

template<typename ModelT, typename SceneT>
  void
  faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::SAOptimize (std::vector<int> & cc_indices, std::vector<bool> & initial_solution)
  {

    //temporal copy of recogniton_models_
    std::vector<boost::shared_ptr<RecognitionModel<ModelT> > > recognition_models_copy;
    recognition_models_copy = recognition_models_;

    recognition_models_.clear ();

    for (size_t j = 0; j < cc_indices.size (); j++)
      recognition_models_.push_back (recognition_models_copy[cc_indices[j]]);

    SAModel<ModelT, SceneT> model;
    fill_structures(cc_indices, initial_solution, model);

    SAModel<ModelT, SceneT> * best = new SAModel<ModelT, SceneT> (model);

    move_manager<ModelT, SceneT> neigh (static_cast<int> (cc_indices.size ()), use_replace_moves_);
    boost::shared_ptr<std::map< std::pair<int, int>, bool > > intersect_map;
    intersect_map.reset(new std::map< std::pair<int, int>, bool >);

    if(use_replace_moves_)
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
            std::map<int, int>::iterator it1, it2;
            it1 = model_to_planar_model_.find(static_cast<int>(i));
            it2 = model_to_planar_model_.find(static_cast<int>(j));
            if(it1 != model_to_planar_model_.end() || it2 != model_to_planar_model_.end())
            {
              //PCL_INFO("Conflict with plane\n");
            }
          }
        }
      }

      std::cout << "num_conflicts:" << num_conflicts << " " << recognition_models_.size() * recognition_models_.size() << std::endl;

      /*for (size_t i = 0; i < recognition_models_.size (); i++)
      {
        std::cout << "id:" << recognition_models_[i]->id_ << std::endl;
//#pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_num_procs())
        for (size_t j = i; j < recognition_models_.size (); j++)
        {
          if (i != j)
          {
            float n_conflicts = 0.f;
            bool conflict = false;

            // count scene points explained by both models
            for (size_t k = 0; k < explained_by_RM_.size (); k++)
            {
              if (explained_by_RM_[k] > 1)
              {
                // this point could be a conflict
                bool i_found = false;
                bool j_found = false;
                bool both_found = false;
                for (size_t kk = 0; (kk < points_explained_by_rm_[k].size ()) && !both_found; kk++)
                {
                  if (points_explained_by_rm_[k][kk]->id_ == recognition_models_[i]->id_)
                    i_found = true;

                  if (points_explained_by_rm_[k][kk]->id_ == recognition_models_[j]->id_)
                    j_found = true;

                  if (i_found && j_found)
                    both_found = true;
                }

                if (both_found)
                  n_conflicts += 1.f;

                conflict = (n_conflicts > 10);
                if(conflict) break;
              }
            }

            std::pair<int, int> p = std::make_pair<int, int> (static_cast<int> (i), static_cast<int> (j));
            (*intersect_map)[p] = conflict;
          }
        }
      }*/
    }

    neigh.setExplainedPointIntersections(intersect_map);

    //mets::best_ever_solution best_recorder (best);
    cost_logger_.reset(new CostFunctionLogger<ModelT, SceneT>(*best));
    mets::noimprove_termination_criteria noimprove (max_iterations_);

    switch(opt_type_)
    {
      case 0:
      {
        bool short_circuit = false;
        mets::local_search<move_manager<ModelT, SceneT> > local ( model, *(cost_logger_.get()), neigh, 0, short_circuit);
        {
          pcl::ScopeTime t ("local search...");
          local.search ();
        }
        break;
      }
      case 1:
      {
        //Tabu search
        mets::simple_tabu_list tabu_list ( initial_solution.size() * sqrt ( 1.0*initial_solution.size() ) ) ;
        mets::best_ever_criteria aspiration_criteria ;
        mets::tabu_search<move_manager<ModelT, SceneT> > tabu_search(model,  *(cost_logger_.get()), neigh, tabu_list, aspiration_criteria, noimprove);
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
      case 2:
      {

        mets::local_search<move_manager<ModelT, SceneT> > local ( model,  *(cost_logger_.get()), neigh, 0, false);

        {
          pcl::ScopeTime t ("local search WITHIN B&B...");
          local.search ();
        }

        best_seen_ = static_cast<const SAModel<ModelT, SceneT>&> (cost_logger_->best_seen ());
        std::cout << "*****************************" << std::endl;
        std::cout << "Final cost:" << best_seen_.cost_;
        std::cout << " Number of ef evaluations:" << cost_logger_->getTimesEvaluated();
        std::cout << std::endl;
        std::cout << "Number of accepted moves:" << cost_logger_->getAcceptedMovesSize() << std::endl;
        std::cout << "*****************************" << std::endl;

        clear_structures();
        recognition_models_.clear ();
        for (size_t j = 0; j < cc_indices.size (); j++)
          recognition_models_.push_back (recognition_models_copy[cc_indices[j]]);

        //sort recognition models so that selected models by local search are at the beginning
        //adapt initial_solution accordingly
        std::map<int, int> sorted_to_non_sorted;
        {
          std::vector<boost::shared_ptr<RecognitionModel<ModelT> > > recognition_models_copy2;

          int s = 0;
          for(size_t j=0; j < best_seen_.solution_.size(); j++) {
            if(best_seen_.solution_[j]) {
              recognition_models_copy2.push_back(recognition_models_[j]);
              sorted_to_non_sorted[s] = static_cast<int>(j);
              s++;
              //initial_solution[j] = true;
            }
          }

          for(size_t j=0; j < best_seen_.solution_.size(); j++) {
            if(!best_seen_.solution_[j]) {
              recognition_models_copy2.push_back(recognition_models_[j]);
              sorted_to_non_sorted[s] = static_cast<int>(j);
              s++;
              //initial_solution[j] = false;
            }
          }

          recognition_models_ = recognition_models_copy2;
        }

        for(size_t i=0; i < initial_solution.size(); i++) {
          initial_solution[i] = false;
        }

        fill_structures(cc_indices, initial_solution, model);
        delete best;
        best = new SAModel<ModelT, SceneT>(model);
        cost_logger_.reset(new CostFunctionLogger<ModelT, SceneT> (*best));

        //brute force with branch and bound
        HVGOBinaryOptimizer<ModelT, SceneT>  bin_opt(model,  *(cost_logger_.get()), neigh, initial_solution.size());
        bin_opt.setRecogModels(recognition_models_);
        bin_opt.computeStructures(complete_cloud_occupancy_by_RM_.size(),
                                  explained_by_RM_distance_weighted.size());

        bin_opt.setIncumbent(static_cast<float>(best_seen_.cost_ + 1));
        std::cout << "Incumbent value is:" << static_cast<float>(best_seen_.cost_) << std::endl;
        bin_opt.setOptimizer(this);

        {
          pcl::ScopeTime t ("Brute force search...");
          bin_opt.search();
        }

        best_seen_ = static_cast<const SAModel<ModelT, SceneT>&> (cost_logger_->best_seen ());
        //remap solution using sorted_to_non_sorted
        std::vector<bool> final_solution;
        final_solution.resize(initial_solution.size(), false);
        std::map<int, int>::iterator it;
        for(it = sorted_to_non_sorted.begin(); it != sorted_to_non_sorted.end(); it++)
        {
          final_solution[it->second] = best_seen_.solution_[it->first];
          //std::cout << best_seen_.solution_[it->first] << " - " << final_solution[it->second] << std::endl;
        }

        /*for(size_t i=0; i < final_solution.size(); i++) {
          std::cout << final_solution[i] << " ";
        }
        std::cout << std::endl;*/

        cost_logger_->setBestSolution(final_solution);
        break;
      }
      default:
      {
        //Simulated Annealing
        //mets::linear_cooling linear_cooling;
        mets::exponential_cooling linear_cooling;
        mets::simulated_annealing<move_manager<ModelT, SceneT> > sa (model,  *(cost_logger_.get()), neigh, noimprove, linear_cooling, initial_temp_, 1e-7, 2);
        sa.setApplyAndEvaluate (true);

        {
          pcl::ScopeTime t ("SA search...");
          sa.search ();
        }
        break;
      }
    }

    best_seen_ = static_cast<const SAModel<ModelT, SceneT>&> (cost_logger_->best_seen ());
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
        std::cout << "id: " << recognition_models_[i]->id_s_ << std::endl;
        std::cout << "Median:" << recognition_models_[i]->median_ << std::endl;
        std::cout << "Mean:" << recognition_models_[i]->mean_ << std::endl;
        std::cout << "Color similarity:" << recognition_models_[i]->color_similarity_ << std::endl;
        std::cout << "#outliers:" << recognition_models_[i]->bad_information_ << " " << recognition_models_[i]->outliers_weight_ << std::endl;
        std::cout << "#under table:" << recognition_models_[i]->model_constraints_value_ << std::endl;
        std::cout << "#explained:" << recognition_models_[i]->explained_.size() << std::endl;

        /*int v1, v2, v3;
        vis_.createViewPort (0.0, 0.0, 0.33, 1.0, v1);
        vis_.createViewPort (0.33, 0.0, 0.66, 1.0, v2);
        vis_.createViewPort (0.66, 0.0, 1, 1.0, v3);

        //typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_ps(new pcl::PointCloud<pcl::PointXYZRGB>());
        //pcl::copyPointCloud(*recog_model->cloud_, model_indices_explained, *model_ps);
        typename pcl::PointCloud<ModelT>::Ptr model_ps(new pcl::PointCloud<ModelT>(*(recognition_models_[i]->cloud_)));

        {
          pcl::visualization::PointCloudColorHandlerCustom<ModelT> handler_rgb_model (recognition_models_[i]->cloud_, 255, 0, 0);
          vis_.addPointCloud<ModelT> (recognition_models_[i]->cloud_, handler_rgb_model, "nmodel_orig", v3);

          typename pcl::PointCloud<ModelT>::Ptr outliers(new pcl::PointCloud<ModelT>());
          pcl::copyPointCloud(*recognition_models_[i]->cloud_, recognition_models_[i]->outlier_indices_, *outliers);
          pcl::visualization::PointCloudColorHandlerCustom<ModelT> handler_outliers (outliers, 255, 255, 0);
          vis_.addPointCloud<ModelT> (outliers, handler_outliers, "outliers", v3);
        }*/

        /*pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_rgb_model (model_ps);
        vis_.addPointCloud<pcl::PointXYZ> (model_ps, handler_rgb_model, "nmodel", v2);

        typename pcl::PointCloud<pcl::PointXYZ>::Ptr explained_ps(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*scene_cloud_downsampled_, recognition_models_[i]->explained_, *explained_ps);

        std::cout << recognition_models_[i]->cloud_->points.size() << " " << explained_ps->points.size() << std::endl;

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZ> handler_rgb_scene (explained_ps);
        vis_.addPointCloud<pcl::PointXYZ> (explained_ps, handler_rgb_scene, "scene_cloud", v1);

        {
          typename pcl::PointCloud<pcl::PointXYZ>::Ptr unexplained(new pcl::PointCloud<pcl::PointXYZ>());
          pcl::copyPointCloud(*scene_cloud_downsampled_, recognition_models_[i]->unexplained_in_neighborhood, *unexplained);
          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_outliers (unexplained, 0, 255, 255);
          vis_.addPointCloud<pcl::PointXYZ> (unexplained, handler_outliers, "unexplained", v1);
        }*/

        //vis_.spin();
        //vis_.removeAllPointClouds();
      }
    }

    delete best;

    recognition_models_ = recognition_models_copy;

  }

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename ModelT, typename SceneT>
  void
  faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::verify ()
  {
    initialize ();

    //for each connected component, find the optimal solution
    for (int c = 0; c < n_cc_; c++)
    {
      //TODO: Check for trivial case...
      //TODO: Check also the number of hypotheses and use exhaustive enumeration if smaller than 10
      std::vector<bool> subsolution (cc_[c].size (), initial_status_);
      SAOptimize (cc_[c], subsolution);
      for (size_t i = 0; i < subsolution.size (); i++)
      {
        //mask_[indices_[cc_[c][i]]] = (subsolution[i]);
        mask_[cc_[c][i]] = (subsolution[i]);
      }
    }
  }

inline void softBining(float val, int pos1, float bin_size, int max_pos, int & pos2, float & w1, float & w2) {
  float c1 = pos1 * bin_size + bin_size / 2;
  pos2 = 0;
  float c2 = 0;
  if(pos1 == 0) {
    pos2 = 1;
    c2 = pos2 * bin_size + bin_size / 2;
  } else if(pos1 == (max_pos-1)) {
    pos2 = max_pos-2;
    c2 = pos2 * bin_size + bin_size / 2;
  } else {
    if(val > c1) {
      pos2 = pos1 + 1;
    } else {
      pos2 = pos1 - 1;
    }

    c2 = pos2 * bin_size + bin_size / 2;
  }

  w1 = (val - c1) / (c2 - c1);
  w2 = (c2 - val) / (c2 - c1);
}

template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::computeYUVHistogram(std::vector<Eigen::Vector3f> & yuv_values,
                                                                                      Eigen::VectorXf & histogram)
{
  int size_uv = 8;
  int size_y = 3;
  histogram = Eigen::VectorXf::Zero(size_uv * size_uv * size_y);
  float size_bin_uv = 1.f / static_cast<float>(size_uv);
  float size_bin_y = 1.f / static_cast<float>(size_y);

  for(size_t i=0; i < yuv_values.size(); i++) {
    int uu;
    int vv;
    int yy;

    uu = std::floor (yuv_values[i][1] / 255.f * size_uv);
    vv = std::floor (yuv_values[i][2] / 255.f * size_uv);
    yy = std::floor ((yuv_values[i][0] - 16.f) / (235.f - 16.f + 1.f) * size_y);
    assert (yy < size_y && uu < size_uv && vv < size_uv);

    //soft bining for all 3 channels
      //determine two positions for all 3 channels
      //determine weightsy
      //increase histogram

    int pos2_y=0;
    float w1_y=0, w2_y=0;
    softBining(yuv_values[i][0] / 219.f, yy, size_bin_y, size_y, pos2_y, w1_y, w2_y);

    int pos2_u=0;
    float w1_u=0, w2_u=0;
    softBining(yuv_values[i][1] / 255.f, uu, size_bin_uv, size_uv, pos2_u, w1_u, w2_u);

    int pos2_v=0;
    float w1_v=0, w2_v=0;
    softBining(yuv_values[i][2] / 255.f, vv, size_bin_uv, size_uv, pos2_v, w1_v, w2_v);

    histogram[yy * size_uv * size_uv + uu * size_uv + vv] += w1_y * w1_u * w1_v;
    histogram[yy * size_uv * size_uv + pos2_u * size_uv + vv] += w1_y * w2_u * w1_v;
    histogram[yy * size_uv * size_uv + uu * size_uv + pos2_v] += w1_y * w1_u * w2_v;
    histogram[yy * size_uv * size_uv + pos2_u * size_uv + pos2_v] += w1_y * w2_u * w2_v;

    histogram[pos2_y * size_uv * size_uv + uu * size_uv + vv] += w2_y * w1_u * w1_v;
    histogram[pos2_y * size_uv * size_uv + pos2_u * size_uv + vv] += w2_y * w2_u * w1_v;
    histogram[pos2_y * size_uv * size_uv + uu * size_uv + pos2_v] += w2_y * w1_u * w2_v;
    histogram[pos2_y * size_uv * size_uv + pos2_u * size_uv + pos2_v] += w2_y * w2_u * w2_v;
    //histogram[pos2_y * size_uv * size_uv + pos2_u * size_uv + pos2_v] += w2_y * w2_u * w2_v;
    //histogram[yy * size_uv * size_uv + uu * size_uv + vv]++;
  }

  //we could do a high pass filter here
  float max_value = 0;
  for(size_t i=0; i < histogram.size(); i++) {
    if(histogram[i] > max_value) {
      max_value = histogram[i];
    }
  }

/*for(size_t i=0; i < histogram.size(); i++) {
    if(histogram[i] < (max_value * 0.1f)) {
      histogram[i] = 0.f;
    }
  }
  */
}

//Hue values between 0 and 1
template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::computeHueHistogram(std::vector<Eigen::Vector3f> & hsv_values,
                                                                                      Eigen::VectorXf & histogram)
{

  histogram = Eigen::VectorXf::Zero(27);
  float size_bin = 1.f / 25.f;
  float high_sat, low_sat, val;
  val = 0.15f;
  high_sat = 1.f - val;
  low_sat = val;

  for(size_t i=0; i < hsv_values.size(); i++) {
    if(hsv_values[i][1] < low_sat) //low value, dark areas
    {
      //std::cout << "high sat" << std::endl;
      histogram[25]++;
    }
    else if (hsv_values[i][1] > high_sat) //bright areas
    {
      histogram[26]++;
      //std::cout << "low sat" << hsv_values[i][1] << std::endl;
    }
    else
    {
      int pos = floor(hsv_values[i][0] / size_bin);
      if(pos < 0)
        pos = 0;

      if(pos > 24)
        pos = 24;

      float c1 = pos * size_bin + size_bin / 2;
      int pos2 = 0;
      float c2 = 0;
      if(pos == 0) {
        pos2 = 1;
        c2 = pos2 * size_bin + size_bin / 2;;
      } else if(pos == 24) {
        pos2 = 23;
        c2 = pos2 * size_bin + size_bin / 2;;
      } else {
        if(hsv_values[i][0] > c1) {
          pos2 = pos + 1;
        } else {
          pos2 = pos - 1;
        }

        c2 = pos2 * size_bin + size_bin / 2;
      }

      histogram[pos] += (hsv_values[i][0] - c1) / (c2 - c1);
      histogram[pos2] += (c2 - hsv_values[i][0]) / (c2 - c1);
      //histogram[pos]++;
    }
  }
}

template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::computeGSHistogram
    (std::vector<float> & gs_values, Eigen::MatrixXf & histogram)
{
    float max = 255.f;
    float min = 0.f;
    int dim = 1;
    int hist_size = max - min + 1;

    histogram = Eigen::MatrixXf (hist_size, dim);
    histogram.setZero ();
    for (size_t j = 0; j < gs_values.size (); j++)
    {
        int pos = std::floor (static_cast<float> (gs_values[j] - min) / (max - min) * hist_size);
        if(pos < 0)
          pos = 0;

        if(pos > hist_size)
          pos = hist_size - 1;

        histogram (pos, 0)++;
    }
}

template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::computeRGBHistograms (std::vector<Eigen::Vector3f> & rgb_values, Eigen::MatrixXf & rgb, int dim, float min, float max, bool soft)
{
  int hist_size = max - min + 1;
  float size_bin = 1.f / hist_size;
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

      /*if (soft)
      {
        float c1 = pos * size_bin + size_bin / 2;
        int pos2 = 0;
        float c2 = 0;
        if (pos == 0)
        {
          pos2 = 1;
          c2 = pos2 * size_bin + size_bin / 2;
        }
        else if (pos == 24)
        {
          pos2 = 23;
          c2 = pos2 * size_bin + size_bin / 2;
        }
        else
        {
          if ((static_cast<float> (rgb_values[j][i]) / 255.f) > c1)
          {
            pos2 = pos + 1;
          }
          else
          {
            pos2 = pos - 1;
          }

          c2 = pos2 * size_bin + size_bin / 2;
        }

        rgb (pos, i) += ((static_cast<float> (rgb_values[j][i]) / 255.f) - c1) / (c2 - c1);
        rgb (pos2, i) += (c2 - (static_cast<float> (rgb_values[j][i]) / 255.f)) / (c2 - c1);
      }

      else
      {*/
        rgb (pos, i)++;
      //}

    }
  }
}

template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::specifyRGBHistograms (Eigen::MatrixXf & src, Eigen::MatrixXf & dst, Eigen::MatrixXf & lookup, int dim)
{
  for(size_t i=0; i < dim; i++) {
    src.col(i) /= src.col(i).sum();
    dst.col(i) /= dst.col(i).sum();
  }

  Eigen::MatrixXf src_cumulative(src.rows(), dim);
  Eigen::MatrixXf dst_cumulative(dst.rows(), dim);
  lookup = Eigen::MatrixXf(src.rows(), dim);
  lookup.setZero();

  src_cumulative.setZero();
  dst_cumulative.setZero();

  for (size_t i = 0; i < dim; i++)
  {
    src_cumulative (0, i) = src (0, i);
    dst_cumulative (0, i) = dst (0, i);
    for (size_t j = 1; j < src_cumulative.rows (); j++)
    {
      src_cumulative (j, i) = src_cumulative (j - 1, i) + src (j, i);
      dst_cumulative (j, i) = dst_cumulative (j - 1, i) + dst (j, i);
    }

    /*int last = 0;
            for(k=0; k < 256; k++) {
                    for(z = last; z < 256; z++) {
                            if ( G[z] - T[k]  >= 0 ) {
                                    if(z > 0 && T[k] - G[z-1] < G[z] - T[k])
                                            z--;
                                    map[k] = (unsigned char) z;
                                    last = z;
                                    break;
                            }
                    }
            }*/

    int last = 0;
    for (int k = 0; k < src_cumulative.rows (); k++)
    {
      for (int z = last; z < src_cumulative.rows (); z++)
      {
        if (src_cumulative (z, i) - dst_cumulative (k, i) >= 0)
        {
          if (z > 0 && dst_cumulative (k, i) - src_cumulative (z - 1, i) < src_cumulative (z, i) - dst_cumulative (k, i))
            z--;

          lookup (k, i) = z;
          last = z;
          break;
        }
      }
    }

    int min = 0;
    for (int k = 0; k < src_cumulative.rows (); k++)
    {
      if (lookup (k, i) != 0)
      {
        min = lookup (k, i);
        break;
      }
    }

    for (int k = 0; k < src_cumulative.rows (); k++)
    {
      if (lookup (k, i) == 0)
        lookup (k, i) = min;
      else
        break;
    }

    //max mapping extension
    int max = 0;
    for (int k = (src_cumulative.rows () - 1); k >= 0; k--)
    {
      if (lookup (k, i) != 0)
      {
        max = lookup (k, i);
        break;
      }
    }

    for (int k = (src_cumulative.rows () - 1); k >= 0; k--)
    {
      if (lookup (k, i) == 0)
        lookup (k, i) = max;
      else
        break;
    }
  }

  Eigen::MatrixXf src_specified(src.rows(), dim);
  src_specified.setZero();
  for (size_t i = 0; i < dim; i++)
  {
    for (size_t j = 0; j < src_cumulative.rows (); j++) {
      src_specified(lookup(j, i), i) += src(j,i);
    }
  }

  src = src_specified;
}

template<typename ModelT, typename SceneT>
bool
faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::handlingNormals (boost::shared_ptr<RecognitionModel<ModelT> > & recog_model, int i, bool is_planar_model, int object_models_size)
{
    //std::cout << visible_normal_models_.size() << " " << object_models_size << " " << complete_models_.size() << std::endl;
    if(visible_normal_models_.size() == object_models_size && !is_planar_model)
    {

      //std::cout << "is planar model:" << (int)is_planar_model << std::endl;
      pcl::PointCloud<pcl::Normal>::ConstPtr model_normals = visible_normal_models_[i];
      pcl::ScopeTime t("Using model normals and checking nans");

      //std::cout << "Using model normals" << std::endl;
      recog_model->normals_.reset (new pcl::PointCloud<pcl::Normal> ());
      recog_model->normals_->points.resize(recog_model->cloud_->points.size ());

      //std::cout << model_normals->points.size() << " " << recog_model->cloud_->points.size () << std::endl;
      //check nans...
      int j = 0;
      for (size_t i = 0; i < recog_model->cloud_->points.size (); ++i)
      {
        if (!pcl_isfinite (recog_model->cloud_->points[i].x) || !pcl_isfinite (recog_model->cloud_->points[i].y)
            || !pcl_isfinite (recog_model->cloud_->points[i].z))
          continue;

        if (!pcl_isfinite (model_normals->points[i].normal_x) || !pcl_isfinite (model_normals->points[i].normal_y)
            || !pcl_isfinite (model_normals->points[i].normal_z))
            continue
            ;
        recog_model->cloud_->points[j] = recog_model->cloud_->points[i];
        recog_model->normals_->points[j] = model_normals->points[i];
        j++;
      }

      recog_model->cloud_->points.resize (j);
      recog_model->cloud_->width = j;
      recog_model->cloud_->height = 1;

      recog_model->normals_->points.resize (j);
      recog_model->normals_->width = j;
      recog_model->normals_->height = 1;

      if (recog_model->cloud_->points.size () <= 0)
      {
        PCL_WARN("The model cloud has no points..\n");
        return false;
      }

    }
    else
    {

      pcl::ScopeTime t("Computing normals and checking nans");

      int j = 0;
      for (size_t i = 0; i < recog_model->cloud_->points.size (); ++i)
      {
        if (!pcl_isfinite (recog_model->cloud_->points[i].x) || !pcl_isfinite (recog_model->cloud_->points[i].y)
            || !pcl_isfinite (recog_model->cloud_->points[i].z))
          continue;

        recog_model->cloud_->points[j] = recog_model->cloud_->points[i];
        j++;
        //std::cout << "there are nans..." << std::endl;
      }

      recog_model->cloud_->points.resize (j);
      recog_model->cloud_->width = j;
      recog_model->cloud_->height = 1;

      if (recog_model->cloud_->points.size () <= 0)
      {
        PCL_WARN("The model cloud has no points..\n");
        return false;
      }

      {
        //compute normals unless given (now do it always...)
        typename pcl::search::KdTree<ModelT>::Ptr normals_tree (new pcl::search::KdTree<ModelT>);
        typedef typename pcl::NormalEstimationOMP<ModelT, pcl::Normal> NormalEstimator_;
        NormalEstimator_ n3d;
        recog_model->normals_.reset (new pcl::PointCloud<pcl::Normal> ());
        normals_tree->setInputCloud (recog_model->cloud_);
        n3d.setRadiusSearch (radius_normals_);
        n3d.setSearchMethod (normals_tree);
        n3d.setInputCloud ((recog_model->cloud_));
        n3d.compute (*(recog_model->normals_));

        //check nans...
        int j = 0;
        for (size_t i = 0; i < recog_model->normals_->points.size (); ++i)
        {
          if (!pcl_isfinite (recog_model->normals_->points[i].normal_x) || !pcl_isfinite (recog_model->normals_->points[i].normal_y)
              || !pcl_isfinite (recog_model->normals_->points[i].normal_z))
            continue;

          recog_model->normals_->points[j] = recog_model->normals_->points[i];
          recog_model->cloud_->points[j] = recog_model->cloud_->points[i];
          j++;
        }

        recog_model->normals_->points.resize (j);
        recog_model->normals_->width = j;
        recog_model->normals_->height = 1;

        recog_model->cloud_->points.resize (j);
        recog_model->cloud_->width = j;
        recog_model->cloud_->height = 1;
      }
    }

    return true;
}

  template<typename ModelT, typename SceneT>
  bool
  faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::addModel (int i, boost::shared_ptr<RecognitionModel<ModelT> > & recog_model)
  {

    int model_id = i;
    bool is_planar_model = false;
    std::map<int, int>::iterator it1;
    it1 = model_to_planar_model_.find(model_id);
    if(it1 != model_to_planar_model_.end())
    is_planar_model = true;

    if (normals_for_visibility_.size () == complete_models_.size ())
    {
      pcl::ScopeTime t("normal for visibililty");
      //pcl::PointCloud<pcl::Normal>::Ptr filtered_normals (new pcl::PointCloud<pcl::Normal> ());
      //pcl::copyPointCloud (*normals_for_visibility_[i], visible_indices_[i], *filtered_normals);
      //assert(filtered_normals->points.size() == visible_models_[i]->points.size());

      std::vector<int> keep;
      for (size_t k = 0; k < visible_models_[i]->points.size (); k++)
      {
        Eigen::Vector3f normal_p = normals_for_visibility_[i]->points[visible_indices_[i][k]].getNormalVector3fMap ();
        //Eigen::Vector3f normal_vp = visible_models_[i]->points[k].getVector3fMap () * -1.f;
        Eigen::Vector3f normal_vp = Eigen::Vector3f::UnitZ() * -1.f;

        normal_p.normalize ();
        normal_vp.normalize ();

        if (normal_p.dot (normal_vp) > 0.2f)
          keep.push_back (static_cast<int> (k));
      }

      recog_model->cloud_.reset (new pcl::PointCloud<ModelT> ());
      pcl::copyPointCloud (*visible_models_[i], keep, *recog_model->cloud_);

    }
    else
    {
      recog_model->cloud_.reset (new pcl::PointCloud<ModelT> (*visible_models_[i]));
      //recog_model->cloud_ = visible_models_[i];
    }

    recog_model->complete_cloud_.reset (new pcl::PointCloud<ModelT> (*complete_models_[i]));

    size_t object_models_size = complete_models_.size() - planar_models_.size();
    float extra_weight = 1.f;
    if(extra_weights_.size() != 0 && (extra_weights_.size() == object_models_size))
      extra_weight = extra_weights_[i];

    if(object_ids_.size() == complete_models_.size()) {
      recog_model->id_s_ = object_ids_[i];
    }

    bool handling_normals_b = handlingNormals(recog_model, i, is_planar_model, object_models_size);
    if(!handling_normals_b)
        return false;

    //pcl::ScopeTime tt_nn("Computing outliers and explained points...");
    std::vector<int> explained_indices;
    std::vector<float> outliers_weight;
    std::vector<float> explained_indices_distances;
    std::vector<float> unexplained_indices_weights;

    std::vector<int> nn_indices;
    std::vector<float> nn_distances;

    //which point first from the scene is explained by a point j_k with dist d_k from the model
    std::map<int, boost::shared_ptr<std::vector<std::pair<int, float> > > > model_explains_scene_points;
    std::map<int, boost::shared_ptr<std::vector<std::pair<int, float> > > > model_explains_scene_points_color_weight;
    std::map<int, boost::shared_ptr<std::vector<std::pair<int, float> > > >::iterator it;

    outliers_weight.resize (recog_model->cloud_->points.size ());
    recog_model->outlier_indices_.resize (recog_model->cloud_->points.size ());
    recog_model->outliers_3d_indices_.resize (recog_model->cloud_->points.size ());
    recog_model->color_outliers_indices_.resize (recog_model->cloud_->points.size ());
    recog_model->scene_point_explained_by_hypothesis_.resize(scene_cloud_downsampled_->points.size(), false);

    /*typename pcl::octree::OctreePointCloudSearch<SceneT> octree (0.01f);
    octree.setInputCloud (scene_cloud_downsampled_);
    octree.addPointsFromInputCloud ();*/

    /* (0.01f);
    if(occ_edges_available_)
    {
        octree_occ_edges.setInputCloud (occ_edges_);
        octree_occ_edges.addPointsFromInputCloud ();
    }*/

    /*pcl::visualization::PCLVisualizer vis("TEST");
    int v1,v2, v3;
    vis.addCoordinateSystem(2);
    vis.createViewPort(0,0,0.33,1,v1);
    vis.createViewPort(0.33,0,0.66,1,v2);
    vis.createViewPort(0.66,0,1,1,v3);
    vis.addPointCloud<SceneT>(occlusion_cloud_, "scene", v1);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*recog_model->cloud_, *model_cloud);

    vis.addPointCloud<pcl::PointXYZRGB>(model_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(model_cloud), "model");
    vis.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(model_cloud, recog_model->normals_, 10, 0.01, "normals", v3);
    //vis.addPointCloud<pcl::PointXYZRGB>(model_cloud_gs_specified, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(model_cloud_gs_specified), "model_specified", v3);
    vis.spin();*/

    //TESTING: gather explained points and compute RGB histograms to specify them afterwards
    Eigen::MatrixXf lookup;

    if(!is_planar_model && !ignore_color_even_if_exists_ && use_histogram_specification_)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud_specified(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*recog_model->cloud_, *model_cloud_specified);

        float rgb_m, rgb_s;
        bool exists_m;
        bool exists_s;

        typedef pcl::PointCloud<ModelT> CloudM;
        typedef pcl::PointCloud<SceneT> CloudS;
        typedef typename pcl::traits::fieldList<typename CloudS::PointType>::type FieldListS;
        typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

        std::set<int> explained_indices_points;

        for (size_t i = 0; i < recog_model->cloud_->points.size (); i++)
        {
          if (octree_scene_downsampled_->radiusSearch (recog_model->cloud_->points[i], inliers_threshold_, nn_indices, nn_distances,
                                                      std::numeric_limits<int>::max ()) > 0)
          {
            for (size_t k = 0; k < nn_distances.size (); k++)
            {
                explained_indices_points.insert(nn_indices[k]);
            }
          }
        }

        std::vector<Eigen::Vector3f> model_rgb_values, scene_rgb_values;
        std::vector<float> model_gs_values, scene_gs_values;

        //compute RGB histogram for the model points
        for (size_t i = 0; i < recog_model->cloud_->points.size (); i++)
        {
            pcl::for_each_type<FieldListM> (
                                            pcl::CopyIfFieldExists<typename CloudM::PointType, float> (
                                                                                                       recog_model->cloud_->points[i],
                                                                                                       "rgb", exists_m, rgb_m));
            if (exists_m)
            {
              uint32_t rgb = *reinterpret_cast<int*> (&rgb_m);
              uint8_t rm = (rgb >> 16) & 0x0000ff;
              uint8_t gm = (rgb >> 8) & 0x0000ff;
              uint8_t bm = (rgb) & 0x0000ff;

              float LRefm, aRefm, bRefm;

              RGB2CIELAB (rm, gm, bm, LRefm, aRefm, bRefm); //this is called in parallel and initially fill values on static thing...
              LRefm /= 100.0f; aRefm /= 120.0f; bRefm /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

              model_rgb_values.push_back(Eigen::Vector3f(rm, gm, bm));
              //model_gs_values.push_back( (rm + gm + bm) / 3.f);
              model_gs_values.push_back(LRefm * 255.f);
            }
        }

        //compute RGB histogram for the explained points
        std::set<int>::iterator it;
        for(it=explained_indices_points.begin(); it != explained_indices_points.end(); it++)
        {
            pcl::for_each_type<FieldListS> (
                        pcl::CopyIfFieldExists<typename CloudS::PointType, float> (scene_cloud_downsampled_->points[*it],
                                                                                                       "rgb", exists_s, rgb_s));

            if(exists_s)
            {
                uint32_t rgb = *reinterpret_cast<int*> (&rgb_s);
                uint8_t rs = (rgb >> 16) & 0x0000ff;
                uint8_t gs = (rgb >> 8) & 0x0000ff;
                uint8_t bs = (rgb) & 0x0000ff;

                float LRefs, aRefs, bRefs;
                RGB2CIELAB (rs, gs, bs, LRefs, aRefs, bRefs);
                LRefs /= 100.0f; aRefs /= 120.0f; bRefs /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

                scene_rgb_values.push_back(Eigen::Vector3f(rs, gs, bs));
                //scene_gs_values.push_back( (rs + gs + bs) / 3.f);
                scene_gs_values.push_back(LRefs * 255.f);
            }
        }

        {
            Eigen::MatrixXf gs_model, gs_scene;
            computeGSHistogram(model_gs_values, gs_model);
            computeGSHistogram(scene_gs_values, gs_scene);

            //histogram specification, adapt model values to scene values
            specifyRGBHistograms(gs_scene, gs_model, lookup, 1);

            /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud_gs(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::copyPointCloud(*recog_model->cloud_, *model_cloud_gs);
            for(size_t k=0; k < model_cloud_gs->points.size(); k++)
            {
                float gs = (model_cloud_gs->points[k].r + model_cloud_gs->points[k].g + model_cloud_gs->points[k].b) / 3.f;
                gs = model_gs_values[k];
                model_cloud_gs->points[k].r =
                model_cloud_gs->points[k].g =
                model_cloud_gs->points[k].b = gs;
            }

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud_gs_specified(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::copyPointCloud(*recog_model->cloud_, *model_cloud_gs_specified);
            for(size_t k=0; k < model_cloud_gs_specified->points.size(); k++)
            {
                float gs = (model_cloud_gs_specified->points[k].r + model_cloud_gs_specified->points[k].g + model_cloud_gs_specified->points[k].b) / 3.f;
                gs = model_gs_values[k];
                int pos = std::floor (static_cast<float> (gs) / 255.f * 256);
                float gs_specified = lookup(pos, 0);
                model_cloud_gs_specified->points[k].r =
                model_cloud_gs_specified->points[k].g =
                model_cloud_gs_specified->points[k].b = gs_specified;
            }

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::copyPointCloud(*occlusion_cloud_, *scene_cloud);
            for(size_t k=0; k < scene_cloud->points.size(); k++)
            {
                //float gs = (scene_cloud->points[k].r + scene_cloud->points[k].g + scene_cloud->points[k].b) / 3.f;
                uint32_t rgb = *reinterpret_cast<int*> (&scene_cloud->points[k].rgb);
                uint8_t rs = (rgb >> 16) & 0x0000ff;
                uint8_t gs = (rgb >> 8) & 0x0000ff;
                uint8_t bs = (rgb) & 0x0000ff;

                float LRefs, aRefs, bRefs;
                RGB2CIELAB (rs, gs, bs, LRefs, aRefs, bRefs);
                LRefs /= 100.0f; aRefs /= 120.0f; bRefs /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

                scene_cloud->points[k].r =
                scene_cloud->points[k].g =
                scene_cloud->points[k].b = LRefs * 255.f;
            }

            pcl::visualization::PCLVisualizer vis("TEST");
            int v1,v2, v3;
            vis.createViewPort(0,0,0.33,1,v1);
            vis.createViewPort(0.33,0,0.66,1,v2);
            vis.createViewPort(0.66,0,1,1,v3);
            vis.addPointCloud<pcl::PointXYZRGB>(scene_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(scene_cloud), "scene", v1);
            vis.addPointCloud<pcl::PointXYZRGB>(model_cloud_gs, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(model_cloud_gs), "model", v2);
            vis.addPointCloud<pcl::PointXYZRGB>(model_cloud_gs_specified, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(model_cloud_gs_specified), "model_specified", v3);
            vis.spin();*/

        }

        /*int rgb_size_hist = 256;
        Eigen::MatrixXf rgb_model, rgb_scene;
        computeRGBHistograms(model_rgb_values, rgb_model, rgb_size_hist);
        computeRGBHistograms(scene_rgb_values, rgb_scene, rgb_size_hist);

        //histogram specification, adapt model values to scene values
        Eigen::MatrixXf lookup;
        specifyRGBHistograms(rgb_scene, rgb_model, lookup);

        //with the lookup table, we can now transform model color space to scene model space
        for(size_t j=0; j < 3; j++)
        {
            for(size_t k=0; k < model_rgb_values.size(); k++)
            {
              int pos = std::floor (static_cast<float> (model_rgb_values[k][j]) / 255.f * rgb_size_hist);
              model_rgb_values[k][j] = lookup(pos, j);
            }
        }

        std::cout << model_cloud_specified->points.size() << " " << model_rgb_values.size() << std::endl;
        for(size_t k=0; k < model_cloud_specified->points.size(); k++)
        {
            model_cloud_specified->points[k].r = model_rgb_values[k][0];
            model_cloud_specified->points[k].g = model_rgb_values[k][1];
            model_cloud_specified->points[k].b = model_rgb_values[k][2];
        }

        //if(recog_model->id_s_.compare("object_26.pcd") == 0)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::copyPointCloud(*recog_model->cloud_, *model_cloud);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::copyPointCloud(*occlusion_cloud_, *scene_cloud);

            pcl::visualization::PCLVisualizer vis("TEST");
            int v1,v2, v3;
            vis.createViewPort(0,0,0.33,1,v1);
            vis.createViewPort(0.33,0,0.66,1,v2);
            vis.createViewPort(0.66,0,1,1,v3);
            vis.addPointCloud<pcl::PointXYZRGB>(scene_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(scene_cloud), "scene", v1);
            vis.addPointCloud<pcl::PointXYZRGB>(model_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(model_cloud), "model", v2);
            vis.addPointCloud<pcl::PointXYZRGB>(model_cloud_specified, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(model_cloud_specified), "model_specified", v3);
            vis.spin();
        }*/

        //pcl::copyPointCloud(*model_cloud_specified, *recog_model->cloud_);
    }

    {
      //pcl::ScopeTime t("NN");
      size_t o = 0;
      int o_color = 0;
      int o_3d = 0;
      //Goes through the visible model points and finds scene points within a radius neighborhood
      //If in this neighborhood, there are no scene points, model point is considered outlier
      //If there are scene points, the model point is associated with the scene point, together with its distance
      //A scene point might end up being explained by the multiple model points

      float rgb_m, rgb_s;
      bool exists_m;
      bool exists_s;

      typedef pcl::PointCloud<ModelT> CloudM;
      typedef pcl::PointCloud<SceneT> CloudS;
      typedef typename pcl::traits::fieldList<typename CloudS::PointType>::type FieldListS;
      typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

      /*pcl::visualization::PCLVisualizer vis("model points with outliers");
      int v1, v2, v3, v4;
      vis.createViewPort(0,0,0.25,1,v1);
      vis.createViewPort(0.25,0,0.5,1,v2);
      vis.createViewPort(0.5,0,0.75,1,v3);
      vis.createViewPort(0.75,0,1,1,v4);

      vis.addPointCloud(recog_model->cloud_, "model cloud", v1);
      vis.addPointCloud(scene_cloud_downsampled_, "scene cloud", v2);
      vis.addPointCloud(occ_edges_, "occ_edges", v4);*/

      int bad_normals = 0;
      for (size_t i = 0; i < recog_model->cloud_->points.size (); i++)
      {
        bool outlier = false;
        int outlier_type = 0;

        /*if (scene_downsampled_tree_->radiusSearch (recog_model->cloud_->points[i], inliers_threshold_, nn_indices, nn_distances,
                                                    std::numeric_limits<int>::max ()) > 0)*/

        if (octree_scene_downsampled_->radiusSearch (recog_model->cloud_->points[i], inliers_threshold_, nn_indices, nn_distances,
                                                    std::numeric_limits<int>::max ()) > 0)
        {

          //std::vector<bool> outliers(nn_distances.size(), false);

          std::vector<float> weights;
          for (size_t k = 0; k < nn_distances.size () && !is_planar_model; k++)
          {
            //check color
            if (!ignore_color_even_if_exists_)
            {
              pcl::for_each_type<FieldListM> (
                                              pcl::CopyIfFieldExists<typename CloudM::PointType, float> (
                                                                                                         recog_model->cloud_->points[i],
                                                                                                         "rgb", exists_m, rgb_m));
              pcl::for_each_type<FieldListS> (
                                              pcl::CopyIfFieldExists<typename CloudS::PointType, float> (scene_cloud_downsampled_->points[nn_indices[k]],
                                                                                                         "rgb", exists_s, rgb_s));

              //std::cout << "color exists:" << exists_m << " " << exists_s << std::endl;
              if (exists_m && exists_s)
              {
                uint32_t rgb = *reinterpret_cast<int*> (&rgb_m);
                unsigned char rm = (rgb >> 16) & 0x0000ff;
                unsigned char gm = (rgb >> 8) & 0x0000ff;
                unsigned char bm = (rgb) & 0x0000ff;

                rgb = *reinterpret_cast<int*> (&rgb_s);
                unsigned char rs = (rgb >> 16) & 0x0000ff;
                unsigned char gs = (rgb >> 8) & 0x0000ff;
                unsigned char bs = (rgb) & 0x0000ff;

                float sigma = 2.f * color_sigma_ * color_sigma_;
                float sigma_y = 8.f * color_sigma_ * color_sigma_;
                Eigen::Vector3f yuvm, yuvs;

                float LRefm, aRefm, bRefm;
                float LRefs, aRefs, bRefs;
                RGB2CIELAB (rm, gm, bm, LRefm, aRefm, bRefm); //this is called in parallel and initially fill values on static thing...
                LRefm /= 100.0f; aRefm /= 120.0f; bRefm /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

                RGB2CIELAB (rs, gs, bs, LRefs, aRefs, bRefs);
                LRefs /= 100.0f; aRefs /= 120.0f; bRefs /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

                if(use_histogram_specification_)
                {
                    LRefm *= 255.f;
                    int pos = std::floor (static_cast<float> (LRefm) / 255.f * 256);
                    float gs_specified = lookup(pos, 0);
                    LRefm = gs_specified / 255.f;
                }

                yuvm = Eigen::Vector3f (static_cast<float> (LRefm), static_cast<float> (aRefm), static_cast<float> (bRefm));
                yuvs = Eigen::Vector3f (static_cast<float> (LRefs), static_cast<float> (aRefs), static_cast<float> (bRefs));

                float color_weight = std::exp ((-0.5f * (yuvm[0] - yuvs[0]) * (yuvm[0] - yuvs[0])) / (sigma));
                color_weight *= std::exp ((-0.5f * (yuvm[1] - yuvs[1]) * (yuvm[1] - yuvs[1])) / (sigma));
                color_weight *= std::exp ((-0.5f * (yuvm[2] - yuvs[2]) * (yuvm[2] - yuvs[2])) / (sigma));

                /*float color_weight2;
                {
                    yuvm = Eigen::Vector3f (static_cast<float> (LRefm), static_cast<float> (aRefm), static_cast<float> (bRefm));
                    yuvs = Eigen::Vector3f (static_cast<float> (LRefs), static_cast<float> (aRefs), static_cast<float> (bRefs));
                    float color_weight = std::exp ((-0.5f * (yuvm[0] - yuvs[0]) * (yuvm[0] - yuvs[0])) / (sigma));
                    color_weight *= std::exp ((-0.5f * (yuvm[1] - yuvs[1]) * (yuvm[1] - yuvs[1])) / (sigma));
                    color_weight *= std::exp ((-0.5f * (yuvm[2] - yuvs[2]) * (yuvm[2] - yuvs[2])) / (sigma));
                    color_weight2 = color_weight;
                }
                color_weight = std::max(color_weight, color_weight2);*/

                weights.push_back(color_weight);
              }
            }
          }

          //std::cout << nn_distances.size() << " " << weights.size() << " " << ignore_color_even_if_exists_ << std::endl;
          std::vector<float> weights_not_sorted = weights;

          std::sort(weights.begin(), weights.end(), std::greater<float>());

          if(is_planar_model || ignore_color_even_if_exists_ || weights[0] > 0.8f) //best weight is not an outlier
          {
            for (size_t k = 0; k < nn_distances.size (); k++)
            {
              std::pair<int, float> pair = std::make_pair (i, nn_distances[k]); //i is a index to a model point and then distance

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

              if(!is_planar_model && !ignore_color_even_if_exists_)
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
            recog_model->color_outliers_indices_[o_color] = static_cast<int> (i);
            outlier = true;
            o_color++;
            outlier_type = 1;
          }
        }
        else
        {
          recog_model->outliers_3d_indices_[o_3d] = static_cast<int> (i);
          outlier = true;
          o_3d++;
        }

        if(outlier)
        {
          //weight outliers based on noise model
          //model points close to occlusion edges or with perpendicular normals
            float d_weight = 1.f;

            if(!is_planar_model && occ_edges_available_)
            {

                  //std::cout << "is not planar model" << std::endl;
                  std::vector<int> pointIdxNKNSearch;
                  std::vector<float> pointNKNSquaredDistance;

                  pcl::PointXYZ p;
                  p.getVector3fMap() = recog_model->cloud_->points[i].getVector3fMap();
                  if (octree_occ_edges_->nearestKSearch (p, 1,
                                                       pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                  {
                      float dist = sqrt(pointNKNSquaredDistance[0]);
                      if(dist < inliers_threshold_)
                      {
                          d_weight = 0.5f;
                          //vis.addSphere(p, 0.02, "sphere", v1);
                          //vis.spin();
                          //vis.removeAllShapes();

                          if(outlier_type)
                              o_color--;
                          else
                              o_3d--;
                      }
                      else
                      {
                          //check for normal
                          /*Eigen::Vector3f plane_normal = recog_model->cloud_->points[i].getVector3fMap() * -1.f;
                          Eigen::Vector3f z_vector = Eigen::Vector3f::UnitZ ();
                          plane_normal.normalize ();
                          Eigen::Vector3f axis = plane_normal.cross (z_vector);
                          double rotation = -asin (axis.norm ());
                          axis.normalize ();

                          Eigen::Affine3f transformPC (Eigen::AngleAxisf (static_cast<float> (rotation), axis));*/
                          //Eigen::Vector3f normal_p = transformPC * recog_model->normals_->points[i].getNormalVector3fMap();
                          Eigen::Vector3f normal_p = recog_model->normals_->points[i].getNormalVector3fMap();
                          Eigen::Vector3f normal_vp = Eigen::Vector3f::UnitZ() * -1.f;
                          //Eigen::Vector3f normal_vp = recog_model->cloud_->points[i].getVector3fMap() * -1.f;
                          normal_p.normalize ();
                          normal_vp.normalize ();

                          float dot = normal_vp.dot(normal_p);
                          float angle = pcl::rad2deg(acos(dot));
                          //std::cout << "lolo:" << normal_p << std::endl;
                          if (angle > 60.f)
                          //if(dot < 0.75f)
                          {
                              if(outlier_type)
                                  o_color--;
                              else
                                  o_3d--;

                              d_weight = 0.1f;
                              bad_normals++;
                          }
                      }
                  }

                      /*float f = 575.f;
                      float cx = 320.f;
                      float cy = 240.f;

                      pcl::PointXYZ p;
                      p.getVector3fMap() = recog_model->cloud_->points[i].getVector3fMap();
                      //project model point on occlusion edges cloud and check if nan or not
                      int u = static_cast<int> (f * p.x / p.z + cx);
                      int v = static_cast<int> (f * p.y / p.z + cy);

                      if ((u >= static_cast<int> (occ_edges_->width)) ||
                          (v >= static_cast<int> (occ_edges_->height)) || (u < 0) || (v < 0))
                      {

                      }
                      else
                      {
                          float z = occ_edges_->at (u, v).z;
                          //std::cout << z << " " << u << " " << v << std::endl;
                          if (!pcl_isnan(z))
                          {
                              bad_normals++;
                              d_weight = 0.5f;
                              if(outlier_type)
                                  o_color--;
                              else
                                  o_3d--;
                          }
                      }*/
              //}
            }

            outliers_weight[o] = regularizer_ * d_weight;
            recog_model->outlier_indices_[o] = static_cast<int> (i);
            o++;
        }
      }

      outliers_weight.resize (o);
      recog_model->outlier_indices_.resize (o);
      recog_model->outliers_3d_indices_.resize (o_3d);
      recog_model->color_outliers_indices_.resize (o_color);
    }

    /*{
      //pcl::ScopeTime t("NN");
      size_t o = 0;
      //Goes through the visible model points and finds scene points within a radius neighborhood
      //If in this neighborhood, there are no scene points, model point is considered outlier
      //If there are scene points, the model point is associated with the scene point, together with its distance
      //A scene point might end up being explained by the multiple model points
      for (size_t i = 0; i < recog_model->cloud_->points.size (); i++)
      {
        //if (!scene_downsampled_tree_->radiusSearch (recog_model->cloud_->points[i], inliers_threshold_, nn_indices, nn_distances,
        //                                            std::numeric_limits<int>::max ()))

        if (!octree.radiusSearch (recog_model->cloud_->points[i], inliers_threshold_, nn_indices, nn_distances,
                                                    std::numeric_limits<int>::max ()))
        {

          //get NN
          std::vector<int> nn_indices_outlier;
          std::vector<float> nn_distances_outlier;
          //scene_downsampled_tree_->nearestKSearch(recog_model->cloud_->points[i], 1, nn_indices_outlier, nn_distances_outlier);
          octree.nearestKSearch(recog_model->cloud_->points[i], 1, nn_indices_outlier, nn_distances_outlier);
          float d = sqrt(nn_distances_outlier[0]);

          //if(d > (inliers_threshold_ * 2))
          //{
          float d_weight = 1.f + ( ( (d-inliers_threshold_) / inliers_threshold_));
          //outlier
          //float d_weight = 1.f;
          outliers_weight[o] = regularizer_ * d_weight;
          recog_model->outlier_indices_[o] = static_cast<int> (i);
          o++;
          //}
        }
        else
        {
          for (size_t k = 0; k < nn_distances.size (); k++)
          {
            std::pair<int, float> pair = std::make_pair (i, nn_distances[k]); //i is a index to a model point and then distance
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
          }
        }
      }

      outliers_weight.resize (o);
      recog_model->outlier_indices_.resize (o);
    }*/

    //using mean
    recog_model->outliers_weight_ = (std::accumulate (outliers_weight.begin (), outliers_weight.end (), 0.f) / static_cast<float> (outliers_weight.size ()));
    //using median
    //std::sort(outliers_weight.begin(), outliers_weight.end());
    //recog_model->outliers_weight_ = outliers_weight[outliers_weight.size() / 2.f];

    if (outliers_weight.size () == 0)
      recog_model->outliers_weight_ = 1.f;

    int p = 0;

    //explained_indices.resize (model_explains_scene_points.size());
    //explained_indices_distances.resize (model_explains_scene_points.size());

    //go through the map and keep the closest model point in case that several model points explain a scene point
    std::vector<Eigen::Vector3f> model_hsv_values, scene_hsv_values;
    std::vector<Eigen::Vector3f> model_rgb_values, scene_rgb_values;
    std::vector<int> model_indices_explained;

    for (it = model_explains_scene_points.begin (); it != model_explains_scene_points.end (); it++, p++)
    {
      size_t closest = 0;
      float min_d = std::numeric_limits<float>::min ();
      for (size_t i = 0; i < it->second->size (); i++)
      {
        if (it->second->at (i).second > min_d)
        {
          min_d = it->second->at (i).second;
          closest = i;
        }
      }

      float d = it->second->at (closest).second;
      float d_weight = -(d * d / (inliers_threshold_)) + 1;

      //it->first is index to scene point
      //using normals to weight inliers
      Eigen::Vector3f scene_p_normal = scene_normals_->points[it->first].getNormalVector3fMap ();
      Eigen::Vector3f model_p_normal = recog_model->normals_->points[it->second->at (closest).first].getNormalVector3fMap ();
      float dotp = scene_p_normal.dot (model_p_normal) * 1.f; //[-1,1] from antiparallel trough perpendicular to parallel

      if (dotp < 0.f)
        dotp = 0.f;
      //else
        //dotp = 1.f; //ATTENTION: Deactivated normal weight!

      //assert(std::abs(scene_normals_->points[it->first].getNormalVector3fMap().norm()) < 1.1f);
      //assert(std::abs(dotp) < 1.1f);

      if (!is_planar_model && !ignore_color_even_if_exists_)
      {
        std::map<int, boost::shared_ptr<std::vector<std::pair<int, float> > > >::iterator it_color;
        it_color = model_explains_scene_points_color_weight.find(it->first);
        if(it != model_explains_scene_points_color_weight.end())
        {
          d_weight *= it_color->second->at(closest).second;
        }
      }

      /*float color_weight = 1.f;

      if (!ignore_color_even_if_exists_)
      {
        float rgb_m, rgb_s;
        bool exists_m;
        bool exists_s;

        typedef pcl::PointCloud<ModelT> CloudM;
        typedef pcl::PointCloud<SceneT> CloudS;
        typedef typename pcl::traits::fieldList<typename CloudS::PointType>::type FieldListS;
        typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

        pcl::for_each_type<FieldListM> (
                                        pcl::CopyIfFieldExists<typename CloudM::PointType, float> (
                                                                                                   recog_model->cloud_->points[it->second->at (
                                                                                                                                               closest).first],
                                                                                                   "rgb", exists_m, rgb_m));
        pcl::for_each_type<FieldListS> (
                                        pcl::CopyIfFieldExists<typename CloudS::PointType, float> (scene_cloud_downsampled_->points[it->first],
                                                                                                   "rgb", exists_s, rgb_s));

        if (exists_m && exists_s)
        {
          color_exist = true;
          uint32_t rgb = *reinterpret_cast<int*> (&rgb_m);
          uint8_t rm = (rgb >> 16) & 0x0000ff;
          uint8_t gm = (rgb >> 8) & 0x0000ff;
          uint8_t bm = (rgb) & 0x0000ff;

          rgb = *reinterpret_cast<int*> (&rgb_s);
          uint8_t rs = (rgb >> 16) & 0x0000ff;
          uint8_t gs = (rgb >> 8) & 0x0000ff;
          uint8_t bs = (rgb) & 0x0000ff;

          model_rgb_values.push_back(Eigen::Vector3f(rm, gm, bm));
          scene_rgb_values.push_back(Eigen::Vector3f(rs, gs, bs));

          map_cloud_points_to_model_rgb_values.push_back(it->second->at (closest).first);
          model_indices_explained.push_back(it->second->at (closest).first);
        }
      }
      else
      {
        color_exist = false;
      }*/

      explained_indices.push_back (it->first);
      explained_indices_distances.push_back (d_weight * dotp * extra_weight);
      recog_model->scene_point_explained_by_hypothesis_[it->first] = true; //this scene point is explained by this hypothesis
    }

    recog_model->model_constraints_value_ = getModelConstraintsValue(recog_model->complete_cloud_);
    recog_model->bad_information_ =  static_cast<int> (recog_model->outlier_indices_.size ());
    //recog_model->bad_information_ = 0;

    /*if (color_exist)
    {

//      int rgb_size_hist = 256;
//      Eigen::MatrixXf rgb_model, rgb_scene;
//      computeRGBHistograms(model_rgb_values, rgb_model, rgb_size_hist);
//      computeRGBHistograms(scene_rgb_values, rgb_scene, rgb_size_hist);
//
//      //histogram specification, adapt model values to scene values
//      Eigen::MatrixXf lookup;
//      specifyRGBHistograms(rgb_scene, rgb_model, lookup);
//
//      //with the lookup table, we can now transform model color space to scene model space
//      for(size_t j=0; j < 3; j++) {
//        for(size_t i=0; i < model_rgb_values.size(); i++)
//        {
//          int pos = std::floor (static_cast<float> (model_rgb_values[i][j]) / 255.f * rgb_size_hist);
//          //model_rgb_values[i][j] = lookup(pos, j);
//        }
//      }

      for(size_t i=0; i < model_rgb_values.size(); i++)
      {
        Eigen::Vector3f yuvm, yuvs;

        float ym = 0.257f * model_rgb_values[i][0] + 0.504f * model_rgb_values[i][1] + 0.098f * model_rgb_values[i][2] + 16; //between 16 and 235
        float um = -(0.148f * model_rgb_values[i][0]) - (0.291f * model_rgb_values[i][1]) + (0.439f * model_rgb_values[i][2]) + 128;
        float vm = (0.439f * model_rgb_values[i][0]) - (0.368f * model_rgb_values[i][1]) - (0.071f * model_rgb_values[i][2]) + 128;

        float ys = 0.257f * scene_rgb_values[i][0] + 0.504f * scene_rgb_values[i][1] + 0.098f * scene_rgb_values[i][2] + 16;
        float us = -(0.148f * scene_rgb_values[i][0]) - (0.291f * scene_rgb_values[i][1]) + (0.439f * scene_rgb_values[i][2]) + 128;
        float vs = (0.439f * scene_rgb_values[i][0]) - (0.368f * scene_rgb_values[i][1]) - (0.071f * scene_rgb_values[i][2]) + 128;

        yuvm = Eigen::Vector3f (static_cast<float> (ym), static_cast<float> (um), static_cast<float> (vm));
        yuvs = Eigen::Vector3f (static_cast<float> (ys), static_cast<float> (us), static_cast<float> (vs));

        assert(yuvm[0] >= 16 && yuvm[0] <= 235);
        model_hsv_values.push_back(yuvm);
        scene_hsv_values.push_back(yuvs);

//        float sigma = color_sigma_ * color_sigma_;
//        yuvm[0] *= 0.5f;
//        yuvs[0] *= 0.5f;
//        float color_weight = std::exp ((-0.5f * (yuvm - yuvs).squaredNorm ()) / (sigma));
//        color_weight_inliers_vector[i] = color_weight;
//
//        //assert(color_weight >= 0);
//        //assert(color_weight <= 1);
//        color_weight_inliers += color_weight;
      }

      Eigen::MatrixXf yuv_model, yuv_scene;
      computeRGBHistograms(model_hsv_values, yuv_model, 1, 16, 235);
      computeRGBHistograms(scene_hsv_values, yuv_scene, 1, 16, 235);

      Eigen::MatrixXf lookup_yuv;
      specifyRGBHistograms(yuv_scene, yuv_model, lookup_yuv, 1);
      recog_model->color_mapping_ = lookup_yuv;

      //with the lookup table, we can now transform model color space to scene model space
      for(size_t j=0; j < 1; j++) {
        for(size_t i=0; i < model_hsv_values.size(); i++)
        {
          int pos = std::floor (static_cast<float> (model_hsv_values[i][j] - 16) / (235.f - 16.f) * (235 - 16));
          //std::cout << pos << std::endl;
          assert(pos >= 0 && pos < (lookup_yuv.rows()));
          //std::cout << lookup_yuv(pos, j) << std::endl;
          model_hsv_values[i][j] = lookup_yuv(pos, j) + 16;
          assert(model_hsv_values[i][j] >= 16 && model_hsv_values[i][j] <= 235);
        }
      }

      color_weight_inliers_vector.resize(model_rgb_values.size());
      float sigma = color_sigma_ * color_sigma_;
      for(size_t i=0; i < model_rgb_values.size(); i++)
      {
        //model_hsv_values[i][1] *= 0.5f;
        //model_hsv_values[i][2] *= 0.5f;

        //scene_hsv_values[i][1] *= 0.5f;
        //scene_hsv_values[i][2] *= 0.5f;

        //float color_weight = std::exp ((-0.5f * (model_hsv_values[i] - scene_hsv_values[i]).squaredNorm ()) / (sigma));
        //float color_weight = std::exp ((-0.5f * (model_hsv_values[i][0] - scene_hsv_values[i][0]) * (model_hsv_values[i][0] - scene_hsv_values[i][0])) / (sigma));
        //color_weight *= std::exp ((-0.5f * (model_hsv_values[i][1] - scene_hsv_values[i][1]) * (model_hsv_values[i][1] - scene_hsv_values[i][1])) / (sigma));
        //color_weight *= std::exp ((-0.5f * (model_hsv_values[i][2] - scene_hsv_values[i][2]) * (model_hsv_values[i][2] - scene_hsv_values[i][2])) / (sigma));

        unsigned char rm = model_rgb_values[i][0];
        unsigned char gm = model_rgb_values[i][1];
        unsigned char bm = model_rgb_values[i][2];

        unsigned char rs = scene_rgb_values[i][0];
        unsigned char gs = scene_rgb_values[i][1];
        unsigned char bs = scene_rgb_values[i][2];

        float sigma = 2.f * color_sigma_ * color_sigma_;

        Eigen::Vector3f yuvm, yuvs;
        float LRefm, aRefm, bRefm;
        float LRefs, aRefs, bRefs;
        RGB2CIELAB (rm, gm, bm, LRefm, aRefm, bRefm); //this is called in parallel and initially fill values on static thing...
        LRefm /= 100.0f; aRefm /= 120.0f; bRefm /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

        RGB2CIELAB (rs, gs, bs, LRefs, aRefs, bRefs);
        LRefs /= 100.0f; aRefs /= 120.0f; bRefs /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

        yuvm = Eigen::Vector3f (static_cast<float> (LRefm), static_cast<float> (aRefm), static_cast<float> (bRefm));
        yuvs = Eigen::Vector3f (static_cast<float> (LRefs), static_cast<float> (aRefs), static_cast<float> (bRefs));

        float color_weight = std::exp ((-0.5f * (yuvm[0] - yuvs[0]) * (yuvm[0] - yuvs[0])) / (sigma));
        color_weight *= std::exp ((-0.5f * (yuvm[1] - yuvs[1]) * (yuvm[1] - yuvs[1])) / (sigma));
        color_weight *= std::exp ((-0.5f * (yuvm[2] - yuvs[2]) * (yuvm[2] - yuvs[2])) / (sigma));

        color_weight_inliers_vector[i] = color_weight;
        color_weight_inliers += color_weight;
      }

      model_hsv_values.resize(recog_model->cloud_->points.size());
      for(size_t i=0; i < recog_model->cloud_->points.size(); i++) {
        typedef pcl::PointCloud<ModelT> CloudM;
        typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

        float rgb_m;
        bool exists_m;;
        pcl::for_each_type<FieldListM> (
                                        pcl::CopyIfFieldExists<typename CloudM::PointType, float> (
                                                                                                   recog_model->cloud_->points[i],
                                                                                                   "rgb", exists_m, rgb_m));
        uint32_t rgb = *reinterpret_cast<int*> (&rgb_m);
        uint8_t rm = (rgb >> 16) & 0x0000ff;
        uint8_t gm = (rgb >> 8) & 0x0000ff;
        uint8_t bm = (rgb) & 0x0000ff;

        float ym = 0.257f * rm + 0.504f * gm + 0.098f * bm + 16; //between 16 and 235
        float um = -(0.148f * rm) - (0.291f * gm) + (0.439f * bm) + 128;
        float vm = (0.439f * rm) - (0.368f * gm) - (0.071f * bm) + 128;

        model_hsv_values[i] = Eigen::Vector3f (static_cast<float> (ym), static_cast<float> (um), static_cast<float> (vm));
        int pos = std::floor (static_cast<float> (model_hsv_values[i][0] - 16) / (235.f - 16.f) * (235 - 16));
        //model_hsv_values[i][0] = std::min(lookup_yuv(pos, 0), 50.f) + 16;
        //model_hsv_values[i][0] = lookup_yuv(pos, 0) + 16;
      }

      Eigen::VectorXf model_hist, scene_hist;
      //computeHueHistogram(model_hsv_values, model_hist);
      //computeHueHistogram(scene_hsv_values, scene_hist);

      computeYUVHistogram(model_hsv_values, model_hist);
      computeYUVHistogram(scene_hsv_values, scene_hist);

      float sum_s, sum_m;
      sum_m = sum_s = 0.f;

      for(size_t i=0; i < model_hist.size(); i++) {
        sum_s += scene_hist[i];
        sum_m += model_hist[i];
      }

      model_hist /= sum_m;
      scene_hist /= sum_s;

      //histogram intersection
      float color_similarity,sum_ma, sum_d, intersection;
      color_similarity = sum_ma = sum_d = sum_s = sum_m = intersection = 0.f;

      for(size_t i=0; i < model_hist.size(); i++) {
        float mi = std::min(scene_hist[i], model_hist[i]);
        color_similarity += mi;
      }

      std::sort(color_weight_inliers_vector.begin(), color_weight_inliers_vector.end());
      //color_weight_inliers_vector.erase(color_weight_inliers_vector.begin(), color_weight_inliers_vector.begin() + (0.1f * color_weight_inliers_vector.size()));

      float mean = std::accumulate(color_weight_inliers_vector.begin(), color_weight_inliers_vector.end(), 0.f) / static_cast<float>(color_weight_inliers_vector.size());
      float median = color_weight_inliers_vector[color_weight_inliers_vector.size() / 2];
      //float median4 = color_weight_inliers_vector[color_weight_inliers_vector.size() / 4];
      //PCL_WARN("histogram %f median %f mean %f\n", color_similarity, median, mean);
      //PCL_INFO("Explained indices: %d Outliers: %d \n", explained_indices.size(), recog_model->outlier_indices_.size());
      //recog_model->bad_information_ += (static_cast<float> (explained_indices.size() / 4) - median4 * static_cast<float> (explained_indices.size() / 4));
      //recog_model->bad_information_ += (static_cast<float> (explained_indices.size()) - (median * static_cast<float> (explained_indices.size())));

      //recog_model->bad_information_ *= 1.f + (1.f - median);
      //recog_model->bad_information_ *= 1.f + (1.f - mean);
      //recog_model->bad_information_ *= 1.f + (1.f - color_similarity);

      recog_model->color_similarity_ = color_similarity;
      recog_model->mean_ = mean;
      recog_model->median_ = median;

      median = std::min(mean, median);
      int sign = 1;
      if(median < 0.5f) {
        sign = -1;
      }

      //std::cout << median << " " << (1.f - std::exp ((-0.5f * (median - 0.5f) * (median - 0.5f)) / (0.5f * 0.5f))) * sign << std::endl;
      median += (1.f - std::exp ((-0.5f * (median - 0.5f) * (median - 0.5f)) / (0.5f * 0.5f))) * sign;

      //recog_model->outliers_weight_ *= 1.f + (1.f - median);
      //recog_model->model_constraints_value_ *= (1.f - std::min(median, 0.9f));
      for(size_t i=0; i < explained_indices_distances.size(); i++) {
        //explained_indices_distances[i] *= color_similarity;
        //explained_indices_distances[i] *= mean * color_similarity;
        explained_indices_distances[i] *= mean * color_similarity;
      }

      //recog_model->bad_information_ += (static_cast<float> (explained_indices.size()) * (1.f - color_similarity));
      //recog_model->bad_information_ += (static_cast<float> (explained_indices.size()) * (1.f - color_similarity)) / regularizer_;
      //recog_model->bad_information_ += (static_cast<float> (recog_model->cloud_->points.size()) * (1.f - color_similarity)) / regularizer_;
      //recog_model->bad_information_ += (static_cast<float> (recog_model->cloud_->points.size()) * (1.f - color_similarity));
    }*/


    //modify the explained weights for planar models if color is being used
    if(!ignore_color_even_if_exists_)
    {
      //check if color actually exists
      bool exists_s;
      float rgb_s;
      typedef pcl::PointCloud<SceneT> CloudS;
      typedef typename pcl::traits::fieldList<typename CloudS::PointType>::type FieldListS;

      pcl::for_each_type<FieldListS> (pcl::CopyIfFieldExists<typename CloudS::PointType, float> (scene_cloud_downsampled_->points[0],
                                                                                                 "rgb", exists_s, rgb_s));

      if (exists_s)
      {
        std::map<int, int>::iterator it1;
        it1 = model_to_planar_model_.find(static_cast<int>(i));
        if(it1 != model_to_planar_model_.end())
        {
          PCL_WARN("Plane found... decrease weight.\n");
          for(size_t k=0; k < explained_indices_distances.size(); k++)
          {
            explained_indices_distances[k] *= 0.5f;
          }
        }
      }
    }

    recog_model->explained_ = explained_indices;
    recog_model->explained_distances_ = explained_indices_distances;
    recog_model->id_ = i;
    //std::cout << "Model:" << recog_model->complete_cloud_->points.size() << " " << recog_model->cloud_->points.size() << std::endl;
    return true;
  }

    template<typename ModelT, typename SceneT>
    void
    faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::getOutliersForAcceptedModels(std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud)
    {
        for(size_t i=0; i < recognition_models_.size(); i++)
        {
            if(mask_[i])
            {
                pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_points(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::copyPointCloud(*(recognition_models_[i]->cloud_), recognition_models_[i]->outlier_indices_, *outlier_points);
                outliers_cloud.push_back(outlier_points);
            }
        }
    }

    template<typename ModelT, typename SceneT>
    void
    faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::getOutliersForAcceptedModels(std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud_color,
                                                                                           std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud_3d)
    {
        for(size_t i=0; i < recognition_models_.size(); i++)
        {
            if(mask_[i])
            {
                {
                    pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_points(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::copyPointCloud(*(recognition_models_[i]->cloud_), recognition_models_[i]->color_outliers_indices_, *outlier_points);
                    outliers_cloud_color.push_back(outlier_points);
                }

                {
                    pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_points(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::copyPointCloud(*(recognition_models_[i]->cloud_), recognition_models_[i]->outliers_3d_indices_, *outlier_points);
                    outliers_cloud_3d.push_back(outlier_points);
                }
            }
        }
    }

template<typename ModelT, typename SceneT>
  void
  faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::computeClutterCue (boost::shared_ptr<RecognitionModel<ModelT> > & recog_model)
  {

    /*int model_id = recog_model->id_;
    bool is_planar_model = false;
    std::map<int, int>::iterator it1;
    it1 = model_to_planar_model_.find(model_id);
    if(it1 != model_to_planar_model_.end())
        is_planar_model = true;

    if(is_planar_model)
    {
        PCL_WARN("clutter cue not being computed, plane...\n");
        return;
    }*/

    if (detect_clutter_)
    {

        /*float rn_sqr = radius_neighborhood_GO_ * radius_neighborhood_GO_;
        std::vector<int> nn_indices;
        std::vector<float> nn_distances;

        std::vector< std::pair<int, float> > unexplained_points_per_model;
        std::pair<int, float> def_value = std::make_pair(-1, std::numeric_limits<float>::infinity());
        unexplained_points_per_model.resize(scene_cloud_downsampled_->points.size(), def_value);

        for (int i = 0; i < static_cast<int> (recog_model->explained_.size ()); i++)
        {
            if (octree_scene_downsampled_->radiusSearch (scene_cloud_downsampled_->points[recog_model->explained_[i]], radius_neighborhood_GO_, nn_indices,
                                      nn_distances, std::numeric_limits<int>::max ()))
            {
                for (size_t k = 0; k < nn_distances.size (); k++)
                {
                    int sidx = nn_indices[k]; //in the neighborhood of an explained point (idx_to_ep)
                    if(recog_model->scene_point_explained_by_hypothesis_[sidx])
                        continue;

                    assert(recog_model->scene_point_explained_by_hypothesis_[recog_model->explained_[i]]);
                    assert(sidx != recog_model->explained_[i]);

                    float d = (scene_cloud_downsampled_->points[recog_model->explained_[i]].getVector4fMap ()
                               - scene_cloud_downsampled_->points[sidx].getVector4fMap ()).squaredNorm ();

                    if(d < unexplained_points_per_model[sidx].second)
                    {
                        //there is an explained point which is closer to this unexplained point
                        unexplained_points_per_model[sidx].second = d;
                        unexplained_points_per_model[sidx].first = recog_model->explained_[i];
                    }
                }
            }
        }

        recog_model->unexplained_in_neighborhood.resize (scene_cloud_downsampled_->points.size ());
        recog_model->unexplained_in_neighborhood_weights.resize (scene_cloud_downsampled_->points.size ());

        int p=0;
        for(size_t i=0; i < unexplained_points_per_model.size(); i++)
        {
            int sidx = unexplained_points_per_model[i].first;
            if(sidx < 0)
                continue;

            //point i is unexplained and in the neighborhood of sidx
            recog_model->unexplained_in_neighborhood[p] = i;

            float d = unexplained_points_per_model[i].second;
            float d_weight = -(d / rn_sqr) + 1; //points that are close have a strong weight

            //using normals to weight clutter points
            const Eigen::Vector3f & scene_p_normal = scene_normals_->points[sidx].getNormalVector3fMap ();
            const Eigen::Vector3f & model_p_normal = scene_normals_->points[i].getNormalVector3fMap ();
            float dotp = scene_p_normal.dot (model_p_normal); //[-1,1] from antiparallel trough perpendicular to parallel

            if (dotp < 0)
                dotp = 0.f;

            float w = d_weight * dotp;
            if (clusters_cloud_->points[i].label != 0 &&
                    (clusters_cloud_->points[i].label == clusters_cloud_->points[sidx].label))
            {
                recog_model->unexplained_in_neighborhood_weights[p] = clutter_regularizer_ * w; //ATTENTION!
            }
            else
            {
                recog_model->unexplained_in_neighborhood_weights[p] = w;
            }

            p++;
        }

        recog_model->unexplained_in_neighborhood_weights.resize (p);
        recog_model->unexplained_in_neighborhood.resize (p);*/

      std::vector<std::pair<int, int> > neighborhood_indices; //first is indices to scene point and second is indices to explained_ scene points
      neighborhood_indices.reserve(recog_model->explained_.size () * std::min(1000, static_cast<int>(scene_cloud_downsampled_->points.size())));
      float rn_sqr = radius_neighborhood_GO_ * radius_neighborhood_GO_;
      std::vector<int> nn_indices;
      std::vector<float> nn_distances;

      {
        //pcl::ScopeTime t("NN clutter");
        int p=0;
        for (int i = 0; i < static_cast<int> (recog_model->explained_.size ()); i++)
        {
          if (scene_downsampled_tree_->radiusSearch (scene_cloud_downsampled_->points[recog_model->explained_[i]], radius_neighborhood_GO_, nn_indices,
                                                     nn_distances, std::numeric_limits<int>::max ()))
          {
            for (size_t k = 0; k < nn_distances.size (); k++)
            {
              if (nn_indices[k] != i) {
                neighborhood_indices.push_back (std::make_pair<int, int> (nn_indices[k], i));
                p++;
              }
            }
          }
        }

        neighborhood_indices.resize(p);
      }

      std::vector<int> exp_idces (recog_model->explained_);

      {
        //pcl::ScopeTime t("Sort and remove duplicates");
        //sort neighborhood indices by id
        std::sort (neighborhood_indices.begin (), neighborhood_indices.end (),
                   boost::bind (&std::pair<int, int>::first, _1) < boost::bind (&std::pair<int, int>::first, _2));

        //erase duplicated unexplained points
        neighborhood_indices.erase (
                                    std::unique (neighborhood_indices.begin (), neighborhood_indices.end (),
                                                 boost::bind (&std::pair<int, int>::first, _1) == boost::bind (&std::pair<int, int>::first, _2)),
                                    neighborhood_indices.end ());

        //sort explained points
        std::sort (exp_idces.begin (), exp_idces.end ());
      }

      recog_model->unexplained_in_neighborhood.resize (neighborhood_indices.size ());
      recog_model->unexplained_in_neighborhood_weights.resize (neighborhood_indices.size ());

      size_t p = 0;
      size_t j = 0;
      for (size_t i = 0; i < neighborhood_indices.size (); i++)
      {
        if ((j < exp_idces.size ()) && (neighborhood_indices[i].first == exp_idces[j]))
        {
          //this index is explained by the hypothesis so ignore it, advance j
          j++;
        }
        else
        {
          //indices_in_nb[i] < exp_idces[j]
          //recog_model->unexplained_in_neighborhood.push_back(neighborhood_indices[i]);
          recog_model->unexplained_in_neighborhood[p] = neighborhood_indices[i].first;

          if (clusters_cloud_->points[recog_model->explained_[neighborhood_indices[i].second]].label != 0
              && (clusters_cloud_->points[recog_model->explained_[neighborhood_indices[i].second]].label
                  == clusters_cloud_->points[neighborhood_indices[i].first].label))
          {

            recog_model->unexplained_in_neighborhood_weights[p] = clutter_regularizer_;

          }
          else
          {
            //neighborhood_indices[i].first gives the index to the scene point and second to the explained scene point by the model causing this...
            //calculate weight of this clutter point based on the distance of the scene point and the model point causing it
            float d =
                static_cast<float> (pow (
                                         (scene_cloud_downsampled_->points[recog_model->explained_[neighborhood_indices[i].second]].getVector3fMap ()
                                             - scene_cloud_downsampled_->points[neighborhood_indices[i].first].getVector3fMap ()).norm (), 2));
            float d_weight = -(d / rn_sqr) + 1; //points that are close have a strong weight

            //using normals to weight clutter points
            Eigen::Vector3f scene_p_normal = scene_normals_->points[neighborhood_indices[i].first].getNormalVector3fMap ();
            Eigen::Vector3f model_p_normal = scene_normals_->points[recog_model->explained_[neighborhood_indices[i].second]].getNormalVector3fMap ();
            float dotp = scene_p_normal.dot (model_p_normal); //[-1,1] from antiparallel trough perpendicular to parallel

            if (dotp < 0)
              dotp = 0.f;

            recog_model->unexplained_in_neighborhood_weights[p] = d_weight * dotp;
            //recog_model->unexplained_in_neighborhood_weights[p] = 0.f; //ATTENTION!!
          }
          p++;
        }
      }

      recog_model->unexplained_in_neighborhood_weights.resize (p);
      recog_model->unexplained_in_neighborhood.resize (p);
      //std::cout << recog_model->id_ << " " << recog_model->unexplained_in_neighborhood.size() << std::endl;
    }
  }

#ifdef FAAT_PCL_RECOGNITION_USE_GPU

#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/octree/octree.hpp>

template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::computeClutterCueGPU ()
{
  pcl::ScopeTime time_gpu("computeClutterCueGPU");
  if (detect_clutter_) {

    //prepare data (query points)
    typedef pcl::gpu::Octree::PointType PointType;
    std::vector<PointType> queries;
    int queries_size = 0;
    std::vector<bool> is_a_planar_model;
    is_a_planar_model.resize(recognition_models_.size ());

    for (int j = 0; j < static_cast<int> (recognition_models_.size ()); j++)
    {
        int model_id = recognition_models_[j]->id_;
        bool is_planar_model = false;
        std::map<int, int>::iterator it1;
        it1 = model_to_planar_model_.find(model_id);
        if(it1 != model_to_planar_model_.end())
            is_planar_model = true;

        if(!is_planar_model)
            queries_size += static_cast<int> (recognition_models_[j]->explained_.size ());

        is_a_planar_model[j] = is_planar_model;
    }

    queries.resize(queries_size);

    int t=0;
    std::vector<int> query_idx_to_model, query_idx_to_explained_point;
    query_idx_to_model.resize(queries_size);
    query_idx_to_explained_point.resize(queries_size);

    for (int j = 0; j < static_cast<int> (recognition_models_.size ()); j++)
    {
        if(is_a_planar_model[j])
            continue;

        for (int i = 0; i < static_cast<int> (recognition_models_[j]->explained_.size ()); i++)
        {

          assert(recognition_models_[j]->scene_point_explained_by_hypothesis_[recognition_models_[j]->explained_[i]]);
          PointType p;
          p.getVector3fMap() = scene_cloud_downsampled_->points[recognition_models_[j]->explained_[i]].getVector3fMap();
          queries[t] = p;
          query_idx_to_model[t] = j;
          query_idx_to_explained_point[t] = recognition_models_[j]->explained_[i];
          t++;
        }
    }

    //build GPU octree
    typedef pcl::gpu::Octree::PointType PointType;
    std::vector<PointType> points;
    points.resize(scene_cloud_downsampled_->points.size());
    for(size_t i=0; i < scene_cloud_downsampled_->points.size(); i++)
    {
        PointType p;
        p.getVector3fMap() = scene_cloud_downsampled_->points[i].getVector3fMap();
        points[i] = p;
    }

    //std::vector< std::vector<std::pair<int, int> > > neighborhood_indices_all_models;
    //neighborhood_indices_all_models.resize(recognition_models_.size());

    std::vector< std::vector< std::pair<int, float> > > unexplained_points_per_model;
    unexplained_points_per_model.resize(recognition_models_.size());
    std::pair<int, float> def_value = std::make_pair(-1, std::numeric_limits<float>::infinity());
    for(size_t i=0; i < recognition_models_.size(); i++)
        unexplained_points_per_model[i].resize(scene_cloud_downsampled_->points.size(), def_value);

    //for each model, vector of neighborhood_indices
    {
        //pcl::ScopeTime t("building and searching...");
        pcl::gpu::Octree::PointCloud cloud_device;
        cloud_device.upload(points);

        pcl::gpu::Octree octree_device;
        octree_device.setCloud(cloud_device);
        octree_device.build();

        //upload query points
        pcl::gpu::Octree::Queries queries_device;
        queries_device.upload(queries);

        //perform search
        const int max_answers = 1000;
        std::vector<int> sizes;
        std::vector<int> rs_indices;
        {
            //pcl::ScopeTime act_search("actual search and download");
            pcl::gpu::NeighborIndices result_device1(queries_device.size(), max_answers);
            octree_device.radiusSearch(queries_device, radius_neighborhood_GO_, max_answers, result_device1);

            //download
            result_device1.sizes.download(sizes);
            result_device1.data.download(rs_indices);
        }

        {
            //pcl::ScopeTime kaka("time spend processing indices");
            for(size_t i = 0; i < queries.size(); ++i)
            {
                int beg = i * max_answers;
                int midx = query_idx_to_model[i];
                int idx_to_ep = query_idx_to_explained_point[i];

                assert(sizes[i] < max_answers);
                assert(recognition_models_[midx]->scene_point_explained_by_hypothesis_[idx_to_ep]);

                for(size_t k=0; k < sizes[i]; k++)
                {
                    int sidx = rs_indices[beg+k]; //in the neighborhood of an explained point (idx_to_ep)
                    if(recognition_models_[midx]->scene_point_explained_by_hypothesis_[sidx])
                        continue;

                    //if the points are equal, then sidx should be explained!
                    assert(idx_to_ep != sidx);
                    //point sidx is not explained

                    float d = (scene_cloud_downsampled_->points[idx_to_ep].getVector4fMap ()
                                  - scene_cloud_downsampled_->points[sidx].getVector4fMap ()).squaredNorm ();

                    if(d < unexplained_points_per_model[midx][sidx].second)
                    {
                        //there is an explained point which is closer to this unexplained point
                        unexplained_points_per_model[midx][sidx].second = d;
                        unexplained_points_per_model[midx][sidx].first = idx_to_ep;
                    }

                    //neighborhood_indices_all_models[midx].push_back (std::make_pair<int, int> (sidx, idx_to_ep));
                    //neighborhood_indices_all_models_set[midx].insert(rs_indices[beg+k]);
                }
            }
        }
    }

    float rn_sqr = radius_neighborhood_GO_ * radius_neighborhood_GO_;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_num_procs())
    for(size_t kk=0; kk < recognition_models_.size(); kk++)
    {
        if(is_a_planar_model[kk])
            continue;

        boost::shared_ptr<RecognitionModel<ModelT> > recog_model = recognition_models_[kk];
        recog_model->unexplained_in_neighborhood.resize (scene_cloud_downsampled_->points.size ());
        recog_model->unexplained_in_neighborhood_weights.resize (scene_cloud_downsampled_->points.size ());

        int p=0;
        for(size_t i=0; i < unexplained_points_per_model[kk].size(); i++)
        {
            int sidx = unexplained_points_per_model[kk][i].first;
            if(sidx < 0)
                continue;

            //point i is unexplained and in the neighborhood of sidx

            recog_model->unexplained_in_neighborhood[p] = i;

            if (clusters_cloud_->points[i].label != 0 &&
                (clusters_cloud_->points[i].label == clusters_cloud_->points[sidx].label))
            {
              recog_model->unexplained_in_neighborhood_weights[p] = clutter_regularizer_;
            }
            else
            {
                float d = unexplained_points_per_model[kk][i].second;
                float d_weight = -(d / rn_sqr) + 1; //points that are close have a strong weight

                //using normals to weight clutter points
                const Eigen::Vector3f & scene_p_normal = scene_normals_->points[sidx].getNormalVector3fMap ();
                const Eigen::Vector3f & model_p_normal = scene_normals_->points[i].getNormalVector3fMap ();
                float dotp = scene_p_normal.dot (model_p_normal); //[-1,1] from antiparallel trough perpendicular to parallel

                if (dotp < 0)
                  dotp = 0.f;

                recog_model->unexplained_in_neighborhood_weights[p] = d_weight * dotp;
            }

            p++;
        }

        recog_model->unexplained_in_neighborhood_weights.resize (p);
        recog_model->unexplained_in_neighborhood.resize (p);
    }
  }
}

#else
  template<typename ModelT, typename SceneT>
  void
  faat_pcl::GlobalHypothesesVerification_1<ModelT, SceneT>::computeClutterCueGPU ()
  {

  }

#endif

#define PCL_INSTANTIATE_faatGoHV_1(T1,T2) template class FAAT_REC_API faat_pcl::GlobalHypothesesVerification_1<T1,T2>;
