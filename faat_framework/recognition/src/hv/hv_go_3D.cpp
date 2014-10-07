/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
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

#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/impl/instantiate.hpp>
#include <faat_pcl/recognition/hv/hv_go_3D.h>
#include "faat_pcl/recognition/impl/hv/occlusion_reasoning.hpp"
#include <pcl/features/normal_3d_omp.h>
#include <functional>
#include <numeric>

template<typename ModelT, typename SceneT>
bool
faat_pcl::GO3D<ModelT, SceneT>::getInlierOutliersCloud(int hyp_idx, typename pcl::PointCloud<ModelT>::Ptr & cloud)
{
  if(hyp_idx < 0 || hyp_idx > (recognition_models_.size() - 1))
    return false;

  boost::shared_ptr<GHVRecognitionModel<ModelT> > recog_model = recognition_models_[hyp_idx];
  cloud.reset(new pcl::PointCloud<ModelT>(*visible_models_[hyp_idx]));


  for(size_t i=0; i < cloud->points.size(); i++)
  {
    cloud->points[i].r = 0;
    cloud->points[i].g = 255;
    cloud->points[i].b = 0;
  }

  for(size_t i=0; i < recog_model->outlier_indices_.size(); i++)
  {
    cloud->points[recog_model->outlier_indices_[i]].r = 255;
    cloud->points[recog_model->outlier_indices_[i]].g = 0;
    cloud->points[recog_model->outlier_indices_[i]].b = 0;
  }
}

/*template<typename ModelT, typename SceneT>
void
faat_pcl::GO3D<ModelT, SceneT>::initialize ()
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

    scene_normals_ = scene_normals_go3D_;
    scene_cloud_downsampled_ = scene_cloud_downsampled_GO3D_;

    explained_by_RM_.resize (scene_cloud_downsampled_->points.size (), 0);
    explained_by_RM_distance_weighted.resize (scene_cloud_downsampled_->points.size (), 0.f);
    unexplained_by_RM_neighboorhods.resize (scene_cloud_downsampled_->points.size (), 0.f);

    octree_scene_downsampled_.reset(new pcl::octree::OctreePointCloudSearch<SceneT>(0.01f));
    octree_scene_downsampled_->setInputCloud(scene_cloud_downsampled_);
    octree_scene_downsampled_->addPointsFromInputCloud();

    //compute segmentation of the scene if detect_clutter_
    if (detect_clutter_)
    {
      pcl::ScopeTime t("Smooth segmentation of the scene");
      //initialize kdtree for search

      scene_downsampled_tree_.reset (new pcl::search::KdTree<SceneT>);
      scene_downsampled_tree_->setInputCloud (scene_cloud_downsampled_);

      if(use_super_voxels_)
      {
        float color_importance_ = 0;
        if(!ignore_color_even_if_exists_)
        {
            color_importance_ = 0.5f;
        }
        float voxel_resolution = 0.005f;
        float seed_resolution = radius_neighborhood_GO_;
        typename pcl::SupervoxelClustering<SceneT> super (voxel_resolution, seed_resolution, false);
        super.setInputCloud (scene_cloud_downsampled_);
        super.setColorImportance (color_importance_);
        super.setSpatialImportance (1.f);
        super.setNormalImportance (1.f);
        std::map <uint32_t, typename pcl::Supervoxel<SceneT>::Ptr > supervoxel_clusters;
        pcl::console::print_highlight ("Extracting supervoxels!\n");
        super.extract (supervoxel_clusters);
        pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());

        pcl::PointCloud<pcl::PointXYZL>::Ptr supervoxels_labels_cloud = super.getLabeledCloud();
        std::cout << scene_cloud_downsampled_->points.size () << " " << supervoxels_labels_cloud->points.size () << std::endl;

        clusters_cloud_rgb_= super.getColoredCloud();
        clusters_cloud_.reset (new pcl::PointCloud<pcl::PointXYZL>(*supervoxels_labels_cloud));
      }
      else
      {

        clusters_cloud_.reset (new pcl::PointCloud<pcl::PointXYZL>);
        clusters_cloud_rgb_.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);

        std::vector<pcl::PointIndices> clusters;
        this->template extractEuclideanClustersSmooth<SceneT, pcl::Normal>
                    (*scene_cloud_downsampled_, *scene_normals_, cluster_tolerance_,
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

    if(!ignore_color_even_if_exists_)
    {
        bool exists_s;
        float rgb_s;
        scene_LAB_values_.resize(scene_cloud_downsampled_->points.size());
        scene_RGB_values_.resize(scene_cloud_downsampled_->points.size());
        scene_GS_values_.resize(scene_cloud_downsampled_->points.size());
        for(size_t i=0; i < scene_cloud_downsampled_->points.size(); i++)
        {
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

                RGB2CIELAB (rs, gs, bs, LRefs, aRefs, bRefs);
                LRefs /= 100.0f; aRefs /= 120.0f; bRefs /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

                scene_LAB_values_[i] = (Eigen::Vector3f(LRefs, aRefs, bRefs));

                float rsf,gsf,bsf;
                rsf = static_cast<float>(rs) / 255.f;
                gsf = static_cast<float>(gs) / 255.f;
                bsf = static_cast<float>(bs) / 255.f;
                scene_RGB_values_[i] = (Eigen::Vector3f(rsf,gsf,bsf));

                scene_GS_values_[i] = (rsf + gsf + bsf) / 3.f;
            }
        }
    }

    //compute cues
    {
      valid_model_.resize(complete_models_.size (), true);
      {
        pcl::ScopeTime tcues ("Computing cues");
        recognition_models_.resize (complete_models_.size ());
  #pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_num_procs())
        for (int i = 0; i < static_cast<int> (complete_models_.size ()); i++)
        {
          //create recognition model
          recognition_models_[i].reset (new GHVRecognitionModel<ModelT> ());
          if(!addModel(i, recognition_models_[i])) {
            valid_model_[i] = false;
            PCL_WARN("Model is not valid\n");
          }
        }
      }

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

          std::vector<bool> banned_vector(size_x * size_y * size_z, false);
          recognition_models_[i]->complete_cloud_occupancy_indices_.resize(complete_models_[i]->points.size ());
          int used = 0;

          for (size_t j = 0; j < complete_models_[i]->points.size (); j++)
          {
            int pos_x, pos_y, pos_z;
            pos_x = static_cast<int> (std::floor ((complete_models_[i]->points[j].x - min_pt_all.x) / res_occupancy_grid_));
            pos_y = static_cast<int> (std::floor ((complete_models_[i]->points[j].y - min_pt_all.y) / res_occupancy_grid_));
            pos_z = static_cast<int> (std::floor ((complete_models_[i]->points[j].z - min_pt_all.z) / res_occupancy_grid_));

            int idx = pos_z * size_x * size_y + pos_y * size_x + pos_x;
            assert(banned_vector.size() > idx);
            if (!banned_vector[idx])
            {
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
      pcl::ScopeTime tcues ("Computing clutter cues");
      computeClutterCueAtOnce ();
    }

    points_explained_by_rm_.clear ();
    points_explained_by_rm_.resize (scene_cloud_downsampled_->points.size ());
    for (size_t j = 0; j < recognition_models_.size (); j++)
    {
      boost::shared_ptr<GHVRecognitionModel<ModelT> > recog_model = recognition_models_[j];
      for (size_t i = 0; i < recog_model->explained_.size (); i++)
      {
        points_explained_by_rm_[recog_model->explained_[i]].push_back (recog_model);
      }
    }

    cc_.clear ();
    n_cc_ = 1;
    cc_.resize (n_cc_);
    for (size_t i = 0; i < recognition_models_.size (); i++)
    {
      if(!valid_model_[i])
        continue;

      cc_[0].push_back (static_cast<int> (i));
    }
  }*/

/*template<typename ModelT, typename SceneT>
bool
faat_pcl::GO3D<ModelT, SceneT>::handlingNormals (boost::shared_ptr<GHVRecognitionModel<ModelT> > & recog_model, int i, bool is_planar_model, int object_models_size)
{
    //std::cout << visible_normal_models_.size() << " " << object_models_size << " " << complete_models_.size() << std::endl;
    if(visible_normal_models_.size() != object_models_size)
    {
        PCL_ERROR("Number of models does not match number of visible normals\n");
        return false;
    }

    if(!is_planar_model)
    {

      //recompute normals and orient them properly based on visible_normal_models_[i]
      pcl::PointCloud<pcl::Normal>::ConstPtr model_normals = visible_normal_models_[i];
      recog_model->normals_.reset (new pcl::PointCloud<pcl::Normal> (*model_normals));
      //ATTENTION: tHIS was different...
      //pcl::ScopeTime t("Using model normals and checking nans");

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
        int flips = 0;
        for (size_t i = 0; i < recog_model->normals_->points.size (); ++i)
        {
          if (!pcl_isfinite (recog_model->normals_->points[i].normal_x) || !pcl_isfinite (recog_model->normals_->points[i].normal_y)
              || !pcl_isfinite (recog_model->normals_->points[i].normal_z))
            continue;

          recog_model->normals_->points[j] = recog_model->normals_->points[i];
          recog_model->cloud_->points[j] = recog_model->cloud_->points[i];

          const Eigen::Vector3f & computed_normal = recog_model->normals_->points[j].getNormalVector3fMap();
          const Eigen::Vector3f & model_normal = model_normals->points[i].getNormalVector3fMap();
          if(model_normal.dot(computed_normal) < 0)
          {
              recog_model->normals_->points[j].getNormalVector3fMap() = computed_normal * -1.f;
              flips++;
          }
          j++;
        }

        std::cout << "flipped vs total:" << flips << " " << recog_model->cloud_->points.size() << std::endl;

        recog_model->normals_->points.resize (j);
        recog_model->normals_->width = j;
        recog_model->normals_->height = 1;

        recog_model->cloud_->points.resize (j);
        recog_model->cloud_->width = j;
        recog_model->cloud_->height = 1;

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
}*/

template<typename ModelT, typename SceneT>
void
faat_pcl::GO3D<ModelT, SceneT>::addModels (std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models, bool occlusion_reasoning)
{
  std::cout << "Called GO3D addModels" << std::endl;
  mask_.clear();

  if (!occlusion_reasoning)
    visible_models_ = models;
  else
  {
    visible_indices_.resize(models.size());

    //pcl::visualization::PCLVisualizer vis("visible model");

    for (size_t i = 0; i < models.size (); i++)
    {

      //a point is occluded if it is occluded in all views
      typename pcl::PointCloud<ModelT>::Ptr filtered (new pcl::PointCloud<ModelT> ());

      //scene-occlusions
      for(size_t k=0; k < occ_clouds_.size(); k++)
      {
        //transform model to camera coordinate
        typename pcl::PointCloud<ModelT>::Ptr model_in_view_coordinates(new pcl::PointCloud<ModelT> ());
        Eigen::Matrix4f trans =  absolute_poses_camera_to_global_[k].inverse();
        pcl::transformPointCloud(*models[i], *model_in_view_coordinates, trans);
        typename pcl::PointCloud<ModelT>::ConstPtr const_filtered(new pcl::PointCloud<ModelT> (*model_in_view_coordinates));

        std::vector<int> indices_cloud_occlusion;
        filtered = faat_pcl::occlusion_reasoning::filter<ModelT,SceneT> (occ_clouds_[k], const_filtered, 525.f, occlusion_thres_, indices_cloud_occlusion);

        std::vector<int> final_indices = indices_cloud_occlusion;
        final_indices.resize(indices_cloud_occlusion.size());

        visible_indices_[i].insert(visible_indices_[i].end(), final_indices.begin(), final_indices.end());
      }

      std::set<int> s( visible_indices_[i].begin(), visible_indices_[i].end() );
      visible_indices_[i].assign( s.begin(), s.end() );

      pcl::copyPointCloud(*models[i], visible_indices_[i], *filtered);

      if(normals_set_ && requires_normals_) {
        pcl::PointCloud<pcl::Normal>::Ptr filtered_normals (new pcl::PointCloud<pcl::Normal> ());
        pcl::copyPointCloud(*complete_normal_models_[i], visible_indices_[i], *filtered_normals);
        visible_normal_models_.push_back(filtered_normals);
      }

      /*pcl::visualization::PointCloudColorHandlerRGBField<ModelT> handler (filtered);
      vis.addPointCloud(filtered, handler, "model");
      vis.spin();
      vis.removeAllPointClouds();*/

      visible_models_.push_back (filtered);
    }

    complete_models_ = models;
  }

  normals_set_ = false;
}

template<typename ModelT, typename SceneT> float faat_pcl::GO3D<ModelT, SceneT>::sRGB_LUT[256] = {- 1};
template<typename ModelT, typename SceneT> float faat_pcl::GO3D<ModelT, SceneT>::sXYZ_LUT[4000] = {- 1};

//template class FAAT_REC_API faat_pcl::GO3D<pcl::PointXYZ,pcl::PointXYZ>;
template class FAAT_REC_API faat_pcl::GO3D<pcl::PointXYZRGB,pcl::PointXYZRGB>;
