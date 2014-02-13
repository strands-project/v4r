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
#include <faat_pcl/recognition/hv/hv_go.h>
//#include <faat_pcl/recognition/hv/hv_go_binary_optimizer.h>
#include <functional>
#include <numeric>
#include <pcl/common/time.h>
#include <boost/graph/connected_components.hpp>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>

template<typename PointT, typename NormalT>
  inline void
  extractEuclideanClustersSmooth (const typename pcl::PointCloud<PointT> &cloud, const typename pcl::PointCloud<NormalT> &normals, float tolerance,
                                  const typename pcl::search::Search<PointT>::Ptr &tree, std::vector<pcl::PointIndices> &clusters, double eps_angle,
                                  float curvature_threshold, unsigned int min_pts_per_cluster,
                                  unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ())
  {

    if (tree->getInputCloud ()->points.size () != cloud.points.size ())
    {
      PCL_ERROR("[pcl::extractEuclideanClusters] Tree built for a different point cloud dataset\n");
      return;
    }
    if (cloud.points.size () != normals.points.size ())
    {
      PCL_ERROR("[pcl::extractEuclideanClusters] Number of points in the input point cloud different than normals!\n");
      return;
    }

    // Create a bool vector of processed point indices, and initialize it to false
    std::vector<bool> processed (cloud.points.size (), false);

    std::vector<int> nn_indices;
    std::vector<float> nn_distances;
    // Process all points in the indices vector
    int size = static_cast<int> (cloud.points.size ());
    for (int i = 0; i < size; ++i)
    {
      if (processed[i])
        continue;

      std::vector<unsigned int> seed_queue;
      int sq_idx = 0;
      seed_queue.push_back (i);

      processed[i] = true;

      while (sq_idx < static_cast<int> (seed_queue.size ()))
      {

        if (normals.points[seed_queue[sq_idx]].curvature > curvature_threshold)
        {
          sq_idx++;
          continue;
        }

        // Search for sq_idx
        if (!tree->radiusSearch (seed_queue[sq_idx], tolerance, nn_indices, nn_distances))
        {
          sq_idx++;
          continue;
        }

        for (size_t j = 1; j < nn_indices.size (); ++j) // nn_indices[0] should be sq_idx
        {
          if (processed[nn_indices[j]]) // Has this point been processed before ?
            continue;

          if (normals.points[nn_indices[j]].curvature > curvature_threshold)
          {
            continue;
          }

          //processed[nn_indices[j]] = true;
          // [-1;1]

          double dot_p = normals.points[seed_queue[sq_idx]].normal[0] * normals.points[nn_indices[j]].normal[0]
              + normals.points[seed_queue[sq_idx]].normal[1] * normals.points[nn_indices[j]].normal[1] + normals.points[seed_queue[sq_idx]].normal[2]
              * normals.points[nn_indices[j]].normal[2];

          if (fabs (acos (dot_p)) < eps_angle)
          {
            processed[nn_indices[j]] = true;
            seed_queue.push_back (nn_indices[j]);
          }
        }

        sq_idx++;
      }

      // If this queue is satisfactory, add to the clusters
      if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)
      {
        pcl::PointIndices r;
        r.indices.resize (seed_queue.size ());
        for (size_t j = 0; j < seed_queue.size (); ++j)
          r.indices[j] = seed_queue[j];

        std::sort (r.indices.begin (), r.indices.end ());
        r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());
        clusters.push_back (r); // We could avoid a copy by working directly in the vector
      }
    }
  }

template<typename ModelT, typename SceneT>
  mets::gol_type
  faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::evaluateSolution (const std::vector<bool> & active, int changed)
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
    float good_info = getExplainedValue ();

    float unexplained_info = getPreviousUnexplainedValue ();
    if(!detect_clutter_) {
      unexplained_info = 0;
    }

    float bad_info = static_cast<float> (getPreviousBadInfo ()) + (recognition_models_[changed]->outliers_weight_
        * static_cast<float> (recognition_models_[changed]->bad_information_)) * sign;

    setPreviousBadInfo (bad_info);

    //float duplicity_cm = static_cast<float> (getDuplicityCM ()) * w_occupied_multiple_cm_;
    float duplicity_cm = 0;

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
  faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::countActiveHypotheses (const std::vector<bool> & sol)
{
  float c = 0;
  for (size_t i = 0; i < sol.size (); i++)
  {
    if (sol[i]) {
      //c++;
      c += static_cast<float>(recognition_models_[i]->explained_.size()) * active_hyp_penalty_;
    }
  }

  return c;
  //return static_cast<float> (c) * active_hyp_penalty_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename ModelT, typename SceneT>
  void
  faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::initialize ()
  {
    //clear stuff
    recognition_models_.clear ();
    unexplained_by_RM_neighboorhods.clear ();
    explained_by_RM_distance_weighted.clear ();
    explained_by_RM_.clear ();
    mask_.clear ();
    indices_.clear ();
    complete_cloud_occupancy_by_RM_.clear ();

    // initialize mask to false
    mask_.resize (complete_models_.size ());
    for (size_t i = 0; i < complete_models_.size (); i++)
      mask_[i] = false;

    indices_.resize (complete_models_.size ());

    NormalEstimator_ n3d;
    scene_normals_.reset (new pcl::PointCloud<pcl::Normal> ());

    int j = 0;
    for (size_t i = 0; i < scene_cloud_downsampled_->points.size (); ++i) {
      if (!pcl_isfinite (scene_cloud_downsampled_->points[i].x) || !pcl_isfinite (scene_cloud_downsampled_->points[i].y)
          || !pcl_isfinite (scene_cloud_downsampled_->points[i].z)) {
        std::cout << "Not finite..." << std::endl;
        continue;
      }

      scene_cloud_downsampled_->points[j] = scene_cloud_downsampled_->points[i];

      j++;
    }

    scene_cloud_downsampled_->points.resize(j);
    scene_cloud_downsampled_->width = j;
    scene_cloud_downsampled_->height = 1;

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

    explained_by_RM_.resize (scene_cloud_downsampled_->points.size (), 0);
    explained_by_RM_distance_weighted.resize (scene_cloud_downsampled_->points.size (), 0.f);
    unexplained_by_RM_neighboorhods.resize (scene_cloud_downsampled_->points.size (), 0.f);

    //compute segmentation of the scene if detect_clutter_
    if (detect_clutter_)
    {
      pcl::ScopeTime t("Smooth segmentation of the scene");
      //initialize kdtree for search
      scene_downsampled_tree_.reset (new pcl::search::KdTree<SceneT>);
      scene_downsampled_tree_->setInputCloud (scene_cloud_downsampled_);

      std::vector<pcl::PointIndices> clusters;

      extractEuclideanClustersSmooth<SceneT, pcl::Normal> (*scene_cloud_downsampled_, *scene_normals_, cluster_tolerance_,
                                                           scene_downsampled_tree_, clusters, eps_angle_threshold_, curvature_threshold_, min_points_);

      clusters_cloud_.reset (new pcl::PointCloud<pcl::PointXYZI>);
      clusters_cloud_->points.resize (scene_cloud_downsampled_->points.size ());
      clusters_cloud_->width = scene_cloud_downsampled_->width;
      clusters_cloud_->height = 1;

      for (size_t i = 0; i < scene_cloud_downsampled_->points.size (); i++)
      {
        pcl::PointXYZI p;
        p.getVector3fMap () = scene_cloud_downsampled_->points[i].getVector3fMap ();
        p.intensity = 0.f;
        clusters_cloud_->points[i] = p;
      }

      float intens_incr = 100.f / static_cast<float> (clusters.size ());
      float intens = intens_incr;
      for (size_t i = 0; i < clusters.size (); i++)
      {
        for (size_t j = 0; j < clusters[i].indices.size (); j++)
        {
          clusters_cloud_->points[clusters[i].indices[j]].intensity = intens;
        }

        intens += intens_incr;
      }
    }

    //compute cues
    {
      pcl::ScopeTime tcues ("Computing cues");
      recognition_models_.resize (complete_models_.size ());
      std::vector<bool> valid_model(complete_models_.size (), true);
#pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_num_procs())
      for (int i = 0; i < static_cast<int> (complete_models_.size ()); i++)
      {
        //create recognition model
        recognition_models_[i].reset (new RecognitionModel ());
        if(!addModel(i, recognition_models_[i])) {
          valid_model[i] = false;
        }

        /*pcl::PointCloud<pcl::Normal>::ConstPtr null_ptr;
        if (extra_weights_.size () == recognition_models_.size ())
        {
          if(visible_normal_models_.size() == recognition_models_.size()) {
            if (!addModel (visible_models_[i], complete_models_[i], visible_normal_models_[i], recognition_models_[i], visible_indices_[i], extra_weights_[i]))
              valid_model[i] = false;
          } else {
            if (!addModel (visible_models_[i], complete_models_[i], null_ptr, recognition_models_[i],visible_indices_[i],  extra_weights_[i]))
              valid_model[i] = false;
          }
        }
        else
        {
          if(visible_normal_models_.size() == recognition_models_.size()) {
            if (!addModel (visible_models_[i], complete_models_[i], visible_normal_models_[i], recognition_models_[i], visible_indices_[i]))
              valid_model[i] = false;
          } else {
            if (!addModel (visible_models_[i], complete_models_[i], null_ptr, recognition_models_[i], visible_indices_[i]))
              valid_model[i] = false;
          }
        }*/
      }

      //go through valid model vector
      int valid = 0;
      for (int i = 0; i < static_cast<int> (valid_model.size ()); i++) {
        if(valid_model[i]) {
          recognition_models_[valid] = recognition_models_[i];
          indices_[valid] = i;
          valid++;
        }
      }

      recognition_models_.resize (valid);
      indices_.resize (valid);

      //compute the bounding boxes for the models
      ModelT min_pt_all, max_pt_all;
      min_pt_all.x = min_pt_all.y = min_pt_all.z = std::numeric_limits<float>::max ();
      max_pt_all.x = max_pt_all.y = max_pt_all.z = (std::numeric_limits<float>::max () - 0.001f) * -1;

      for (size_t i = 0; i < recognition_models_.size (); i++)
      {
        ModelT min_pt, max_pt;
        pcl::getMinMax3D (*complete_models_[indices_[i]], min_pt, max_pt);
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

        std::map<int, bool> banned;
        std::map<int, bool>::iterator banned_it;

        for (size_t j = 0; j < complete_models_[indices_[i]]->points.size (); j++)
        {
          int pos_x, pos_y, pos_z;
          pos_x = static_cast<int> (std::floor ((complete_models_[indices_[i]]->points[j].x - min_pt_all.x) / res_occupancy_grid_));
          pos_y = static_cast<int> (std::floor ((complete_models_[indices_[i]]->points[j].y - min_pt_all.y) / res_occupancy_grid_));
          pos_z = static_cast<int> (std::floor ((complete_models_[indices_[i]]->points[j].z - min_pt_all.z) / res_occupancy_grid_));

          int idx = pos_z * size_x * size_y + pos_y * size_x + pos_x;
          banned_it = banned.find (idx);
          if (banned_it == banned.end ())
          {
            //complete_cloud_occupancy_by_RM_[idx]++;
            recognition_models_[i]->complete_cloud_occupancy_indices_.push_back (idx);
            banned[idx] = true;
          }
        }
      }
    }

    {
      pcl::ScopeTime tcues ("Computing clutter cues");
#pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_num_procs())
      for (int j = 0; j < static_cast<int> (recognition_models_.size ()); j++)
        computeClutterCue (recognition_models_[j]);
    }

    points_explained_by_rm_.clear ();
    points_explained_by_rm_.resize (scene_cloud_downsampled_->points.size ());
    for (size_t j = 0; j < recognition_models_.size (); j++)
    {
      boost::shared_ptr<RecognitionModel> recog_model = recognition_models_[j];
      for (size_t i = 0; i < recog_model->explained_.size (); i++)
      {
        points_explained_by_rm_[recog_model->explained_[i]].push_back (recog_model);
      }
    }

    if (use_conflict_graph_)
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
    }
  }

template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::fill_structures(std::vector<int> & cc_indices, std::vector<bool> & initial_solution, SAModel & model)
{
  for (size_t j = 0; j < recognition_models_.size (); j++)
  {
    if(!initial_solution[j])
      continue;

    boost::shared_ptr<RecognitionModel> recog_model = recognition_models_[j];
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

  int duplicity;
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
faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::clear_structures()
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
  faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::SAOptimize (std::vector<int> & cc_indices, std::vector<bool> & initial_solution)
  {

    //temporal copy of recogniton_models_
    std::vector<boost::shared_ptr<RecognitionModel> > recognition_models_copy;
    recognition_models_copy = recognition_models_;

    recognition_models_.clear ();

    for (size_t j = 0; j < cc_indices.size (); j++)
      recognition_models_.push_back (recognition_models_copy[cc_indices[j]]);

    SAModel model;
    fill_structures(cc_indices, initial_solution, model);

    SAModel * best = new SAModel (model);

    move_manager neigh (static_cast<int> (cc_indices.size ()), use_replace_moves_);
    boost::shared_ptr<std::map< std::pair<int, int>, bool > > intersect_map;
    intersect_map.reset(new std::map< std::pair<int, int>, bool >);
    for (size_t i = 0; i < recognition_models_.size (); i++)
    {
      for (size_t j = i; j < recognition_models_.size (); j++)
      {
        if (i != j)
        {
          float n_conflicts = 0.f;
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
            }
          }

          bool conflict = false;
          conflict = (n_conflicts > 100);
          std::pair<int, int> p = std::make_pair<int, int> (static_cast<int> (i), static_cast<int> (j));
          (*intersect_map)[p] = conflict;
        }
      }
    }

    neigh.setExplainedPointIntersections(intersect_map);

    //mets::best_ever_solution best_recorder (best);
    cost_logger_.reset(new CostFunctionLogger(*best));
    mets::noimprove_termination_criteria noimprove (max_iterations_);

    switch(opt_type_)
    {
      case 0:
      {
        mets::local_search<move_manager> local ( model, *cost_logger_, neigh, 0, false);
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
        mets::tabu_search<move_manager> tabu_search(model, *cost_logger_, neigh, tabu_list, aspiration_criteria, noimprove);
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

        mets::local_search<move_manager> local ( model, *cost_logger_, neigh, 0, false);

        {
          pcl::ScopeTime t ("local search WITHIN B&B...");
          local.search ();
        }

        best_seen_ = static_cast<const SAModel&> (cost_logger_->best_seen ());
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
          std::vector<boost::shared_ptr<RecognitionModel> > recognition_models_copy2;

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
        best = new SAModel(model);
        cost_logger_.reset(new CostFunctionLogger(*best));

        //brute force with branch and bound
        HVGOBinaryOptimizer bin_opt(model, *cost_logger_, neigh, initial_solution.size());
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

        best_seen_ = static_cast<const SAModel&> (cost_logger_->best_seen ());
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
        mets::simulated_annealing<move_manager> sa (model, *cost_logger_, neigh, noimprove, linear_cooling, initial_temp_, 1e-7, 2);
        sa.setApplyAndEvaluate (true);

        {
          pcl::ScopeTime t ("SA search...");
          sa.search ();
        }
        break;
      }
    }

    best_seen_ = static_cast<const SAModel&> (cost_logger_->best_seen ());
    std::cout << "*****************************" << std::endl;
    std::cout << "Final cost:" << best_seen_.cost_;
    std::cout << " Number of ef evaluations:" << cost_logger_->getTimesEvaluated();
    std::cout << std::endl;
    std::cout << "Number of accepted moves:" << cost_logger_->getAcceptedMovesSize() << std::endl;
    std::cout << "*****************************" << std::endl;

    for (size_t i = 0; i < best_seen_.solution_.size (); i++) {
      initial_solution[i] = best_seen_.solution_[i];
    }

    for(size_t i = 0; i < initial_solution.size(); i++) {
      if(initial_solution[i]) {
        std::cout << "id: " << recognition_models_[i]->id_s_ << std::endl;
        std::cout << "Median:" << recognition_models_[i]->median_ << std::endl;
        std::cout << "Mean:" << recognition_models_[i]->mean_ << std::endl;
        std::cout << "Color similarity:" << recognition_models_[i]->color_similarity_ << std::endl;
        std::cout << "#outliers:" << recognition_models_[i]->bad_information_ << " " << recognition_models_[i]->outliers_weight_ << std::endl;
        std::cout << "#under table:" << recognition_models_[i]->model_constraints_value_ << std::endl;

        /*pcl::visualization::PCLVisualizer vis_ ("test histograms");
        int v1, v2, v3;
        vis_.createViewPort (0.0, 0.0, 0.33, 1.0, v1);
        vis_.createViewPort (0.33, 0.0, 0.66, 1.0, v2);
        vis_.createViewPort (0.66, 0.0, 1, 1.0, v3);

        //typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_ps(new pcl::PointCloud<pcl::PointXYZRGB>());
        //pcl::copyPointCloud(*recog_model->cloud_, model_indices_explained, *model_ps);
        typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_ps(new pcl::PointCloud<pcl::PointXYZRGB>(*(recognition_models_[i]->cloud_)));

        {
          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_model (recognition_models_[i]->cloud_);
          vis_.addPointCloud<pcl::PointXYZRGB> (recognition_models_[i]->cloud_, handler_rgb_model, "nmodel_orig", v3);

          typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr outliers(new pcl::PointCloud<pcl::PointXYZRGB>());
          pcl::copyPointCloud(*recognition_models_[i]->cloud_, recognition_models_[i]->outlier_indices_, *outliers);
          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> handler_outliers (outliers, 255, 255, 0);
          vis_.addPointCloud<pcl::PointXYZRGB> (outliers, handler_outliers, "outliers", v3);
        }

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_model (model_ps);
        vis_.addPointCloud<pcl::PointXYZRGB> (model_ps, handler_rgb_model, "nmodel", v2);

        typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr explained_ps(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*scene_cloud_downsampled_, recognition_models_[i]->explained_, *explained_ps);

        std::cout << recognition_models_[i]->cloud_->points.size() << " " << explained_ps->points.size() << std::endl;

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_scene (explained_ps);
        vis_.addPointCloud<pcl::PointXYZRGB> (explained_ps, handler_rgb_scene, "scene_cloud", v1);

        {
          typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr unexplained(new pcl::PointCloud<pcl::PointXYZRGB>());
          pcl::copyPointCloud(*scene_cloud_downsampled_, recognition_models_[i]->unexplained_in_neighborhood, *unexplained);
          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> handler_outliers (unexplained, 0, 255, 255);
          vis_.addPointCloud<pcl::PointXYZRGB> (unexplained, handler_outliers, "unexplained", v1);
        }
        vis_.spin();*/
      }
    }

    delete best;

    recognition_models_ = recognition_models_copy;

  }

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename ModelT, typename SceneT>
  void
  faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::verify ()
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
        mask_[indices_[cc_[c][i]]] = (subsolution[i]);
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
faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::computeYUVHistogram(std::vector<Eigen::Vector3f> & yuv_values,
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
faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::computeHueHistogram(std::vector<Eigen::Vector3f> & hsv_values,
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

void inline
computeRGBHistograms (std::vector<Eigen::Vector3f> & rgb_values, Eigen::MatrixXf & rgb, int dim = 3, float min = 0.f, float max = 255.f, bool soft = false)
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

void inline
specifyRGBHistograms (Eigen::MatrixXf & src, Eigen::MatrixXf & dst, Eigen::MatrixXf & lookup, int dim = 3)
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

//if (!addModel (visible_models_[i], complete_models_[i], visible_normal_models_[i], recognition_models_[i], visible_indices_[i]))
/*template<typename ModelT, typename SceneT>
  bool
  faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::addModel (typename pcl::PointCloud<ModelT>::ConstPtr & model,
                                                                    typename pcl::PointCloud<ModelT>::ConstPtr & complete_model,
                                                                    pcl::PointCloud<pcl::Normal>::ConstPtr & model_normals,
                                                                    boost::shared_ptr<RecognitionModel> & recog_model,
                                                                    std::vector<int> & visible_indices,
                                                                    float extra_weight)*/

  template<typename ModelT, typename SceneT>
  bool
  faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::addModel (int i, boost::shared_ptr<RecognitionModel> & recog_model)
  {

    if (normals_for_visibility_.size () == complete_models_.size ())
    {
      pcl::PointCloud<pcl::Normal>::Ptr filtered_normals (new pcl::PointCloud<pcl::Normal> ());
      pcl::copyPointCloud (*normals_for_visibility_[i], visible_indices_[i], *filtered_normals);

      assert(filtered_normals->points.size() == visible_models_[i]->points.size());

      std::vector<int> keep;
      for (size_t k = 0; k < visible_models_[i]->points.size (); k++)
      {
        Eigen::Vector3f normal_p = filtered_normals->points[k].getNormalVector3fMap ();
        Eigen::Vector3f normal_vp = visible_models_[i]->points[k].getVector3fMap () * -1.f;

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
    }

    recog_model->complete_cloud_.reset (new pcl::PointCloud<ModelT> (*complete_models_[i]));

    float extra_weight = 1.f;
    if(extra_weights_.size() == complete_models_.size())
      extra_weight = extra_weights_[i];

    if(object_ids_.size() == complete_models_.size()) {
      recog_model->id_s_ = object_ids_[i];
    }

    if(visible_normal_models_.size() == complete_models_.size())
    {

      pcl::PointCloud<pcl::Normal>::ConstPtr model_normals = visible_normal_models_[i];
      //pcl::ScopeTime t("Using model normals and checking nans");

      //std::cout << "Using model normals" << std::endl;
      recog_model->normals_.reset (new pcl::PointCloud<pcl::Normal> ());
      recog_model->normals_->points.resize(recog_model->cloud_->points.size ());

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

    } else {

      //std::cout << "Computing model normals" << std::endl;
      //pcl::ScopeTime t("Computing normals and checking nans");
      /*pcl::VoxelGrid<ModelT> voxel_grid;
      voxel_grid.setInputCloud (model);
      voxel_grid.setLeafSize (size_model, size_model, size_model);
      voxel_grid.filter (*(recog_model->cloud_));

      pcl::VoxelGrid<ModelT> voxel_grid2;
      voxel_grid2.setInputCloud (complete_model);
      voxel_grid2.setLeafSize (size_model, size_model, size_model);
      voxel_grid2.filter (*(recog_model->complete_cloud_));*/

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
        //pcl::ScopeTime t("Computing normals");
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

    //pcl::ScopeTime tt_nn("Computing outliers and explained points...");
    std::vector<int> explained_indices;
    std::vector<float> outliers_weight;
    std::vector<float> explained_indices_distances;
    std::vector<float> unexplained_indices_weights;

    std::vector<int> nn_indices;
    std::vector<float> nn_distances;

    //which point first from the scene is explained by a point j_k with dist d_k from the model
    std::map<int, boost::shared_ptr<std::vector<std::pair<int, float> > > > model_explains_scene_points;
    std::map<int, boost::shared_ptr<std::vector<std::pair<int, float> > > >::iterator it;

    outliers_weight.resize (recog_model->cloud_->points.size ());
    recog_model->outlier_indices_.resize (recog_model->cloud_->points.size ());

    {
      //pcl::ScopeTime t("NN");
      size_t o = 0;
      //Goes through the visible model points and finds scene points within a radius neighborhood
      //If in this neighborhood, there are no scene points, model point is considered outlier
      //If there are scene points, the model point is associated with the scene point, together with its distance
      //A scene point might end up being explained by the multiple model points
      for (size_t i = 0; i < recog_model->cloud_->points.size (); i++)
      {
        if (!scene_downsampled_tree_->radiusSearch (recog_model->cloud_->points[i], inliers_threshold_, nn_indices, nn_distances,
                                                    std::numeric_limits<int>::max ()))
        {

          //get NN
          std::vector<int> nn_indices_outlier;
          std::vector<float> nn_distances_outlier;
          scene_downsampled_tree_->nearestKSearch(recog_model->cloud_->points[i], 1, nn_indices_outlier, nn_distances_outlier);
          float d = sqrt(nn_distances_outlier[0]);
          float d_weight = 1.f + ( (d / inliers_threshold_));

          //outlier
          //std::cout << d_weight << " " << d << std::endl;
          assert(d_weight >= 1.0);
          outliers_weight[o] = regularizer_ * d_weight;
          recog_model->outlier_indices_[o] = static_cast<int> (i);
          o++;
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
    }

    recog_model->outliers_weight_ = (std::accumulate (outliers_weight.begin (), outliers_weight.end (), 0.f)
        / static_cast<float> (outliers_weight.size ()));
    if (outliers_weight.size () == 0)
      recog_model->outliers_weight_ = 1.f;

    int p = 0;
    float color_weight_inliers = 0;
    float color_weight_outliers = 0;
    std::vector<float> color_weight_inliers_vector;
    color_weight_inliers_vector.resize(model_explains_scene_points.size());
    bool color_exist = false;
    std::vector<int> map_cloud_points_to_model_rgb_values;

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

      float color_weight = 1.f;

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
      }

      explained_indices.push_back (it->first);

      //***** ATTENTION ''''*/
      if(color_exist) {
        explained_indices_distances.push_back (d_weight * dotp * extra_weight /* color_weight*/);
      } else {
        explained_indices_distances.push_back (d_weight * dotp * extra_weight);
      }

    }

    recog_model->model_constraints_value_ = getModelConstraintsValue(recog_model->complete_cloud_);
    recog_model->bad_information_ =  static_cast<int> (recog_model->outlier_indices_.size ());
    //recog_model->bad_information_ = 0;
    if (color_exist)
    {

      /*int rgb_size_hist = 256;
      Eigen::MatrixXf rgb_model, rgb_scene;
      computeRGBHistograms(model_rgb_values, rgb_model, rgb_size_hist);
      computeRGBHistograms(scene_rgb_values, rgb_scene, rgb_size_hist);

      //histogram specification, adapt model values to scene values
      Eigen::MatrixXf lookup;
      specifyRGBHistograms(rgb_scene, rgb_model, lookup);

      //with the lookup table, we can now transform model color space to scene model space
      for(size_t j=0; j < 3; j++) {
        for(size_t i=0; i < model_rgb_values.size(); i++)
        {
          int pos = std::floor (static_cast<float> (model_rgb_values[i][j]) / 255.f * rgb_size_hist);
          //model_rgb_values[i][j] = lookup(pos, j);
        }
      }*/

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

        /*float sigma = color_sigma_ * color_sigma_;
        yuvm[0] *= 0.5f;
        yuvs[0] *= 0.5f;
        float color_weight = std::exp ((-0.5f * (yuvm - yuvs).squaredNorm ()) / (sigma));
        color_weight_inliers_vector[i] = color_weight;

        //assert(color_weight >= 0);
        //assert(color_weight <= 1);
        color_weight_inliers += color_weight;*/
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
        /*model_hsv_values[i][1] *= 0.5f;
        model_hsv_values[i][2] *= 0.5f;

        scene_hsv_values[i][1] *= 0.5f;
        scene_hsv_values[i][2] *= 0.5f;*/

        //float color_weight = std::exp ((-0.5f * (model_hsv_values[i] - scene_hsv_values[i]).squaredNorm ()) / (sigma));
        float color_weight = std::exp ((-0.5f * (model_hsv_values[i][0] - scene_hsv_values[i][0]) * (model_hsv_values[i][0] - scene_hsv_values[i][0])) / (sigma));
        color_weight *= std::exp ((-0.5f * (model_hsv_values[i][1] - scene_hsv_values[i][1]) * (model_hsv_values[i][1] - scene_hsv_values[i][1])) / (sigma));
        color_weight *= std::exp ((-0.5f * (model_hsv_values[i][2] - scene_hsv_values[i][2]) * (model_hsv_values[i][2] - scene_hsv_values[i][2])) / (sigma));
        color_weight_inliers_vector[i] = color_weight;
        color_weight_inliers += color_weight;
      }

      /*if(static_cast<int> (explained_indices.size ()) > static_cast<int> (recog_model->outlier_indices_.size ()))
      {
        recog_model->bad_information_ = static_cast<float> (recog_model->outlier_indices_.size ())
                                                    / static_cast<float> (explained_indices.size ())
                                        * static_cast<int> (recog_model->outlier_indices_.size ());
      }*/

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
        model_hsv_values[i][0] = lookup_yuv(pos, 0) + 16;
      }

      Eigen::VectorXf model_hist, scene_hist;
      /*computeHueHistogram(model_hsv_values, model_hist);
      computeHueHistogram(scene_hsv_values, scene_hist);*/

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
      color_weight_inliers_vector.erase(color_weight_inliers_vector.begin(), color_weight_inliers_vector.begin() + (0.1f * color_weight_inliers_vector.size()));

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

      std::cout << median << " " << (1.f - std::exp ((-0.5f * (median - 0.5f) * (median - 0.5f)) / (0.5f * 0.5f))) * sign << std::endl;
      median += (1.f - std::exp ((-0.5f * (median - 0.5f) * (median - 0.5f)) / (0.5f * 0.5f))) * sign;

      recog_model->outliers_weight_ *= 1.f + (1.f - median);
      recog_model->model_constraints_value_ *= (1.f - std::min(median, 0.9f));
      for(size_t i=0; i < explained_indices_distances.size(); i++) {
        //explained_indices_distances[i] *= color_similarity;
        //explained_indices_distances[i] *= mean * color_similarity;
        explained_indices_distances[i] *= median * color_similarity;
      }

      //recog_model->bad_information_ += (static_cast<float> (explained_indices.size()) * (1.f - color_similarity));
      //recog_model->bad_information_ += (static_cast<float> (explained_indices.size()) * (1.f - color_similarity)) / regularizer_;
      //recog_model->bad_information_ += (static_cast<float> (recog_model->cloud_->points.size()) * (1.f - color_similarity)) / regularizer_;
      //recog_model->bad_information_ += (static_cast<float> (recog_model->cloud_->points.size()) * (1.f - color_similarity));
    }

    recog_model->explained_ = explained_indices;
    recog_model->explained_distances_ = explained_indices_distances;
    //std::cout << "Model:" << recog_model->complete_cloud_->points.size() << " " << recog_model->cloud_->points.size() << std::endl;
    return true;
  }

template<typename ModelT, typename SceneT>
  void
  faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::computeClutterCue (boost::shared_ptr<RecognitionModel> & recog_model)
  {
    if (detect_clutter_)
    {

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

          if (clusters_cloud_->points[recog_model->explained_[neighborhood_indices[i].second]].intensity != 0.f
              && (clusters_cloud_->points[recog_model->explained_[neighborhood_indices[i].second]].intensity
                  == clusters_cloud_->points[neighborhood_indices[i].first].intensity))
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

            //recog_model->unexplained_in_neighborhood_weights[p] = d_weight * dotp;
            recog_model->unexplained_in_neighborhood_weights[p] = 0.f; //ATTENTION!!
          }
          p++;
        }
      }

      recog_model->unexplained_in_neighborhood_weights.resize (p);
      recog_model->unexplained_in_neighborhood.resize (p);
    }
  }

template<typename ModelT, typename SceneT>
float
faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::HVGOBinaryOptimizer::computeBound(SAModel & model, int d) {

  float inliers_so_far = opt_->getExplainedValue();
  float hyp_penalty = opt_->getHypPenalty();
  float bad_info_so_far = opt_->getPreviousBadInfo();
  float dup_info_so_far = opt_->getDuplicity();
  float dup_cm_info_so_far = opt_->getDuplicityCM() * opt_->getOccupiedMultipleW();

  std::vector<double> explained_by_RM_local;
  opt_->getExplainedByRM(explained_by_RM_local);

  std::vector<int> indices_to_update_in_RM_local;
  for(size_t i=d; i < sol_length_; i++)
  {
    indices_to_update_in_RM_local.resize(recognition_models__[i]->explained_.size());

    float explained_for_this_hyp = opt_->getExplainedByIndices(recognition_models__[i]->explained_,
                                                                recognition_models__[i]->explained_distances_,
                                                                explained_by_RM_local,
                                                                indices_to_update_in_RM_local);

    inliers_so_far += explained_for_this_hyp;
    if(explained_for_this_hyp > 0) {
      for(size_t j=0; j < indices_to_update_in_RM_local.size(); j++) {
        explained_by_RM_local[recognition_models__[i]->explained_[indices_to_update_in_RM_local[j]]]
           = recognition_models__[i]->explained_distances_[indices_to_update_in_RM_local[j]];
      }
    }
    /*if(-inliers_so_far > incumbent_) {
      return static_cast<float>(-inliers_so_far);
    }*/
  }

  float min_bad_info = std::numeric_limits<float>::max();
  for(size_t i=d; i < sol_length_; i++)
  {
    if(recognition_models__[i]->bad_information_ < min_bad_info) {
      min_bad_info = recognition_models__[i]->bad_information_;
    }
  }
  bad_info_so_far += min_bad_info * recognition_models__[0]->outliers_weight_;

  //duplicity using map
  /*float min_duplicity, min_duplicity_cm;
  min_duplicity = min_duplicity_cm = std::numeric_limits<float>::max();
  //std::map<std::pair<int, int>, float>::iterator it;
  int active_in_x1 = 0;
  for(int i=0; i < d; i++) {
    if(model.solution_[i]) {
      active_in_x1++;
      for(int j=d; j < sol_length_; j++) {
        //std::pair<int, int> p = std::make_pair<int, int>(i,j);
        //it = intersection_.find(p);
        float inter = intersection_[i * recognition_models__.size() + j];
        if(min_duplicity > inter) {
          min_duplicity = inter;
        }

        inter = intersection_full_[i * recognition_models__.size() + j];
        if(min_duplicity_cm > inter) {
          min_duplicity_cm = inter;
        }
      }
    }
  }

  if(active_in_x1 > 0) {
    //std::cout << min_duplicity << " " << min_duplicity_cm << std::endl;
    dup_info_so_far += min_duplicity * 2;
    dup_cm_info_so_far += min_duplicity_cm * opt_->getOccupiedMultipleW() * 2;
  }*/

  return static_cast<float>(-inliers_so_far + hyp_penalty + bad_info_so_far + dup_cm_info_so_far + dup_info_so_far);
}

template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::HVGOBinaryOptimizer::search_recursive (mets::feasible_solution & sol, int d)
throw(mets::no_moves_error)
{
  if(d == (sol_length_))
    return;

  //do we need to expand this branch?
  //compute lower bound and compare with incumbent
  SAModel model;
  model.copy_from(sol);
  float lower_bound = computeBound(model, d);
  if(lower_bound > incumbent_)
  {
    if(d <= (sol_length_ * 0.1)) {
      std::cout << "LB gt than incumbent " << lower_bound << " " << incumbent_ << " " << d << " from " << sol_length_ << std::endl;
    }
    return;
  }

  //right branch, switch value of d hypothesis, evaluate and call recursive
  typedef mets::abstract_search<faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::move_manager> base_t;
  move m(d);
  m.apply(sol);
  base_t::solution_recorder_m.accept(sol);
  this->notify();

  if(incumbent_ > static_cast<mets::evaluable_solution&>(base_t::working_solution_m)
      .cost_function()) {
    incumbent_ = static_cast<mets::evaluable_solution&>(base_t::working_solution_m)
            .cost_function();
        std::cout << "Updating incumbent_ " << incumbent_ << std::endl;
  }

  search_recursive(sol, d+1);
  m.unapply(sol);

  //left branch, same solution without evaluating
  search_recursive(sol, d+1);

}

template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::HVGOBinaryOptimizer::computeStructures
                                                                                       (int size_full_occupancy, int size_explained)
 {

  intersection_.resize(recognition_models__.size() * recognition_models__.size(), 0);
  intersection_full_.resize(recognition_models__.size() * recognition_models__.size(), 0);

  for(size_t i=0; i < recognition_models__.size(); i++) {
    std::vector<int> complete_cloud_occupancy_by_RM;
    complete_cloud_occupancy_by_RM.resize(size_full_occupancy);

    std::vector<int> explained_by_RM;
    explained_by_RM.resize(size_explained);

    //fill complete_cloud_occupancy_by_RM with model i
    for (size_t kk = 0; kk < recognition_models__[i]->complete_cloud_occupancy_indices_.size (); kk++)
    {
      int idx = recognition_models__[i]->complete_cloud_occupancy_indices_[kk];
      complete_cloud_occupancy_by_RM[idx] = 1;
    }

    for (size_t kk = 0; kk < recognition_models__[i]->explained_.size (); kk++) {
      int idx = recognition_models__[i]->explained_[kk];
      explained_by_RM[idx] = 1;
    }

    for(size_t j=i; j < recognition_models__.size(); j++) {
      //count full model duplicates
      int c=0;
      for (size_t kk = 0; kk < recognition_models__[j]->complete_cloud_occupancy_indices_.size (); kk++)
      {
        int idx = recognition_models__[j]->complete_cloud_occupancy_indices_[kk];
        if(complete_cloud_occupancy_by_RM[idx] > 0) {
          c++;
        }
      }

      //std::pair<int,int> p = std::make_pair<int,int>(i,j);
      intersection_full_[i * recognition_models__.size() + j] = c;

      //count visible duplicates
      c = 0;
      for (size_t kk = 0; kk < recognition_models__[j]->explained_.size (); kk++)
      {
        int idx = recognition_models__[j]->explained_[kk];
        if(explained_by_RM[idx] > 0) {
          c++;
        }
      }

      intersection_[i * recognition_models__.size() + j] = c;
    }
  }
}

template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::HVGOBinaryOptimizer::search ()
throw(mets::no_moves_error)
{
  typedef mets::abstract_search<faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::move_manager> base_t;
  base_t::solution_recorder_m.accept(base_t::working_solution_m);

  best_cost_ =
    static_cast<mets::evaluable_solution&>(base_t::working_solution_m)
    .cost_function();

  std::cout << "Initial cost HVGOBinaryOptimizer:" << static_cast<float>(best_cost_) << std::endl;

  search_recursive(base_t::working_solution_m, 0);
}

template<typename ModelT, typename SceneT>
void
faat_pcl::GlobalHypothesesVerification<ModelT, SceneT>::move_manager::refresh(mets::feasible_solution& s)
{
  for (iterator ii = begin (); ii != end (); ++ii)
    delete (*ii);

  SAModel& model = dynamic_cast<SAModel&> (s);
  moves_m.clear();
  moves_m.resize(model.solution_.size() + model.solution_.size()*model.solution_.size());
  for (int ii = 0; ii != model.solution_.size(); ++ii)
    moves_m[ii]  = new move (ii);

  if(use_replace_moves_) {
    //based on s and the explained point intersection, create some replace_hyp_move
    //go through s and select active hypotheses and non-active hypotheses
    //check for each pair if the intersection is big enough
    //if positive, create a replace_hyp_move that will deactivate the act. hyp and activate the other one
    //MAYBE it would be interesting to allow this changes when the temperature is low or
    //there has been some iterations without an improvement
    std::vector<int> active, inactive;
    active.resize(model.solution_.size());
    inactive.resize(model.solution_.size());
    int nact, ninact;
    nact = ninact = 0;
    for(int i=0; i <static_cast<int>(model.solution_.size()); i++) {
      if(model.solution_[i]) {
        active[nact] = i;
        nact++;
      } else {
        inactive[ninact] = i;
        ninact++;
      }
    }

    active.resize(nact);
    inactive.resize(ninact);

    int nm=0;
    for(size_t i=0; i < active.size(); ++i) {
      for(size_t j=(i+1); j < inactive.size(); ++j) {
        std::map< std::pair<int, int>, bool>::iterator it;
        it = intersections_->find(std::make_pair<int, int>(std::min(active[i], inactive[j]),std::max(active[i], inactive[j])));
        assert(it != intersections_->end());
        if((*it).second) {
          moves_m[model.solution_.size() + nm] = new replace_hyp_move (active[i], inactive[j], model.solution_.size());
          nm++;
        }
      }
    }

    moves_m.resize(model.solution_.size() + nm);
  } else {
    moves_m.resize(model.solution_.size());
  }
  std::random_shuffle (moves_m.begin (), moves_m.end ());
  //std::cout << moves_m.size() << std::endl;
}

#define PCL_INSTANTIATE_faatGoHV(T1,T2) template class FAAT_REC_API faat_pcl::GlobalHypothesesVerification<T1,T2>;
