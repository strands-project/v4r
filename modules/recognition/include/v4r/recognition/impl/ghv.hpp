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

#include <v4r/common/normals.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/recognition/ghv.h>
#include <functional>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <pcl/common/angles.h>
#include <pcl/common/time.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <omp.h>
#include <iomanip>

namespace v4r {

template<typename ModelT, typename SceneT>
mets::gol_type
GHV<ModelT, SceneT>::evaluateSolution (const std::vector<bool> & active, int changed)
{
    int sign = 1;
    if ( !active[changed]) //it has been deactivated
    {
        sign = -1;
        tmp_solution_(changed) = 0;
    }
    else
        tmp_solution_(changed) = 1;

    float num_active_hypotheses = tmp_solution_.sum();
    #pragma omp parallel for schedule(dynamic)
    for(size_t row_id=0; row_id < scene_explained_weight_compressed_.rows(); row_id++)
    {
        double max = std::numeric_limits<double>::min();
        for(size_t col_id=0; col_id<active.size(); col_id++)
        {
            if ( active[col_id] && scene_explained_weight_compressed_(row_id,col_id)>max)
                max = scene_explained_weight_compressed_(row_id,col_id);
        }
        max_scene_explained_weight_(row_id)=max;
    }

    if(num_active_hypotheses > 0.5f)    // since we do not use integer
    {
        model_fitness_ = model_fitness_v_.dot(tmp_solution_);
        scene_fitness_ = max_scene_explained_weight_.sum();
        pairwise_cost_ = 0.5 * tmp_solution_.transpose() * intersection_cost_ * tmp_solution_;
        cost_ = -(   param_.regularizer_ * scene_fitness_ + model_fitness_
                   - param_.clutter_regularizer_ * pairwise_cost_ );
    }
    else
    {
        cost_ = model_fitness_ = scene_fitness_ =  pairwise_cost_ = 0.f;
    }

    if(cost_logger_) {
        cost_logger_->increaseEvaluated();
        cost_logger_->addCostEachTimeEvaluated(cost_);
    }

    return static_cast<mets::gol_type> (cost_); //return the dual to our max problem
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::computePairwiseIntersection()
{
    intersection_cost_ = Eigen::MatrixXf::Zero(recognition_models_.size(), recognition_models_.size());

    for(size_t i=1; i<recognition_models_.size(); i++)
    {
        HVRecognitionModel<ModelT> &rm_a = *recognition_models_[i];
        for(size_t j=0; j<i; j++)
        {
            const HVRecognitionModel<ModelT> &rm_b = *recognition_models_[j];

            size_t num_intersections = 0, total_rendered_points = 0;

            for(size_t view=0; view<rm_a.image_mask_.size(); view++)
            {
                for(int px=0; px<rm_a.image_mask_[view].size(); px++)
                {
                    if( rm_a.image_mask_[view][px] && rm_b.image_mask_[view][px])
                        num_intersections++;

                    if ( rm_a.image_mask_[view][px] || rm_b.image_mask_[view][px] )
                        total_rendered_points++;
                }
            }

            float conflict_cost = static_cast<float> (num_intersections) / total_rendered_points;
            intersection_cost_(i,j) = intersection_cost_(j,i) = conflict_cost;
        }

        if(!param_.visualize_pairwise_cues_)
            rm_a.image_mask_.clear();
    }
}


template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::removeModelsWithLowVisibility()
{
    recognition_models_map_.clear();
    recognition_models_map_.resize(recognition_models_.size());
    size_t kept=0;
    for(size_t i=0; i<recognition_models_.size(); i++)
    {
        const typename HVRecognitionModel<ModelT>::Ptr &rm = recognition_models_[i];

        if( (float)rm->visible_cloud_->points.size() / (float)rm->complete_cloud_->points.size() < param_.min_visible_ratio_)
            continue;

        recognition_models_[kept] = rm;
        recognition_models_map_[kept] = i;
        kept++;
    }
    recognition_models_.resize(kept);
    recognition_models_map_.resize(kept);
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::extractEuclideanClustersSmooth()

{
   size_t max_pts_per_cluster = std::numeric_limits<int>::max();

   CHECK (octree_scene_downsampled_ && octree_scene_downsampled_->getInputCloud ()->points.size () == scene_cloud_downsampled_->points.size ()
           && scene_cloud_downsampled_->points.size () == scene_normals_->points.size ());

  // Create a bool vector of processed point indices, and initialize it to false
  std::vector<bool> processed (scene_cloud_downsampled_->points.size (), false);
  std::vector<std::vector<int> > clusters;
  std::vector<int> nn_indices;
  std::vector<float> nn_distances;
  // Process all points in the indices vector
  for (size_t i = 0; i < scene_cloud_downsampled_->points.size (); ++i)
  {
    if (processed[i])
      continue;

    std::vector<size_t> seed_queue;
    size_t sq_idx = 0;
    seed_queue.push_back (i);

    processed[i] = true;

    while (sq_idx < seed_queue.size ())
    {

        size_t sidx = seed_queue[sq_idx];
      if (scene_normals_->points[ sidx ].curvature > param_.curvature_threshold_)
      {
        sq_idx++;
        continue;
      }

      // Search for sq_idx - scale radius with distance of point (due to noise)
      float radius = param_.cluster_tolerance_;
      float curvature_threshold = param_.curvature_threshold_;
      float eps_angle_threshold = param_.eps_angle_threshold_;

      if ( param_.z_adaptive_ )
      {
          radius = param_.cluster_tolerance_ * ( 1 + (std::max(scene_cloud_downsampled_->points[sidx].z, 1.f) - 1.f));
          curvature_threshold = param_.curvature_threshold_ * ( 1 + (std::max(scene_cloud_downsampled_->points[sidx].z, 1.f) - 1.f));
          eps_angle_threshold = param_.eps_angle_threshold_ * ( 1 + (std::max(scene_cloud_downsampled_->points[sidx].z, 1.f) - 1.f));
      }

      if (!octree_scene_downsampled_->radiusSearch (sidx, radius, nn_indices, nn_distances))
      {
        sq_idx++;
        continue;
      }

      for (size_t j = 1; j < nn_indices.size (); ++j) // nn_indices[0] should be sq_idx
      {
        if (processed[nn_indices[j]]) // Has this point been processed before ?
          continue;

        if (scene_normals_->points[nn_indices[j]].curvature > curvature_threshold)
          continue;

        //processed[nn_indices[j]] = true;
        // [-1;1]

        double dot_p = scene_normals_->points[ sidx ].normal[0] * scene_normals_->points[nn_indices[j]].normal[0]
            + scene_normals_->points[ sidx ].normal[1] * scene_normals_->points[nn_indices[j]].normal[1] + scene_normals_->points[sidx].normal[2]
            * scene_normals_->points[ nn_indices[j] ].normal[2];

        if (fabs (acos (dot_p)) < eps_angle_threshold)
        {
          processed[nn_indices[j]] = true;
          seed_queue.push_back (nn_indices[j]);
        }
      }

      sq_idx++;
    }

    // If this queue is satisfactory, add to the clusters
    if (seed_queue.size () >= param_.min_points_ && seed_queue.size () <= max_pts_per_cluster)
    {
      std::vector<int> r;
      r.resize (seed_queue.size ());
      for (size_t j = 0; j < seed_queue.size (); ++j)
        r[j] = seed_queue[j];

      std::sort (r.begin (), r.end ());
      r.erase (std::unique (r.begin (), r.end ()), r.end ());
      clusters.push_back (r); // We could avoid a copy by working directly in the vector
    }
  }

  scene_smooth_labels_.clear();
  scene_smooth_labels_.resize(scene_cloud_downsampled_->points.size(), 0);
  smooth_label_count_.resize( clusters.size() + 1);

  size_t total_labeled_pts = 0;
  for (size_t i = 0; i < clusters.size (); i++)
  {
      smooth_label_count_[i+1] = clusters[i].size();
      total_labeled_pts +=clusters[i].size();
      for (size_t j = 0; j < clusters[i].size (); j++)
      {
          int idx = clusters[i][j];
          scene_smooth_labels_[ idx ] = i+1;
      }
  }
  smooth_label_count_[0] = scene_cloud_downsampled_->points.size() - total_labeled_pts;
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::individualRejection()
{
    // remove badly explained object hypotheses
    size_t kept=0;
    for (size_t i = 0; i < recognition_models_.size (); i++)
    {
        HVRecognitionModel<ModelT> &rm = *recognition_models_[i];

        if (param_.check_smooth_clusters_)
        {
            bool is_rejected_due_to_smooth_cluster_check = false;
            for(size_t cluster_id=1; cluster_id<rm.explained_pts_per_smooth_cluster_.size(); cluster_id++)  // don't check label 0
            {
                if ( smooth_label_count_[cluster_id] > 100 &&  rm.explained_pts_per_smooth_cluster_[cluster_id] > 100 &&
                     (float)(rm.explained_pts_per_smooth_cluster_[cluster_id]) / smooth_label_count_[cluster_id] < param_.min_ratio_cluster_explained_ )
                {
                    is_rejected_due_to_smooth_cluster_check = true;
                    break;
                }
            }
            if (is_rejected_due_to_smooth_cluster_check)
                continue;
        }

        float visible_ratio = rm.visible_cloud_->points.size() / (float)rm.complete_cloud_->points.size();
        float model_fitness = rm.model_fit_ / rm.visible_cloud_->points.size();

        CHECK(param_.min_model_fitness_lower_bound_ <= param_.min_model_fitness_upper_bound_); // scale model fitness threshold with the visible ratio of model. Highly occluded objects need to have a very strong evidence

        float scale = std::min<float>( 1.f, visible_ratio/0.5f );
        float range = param_.min_model_fitness_upper_bound_ - param_.min_model_fitness_lower_bound_;
        float model_fitness_threshold = param_.min_model_fitness_upper_bound_ - scale * range;

        if( model_fitness > model_fitness_threshold)
        {
            recognition_models_[kept] = recognition_models_[i];
            recognition_models_map_[kept] = recognition_models_map_[i];
            scene_explained_weight_.col(kept).swap( scene_explained_weight_.col(i) );
            kept++;
        }
    }
    std::cout << "Rejected " << recognition_models_.size() - kept << " (out of " << recognition_models_.size() << ") hypotheses due to low model fitness score." << std::endl;
    recognition_models_.resize(kept);
    recognition_models_map_.resize(kept);
    size_t num_rows = scene_explained_weight_.rows();
    scene_explained_weight_.conservativeResize(num_rows, kept);
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::initialize()
{
    solution_.clear ();
    solution_.resize (recognition_models_.size (), false);

    removeSceneNans();

    {
        pcl::ScopeTime t("Computing octree");
        octree_scene_downsampled_.reset(new pcl::octree::OctreePointCloudSearch<SceneT>(0.01f));
        octree_scene_downsampled_->setInputCloud(scene_cloud_downsampled_);
        octree_scene_downsampled_->addPointsFromInputCloud();
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            pcl::ScopeTime t("pose refinement and computing visible model points");
            computeVisibleModelsAndRefinePose();
        }

        #pragma omp section
        {
            if(param_.check_smooth_clusters_)
                extractEuclideanClustersSmooth();
        }
    }

    removeModelsWithLowVisibility();
    ColorTransformOMP::initializeLUT();

    // just initializing to some random negative value to know which ones are explained later on by a simple check for a positive value
    //NOTE: We could initialize to nan or quite_nan but not sure if the comparison behaves the same on all platforms
    scene_model_sqr_dist_ = -1000.f * Eigen::MatrixXf::Ones(scene_cloud_downsampled_->points.size(), recognition_models_.size());

    #pragma omp parallel sections
    {
        #pragma omp section
        if(!param_.ignore_color_even_if_exists_)
        {
            pcl::ScopeTime t("Converting scene color values");
            convertToLABcolor(*scene_cloud_downsampled_, scene_color_channels_);
        }

        #pragma omp section
        {
            pcl::ScopeTime t("Converting model color values");
            for (size_t i = 0; i < recognition_models_.size(); i++)
            {
                HVRecognitionModel<ModelT> &rm = *recognition_models_[i];
                removeNanNormals(rm);

                if(!param_.ignore_color_even_if_exists_)
                    convertToLABcolor(*rm.visible_cloud_, rm.pt_color_);
            }
        }

        #pragma omp section
        {
            pcl::ScopeTime t("Computing model to scene fitness");
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < recognition_models_.size (); i++)
            {
                HVRecognitionModel<ModelT> &rm = *recognition_models_[i];
                computeModel2SceneDistances(rm, i);
            }
        }
    }


    if(param_.use_histogram_specification_)
    {
        pcl::ScopeTime t("Computing histogramm specification");
        for (size_t i = 0; i < recognition_models_.size (); i++)
        {
            HVRecognitionModel<ModelT> &rm = *recognition_models_[i];
            computeLoffset(rm, i);
        }
    }

    {
        pcl::ScopeTime t("Computing fitness score between models and scene");
        scene_explained_weight_ = Eigen::MatrixXf::Zero(scene_cloud_downsampled_->points.size(), recognition_models_.size());

        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < recognition_models_.size (); i++){
            HVRecognitionModel<ModelT> &rm = *recognition_models_[i];
            computeModel2SceneFitness(rm, i);
        }
    }

    individualRejection();

    if(recognition_models_.empty())
        return;

    {
        pcl::ScopeTime t("Compressing scene explained matrix");
        // remove rows of scene explained matrix, whose point is not explained by any hypothesis. Because it is usually very sparse and would take a lot of computation time.
        scene_explained_weight_compressed_ = scene_explained_weight_;
        Eigen::VectorXf min_tmp = scene_explained_weight_.rowwise().maxCoeff();
        size_t kept=0;
        for(size_t pt=0; pt<scene_cloud_downsampled_->points.size(); pt++)
        {
            if( min_tmp(pt) > std::numeric_limits<float>::epsilon() )
            {
                scene_explained_weight_compressed_.row(kept).swap(scene_explained_weight_compressed_.row(pt));
                kept++;
            }
        }
        scene_explained_weight_compressed_.conservativeResize(kept, scene_explained_weight_.cols());

        if(!param_.visualize_go_cues_ && !param_.visualize_model_cues_)
            scene_explained_weight_.resize(0,0);    // not needed any more

        if(!kept)
            return;
    }

    //store model fitness into vector
    model_fitness_v_ = Eigen::VectorXf(recognition_models_.size ());
    for (size_t i = 0; i < recognition_models_.size (); i++)
        model_fitness_v_[i] = recognition_models_[i]->model_fit_;

    {
        pcl::ScopeTime t("Computing pairwise intersection");
        computePairwiseIntersection();
    }

    if(param_.visualize_model_cues_)
    {
        for (size_t i = 0; i < recognition_models_.size (); i++)
            visualizeGOCuesForModel(*recognition_models_[i], i);
    }

    if(param_.visualize_pairwise_cues_)
        visualizePairwiseIntersection();
}

template<typename ModelT, typename SceneT>
std::vector<bool>
GHV<ModelT, SceneT>::optimize ()
{
    std::vector<bool> temp_solution ( recognition_models_.size(), param_.initial_status_);
    if(param_.initial_status_)
        tmp_solution_ = Eigen::VectorXf::Ones (recognition_models_.size());
    else
        tmp_solution_ = Eigen::VectorXf::Zero (recognition_models_.size());

    GHVSAModel<ModelT, SceneT> model;

    double initial_cost  = 0.f;
    max_scene_explained_weight_ = Eigen::VectorXf::Zero(scene_cloud_downsampled_->points.size());
    model.cost_ = static_cast<mets::gol_type> ( initial_cost );
    model.setSolution (temp_solution);
    model.setOptimizer (this);

    GHVSAModel<ModelT, SceneT> *best = new GHVSAModel<ModelT, SceneT> (model);
    GHVmove_manager<ModelT, SceneT> neigh ( recognition_models_.size (), param_.use_replace_moves_);
    neigh.setExplainedPointIntersections(intersection_cost_);

    //mets::best_ever_solution best_recorder (best);
    cost_logger_.reset(new GHVCostFunctionLogger<ModelT, SceneT>(*best));
    mets::noimprove_termination_criteria noimprove (param_.max_iterations_);

    if(param_.visualize_go_cues_)
        cost_logger_->setVisualizeFunction(visualize_cues_during_logger_);

    switch( param_.opt_type_ )
    {
        case OptimizationType::LocalSearch:
        {
            mets::local_search<GHVmove_manager<ModelT, SceneT> > local ( model, *(cost_logger_.get()), neigh, 0, false);
            {
                pcl::ScopeTime t ("local search...");
                local.search ();
            }
            break;
        }
        case OptimizationType::TabuSearch:
        {
            //Tabu search
            //mets::simple_tabu_list tabu_list ( initial_solution.size() * sqrt ( 1.0*initial_solution.size() ) ) ;
            mets::simple_tabu_list tabu_list ( 5 * temp_solution.size()) ;
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
        case OptimizationType::TabuSearchWithLSRM:
        {
            GHVmove_manager<ModelT, SceneT> neigh4 (recognition_models_.size(), false);
            neigh4.setExplainedPointIntersections(intersection_cost_);

            mets::simple_tabu_list tabu_list ( temp_solution.size() * sqrt ( 1.0*temp_solution.size() ) ) ;
            mets::best_ever_criteria aspiration_criteria ;
            mets::tabu_search<GHVmove_manager<ModelT, SceneT> > tabu_search(model,  *(cost_logger_.get()), neigh4, tabu_list, aspiration_criteria, noimprove);
            //mets::tabu_search<move_manager> tabu_search(model, best_recorder, neigh, tabu_list, aspiration_criteria, noimprove);

            {
                pcl::ScopeTime t("TABU search + LS (RM)...");
                try { tabu_search.search (); }
                catch (mets::no_moves_error e) { }

                std::cout << "Tabu search finished... starting LS with RM" << std::endl;

                //after TS, we do LS with RM
                GHVmove_manager<ModelT, SceneT> neigh4RM (recognition_models_.size(), true);
                neigh4RM.setExplainedPointIntersections(intersection_cost_);

                mets::local_search<GHVmove_manager<ModelT, SceneT> > local ( model, *(cost_logger_.get()), neigh4RM, 0, false);
                {
                    pcl::ScopeTime t_local_search ("local search...");
                    local.search ();
                    (void)t_local_search;
                }
            }
            break;

        }
        case OptimizationType::SimulatedAnnealing:
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
        default:
            throw std::runtime_error("Specified optimization type not implememted!");
    }

    best_seen_ = static_cast<const GHVSAModel<ModelT, SceneT>&> (cost_logger_->best_seen());
    std::cout << "*****************************" << std::endl
              << "Final cost:" << best_seen_.cost_ << std::endl
              << "Number of ef evaluations:" << cost_logger_->getTimesEvaluated() << std::endl
              << "Number of accepted moves:" << cost_logger_->getAcceptedMovesSize() << std::endl
              << "*****************************" << std::endl;

    delete best;
    return best_seen_.solution_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::verify()
{
    {
        pcl::ScopeTime t("initialization");
        initialize();
    }

    if(param_.visualize_go_cues_)
        visualize_cues_during_logger_ = boost::bind(&GHV<ModelT, SceneT>::visualizeGOCues, this, _1, _2, _3);

    {
    pcl::ScopeTime t("Optimizing object hypotheses verification cost function");
    std::vector<bool> solution = optimize ();

    // since we remove hypothese containing too few visible points - mask size does not correspond to recognition_models size anymore --> therefore this map stuff
    for(size_t i=0; i<solution_.size(); i++)
        solution_[ i ] = false;

    for(size_t i=0; i<solution.size(); i++)
        solution_[ recognition_models_map_[i] ] = solution[i];
    }

    cleanUp();
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::removeSceneNans()
{
    if(!scene_and_normals_set_from_outside_ || scene_cloud_downsampled_->points.size() != scene_normals_->points.size())
    {
        size_t kept = 0;
        for (size_t i = 0; i < scene_cloud_downsampled_->points.size (); i++) {
            if ( pcl::isFinite( scene_cloud_downsampled_->points[i]) ) {
                scene_cloud_downsampled_->points[kept] = scene_cloud_downsampled_->points[i];
                scene_sampled_indices_[kept] = scene_sampled_indices_[i];
                kept++;
            }
        }

        scene_sampled_indices_.resize(kept);
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
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::specifyHistogram (const Eigen::MatrixXf & src, const Eigen::MatrixXf & dst, Eigen::MatrixXf & lookup) const
{
    if( src.cols() != dst.cols() || src.rows() != dst.rows() )
        throw std::runtime_error ("The given matrices to speficyHistogram must have the same size!");

    //normalize histograms
    size_t dims = src.cols();
    size_t bins = src.rows();

    Eigen::MatrixXf src_normalized(bins, dims), dst_normalized (bins, dims);

    for(size_t i=0; i < dims; i++) {
        src_normalized.col(i) = src.col(i) / src.col(i).sum();
        dst_normalized.col(i) = dst.col(i) / dst.col(i).sum();
    }

    Eigen::MatrixXf src_cumulative = Eigen::MatrixXf::Zero(bins, dims);
    Eigen::MatrixXf dst_cumulative = Eigen::MatrixXf::Zero(bins, dims);
    lookup = Eigen::MatrixXf::Zero(bins, dims);

    for (size_t dim = 0; dim < dims; dim++)
    {
        src_cumulative (0, dim) = src_normalized (0, dim);
        dst_cumulative (0, dim) = dst_normalized (0, dim);
        for (size_t bin = 1; bin < bins; bin++)
        {
            src_cumulative (bin, dim) = src_cumulative (bin - 1, dim) + src_normalized (bin, dim);
            dst_cumulative (bin, dim) = dst_cumulative (bin - 1, dim) + dst_normalized (bin, dim);
        }

        int last = 0;
        for (int bin = 0; bin < bins; bin++)
        {
            for (int z = last; z < bins; z++)
            {
                if (src_cumulative (z, dim) - dst_cumulative (bin, dim) >= 0)
                {
                    if (z > 0 && (dst_cumulative (bin, dim) - src_cumulative (z - 1, dim)) < (src_cumulative (z, dim) - dst_cumulative (bin, dim)))
                        z--;

                    lookup(bin, dim) = z;
                    last = z;
                    break;
                }
            }
        }

        int min = 0;
        for (int k = 0; k < bins; k++)
        {
            if (lookup (k, dim) != 0)
            {
                min = lookup (k, dim);
                break;
            }
        }

        for (int k = 0; k < bins; k++)
        {
            if (lookup (k, dim) == 0)
                lookup (k, dim) = min;
            else
                break;
        }

        //max mapping extension
        int max = 0;
        for (int k = (bins - 1); k >= 0; k--)
        {
            if (lookup (k, dim) != 0)
            {
                max = lookup (k, dim);
                break;
            }
        }

        for (int k = (bins - 1); k >= 0; k--)
        {
            if (lookup (k, dim) == 0)
                lookup (k, dim) = max;
            else
                break;
        }
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
GHV<ModelT, SceneT>::computeModel2SceneDistances(HVRecognitionModel<ModelT> &rm, int model_id)
{
    rm.explained_pts_per_smooth_cluster_.clear();
    rm.explained_pts_per_smooth_cluster_.resize(smooth_label_count_.size(), 0);

    rm.model_scene_c_.resize( rm.visible_cloud_->points.size () * param_.knn_inliers_ );
    size_t kept=0;

    for (size_t m_pt_id = 0; m_pt_id < rm.visible_cloud_->points.size (); m_pt_id++)
    {
        std::vector<int> nn_indices;
        std::vector<float> nn_sqr_distances;
        octree_scene_downsampled_->nearestKSearch(rm.visible_cloud_->points[m_pt_id], param_.knn_inliers_, nn_indices, nn_sqr_distances);

        for (size_t k = 0; k < nn_indices.size(); k++)
        {
              int sidx = nn_indices[ k ];
              double sqr_3D_dist = nn_sqr_distances[k];

//              if (sqr_3D_dist > ( 3 * 3 * param_.inliers_threshold_ * param_.inliers_threshold_ ) )
//                  continue;

              pcl::Correspondence &c = rm.model_scene_c_[ kept ];
              c.index_query = m_pt_id;
              c.index_match = sidx;
              c.distance = sqr_3D_dist;



              double old_s_m_dist = scene_model_sqr_dist_(sidx, model_id);
              if ( old_s_m_dist < -1.f || sqr_3D_dist < old_s_m_dist)
              {
                  scene_model_sqr_dist_(sidx, model_id) = sqr_3D_dist;

                  if( param_.check_smooth_clusters_ && sqr_3D_dist < (param_.occlusion_thres_ * param_.occlusion_thres_ * 1.5 * 1.5)
                          && ( old_s_m_dist < -1.f || old_s_m_dist >= (param_.occlusion_thres_ * param_.occlusion_thres_ * 1.5 * 1.5) )) // if point is not already taken
                  {
                      int l = scene_smooth_labels_[sidx];
                      rm.explained_pts_per_smooth_cluster_[l] ++;
                  }
              }

              kept++;
        }
    }
    rm.model_scene_c_.resize(kept);
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::computeModel2SceneFitness(HVRecognitionModel<ModelT> &rm, size_t model_idx)
{
    Eigen::VectorXf rm_scene_expl_weight = -1000.f * Eigen::VectorXf::Ones (scene_explained_weight_.rows());
    Eigen::VectorXf modelFit             = -1000.f * Eigen::VectorXf::Ones (rm.visible_cloud_->points.size());

    Eigen::VectorXi modelSceneCorrespondence; // saves the correspondence of each visible model point to its closest scene point (weighted by 3D Euclidean Distance and color). Only used for visualization.
    if(param_.visualize_model_cues_)
        modelSceneCorrespondence = -1 * Eigen::VectorXi::Ones (rm.visible_cloud_->points.size()); // negative value means no correspondence

    double w3d = 1 / (param_.inliers_threshold_ * param_.inliers_threshold_);
    double w_color_AB = 1 / (param_.color_sigma_ab_ * param_.color_sigma_ab_);
    double w_color_L = 1 / (param_.color_sigma_l_ * param_.color_sigma_l_);
    double w_normals = 1 / (param_.sigma_normals_deg_ * param_.sigma_normals_deg_);

    for(size_t i=0; i<rm.model_scene_c_.size(); i++)
    {
        const pcl::Correspondence &c = rm.model_scene_c_[i];
        double sqr_3D_dist = c.distance;
        int sidx = c.index_match;
        int midx = c.index_query;

        Eigen::Vector3f normal_m = rm.visible_cloud_normals_->points[midx].getNormalVector3fMap();
        Eigen::Vector3f normal_s = scene_normals_->points[sidx].getNormalVector3fMap();

        normal_m.normalize();
        normal_s.normalize();
        double dotp = normal_m.dot(normal_s);
        if(dotp>0.999f)
            dotp = 0.999f;
        if(dotp<-0.999f)
            dotp = -0.999f;
        double acoss = acos (dotp);
        double normal_angle_deg = pcl::rad2deg( acoss );
        double dist = w3d * sqr_3D_dist + w_normals * normal_angle_deg * normal_angle_deg;

        const Eigen::VectorXf &color_m = rm.pt_color_.row( midx );
        const Eigen::VectorXf &color_s = scene_color_channels_.row( sidx );


        if(param_.color_space_ == ColorTransformOMP::LAB)
        {
            double Ls = color_s(0);
            double As = color_s(1);
            double Bs = color_s(2);
            double Lm = std::max(0.f, std::min(100.f, rm.L_value_offset_ + color_m(0)) );
            double Am = color_m(1);
            double Bm = color_m(2);

            double sqr_color_dist_AB = ( (As-Am)*(As-Am)+(Bs-Bm)*(Bs-Bm) );
            dist += w_color_AB * sqr_color_dist_AB;

            double sqr_color_dist_L = ( (Ls-Lm)*(Ls-Lm) );
            dist += w_color_L * sqr_color_dist_L;
        }
        else
            throw std::runtime_error("Desired color space not implemented so far!");

        float old_scene2model_dist = rm_scene_expl_weight(sidx); ///NOTE: negative if no points explains it yet
        if ( old_scene2model_dist < -1.f || dist<old_scene2model_dist)
            rm_scene_expl_weight(sidx) = dist;    // 0 for perfect fit

        float old_model_pt_dist = modelFit(midx);
        if( old_model_pt_dist < -1.f || dist < old_model_pt_dist )
        {
            modelFit(midx) = dist;
            if(param_.visualize_model_cues_)
                modelSceneCorrespondence(midx) = sidx;
        }
    }

    // now we compute the exponential of the distance to bound it between 0 and 1 (whereby 1 means perfect fit and 0 no fit)
    #pragma omp parallel for schedule(dynamic)
    for(size_t sidx=0; sidx < rm_scene_expl_weight.rows(); sidx++)
    {
        float fit = rm_scene_expl_weight(sidx);
        fit < -1.f ? fit = 0.f : fit = exp(-fit);
        rm_scene_expl_weight(sidx) = fit;
    }

    #pragma omp parallel for schedule(dynamic)
    for(size_t midx=0; midx < modelFit.rows(); midx++)
    {
        float fit = modelFit(midx);
        fit < -1.f ? fit = 0.f : fit = exp(-fit);
        modelFit(midx) = fit;
    }

    rm.model_fit_ = modelFit.sum();
    scene_explained_weight_.col(model_idx) = rm_scene_expl_weight;

    rm.model_scene_c_.clear(); // not needed any more
    if(param_.visualize_model_cues_)    // we store the fit for each model point in the correspondences vector
    {
        rm.model_scene_c_.resize(rm.visible_cloud_->points.size());
        for(size_t midx=0; midx<rm.visible_cloud_->points.size(); midx++)
        {
            int sidx = modelSceneCorrespondence(midx);
            pcl::Correspondence &c = rm.model_scene_c_[midx];
            c.index_query = midx;
            c.index_match = sidx;
            if( sidx>0 )
                c.weight = modelFit(midx);
            else
                c.weight = 0.f;
        }
    }
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::computeLoffset(HVRecognitionModel<ModelT> &rm, int model_id) const
{
    // pre-allocate memory
    size_t kept = 0;
    for(size_t sidx=0; sidx<scene_model_sqr_dist_.rows(); sidx++)
    {
        if( scene_model_sqr_dist_(sidx,model_id) > -1.f )
            kept++;
    }

    Eigen::MatrixXf croppedSceneColorMatrix (kept, scene_color_channels_.cols());
    kept = 0;
    for(size_t sidx=0; sidx<scene_model_sqr_dist_.rows(); sidx++)
    {
        if( scene_model_sqr_dist_(sidx,model_id) > -1.f )
        {
            croppedSceneColorMatrix.row(kept) = scene_color_channels_.row(sidx);
            kept++;
        }
    }

    Eigen::MatrixXf histLm, histLs;
    computeHistogram(rm.pt_color_.col(0), histLm, bins_, Lmin_, Lmax_);
    computeHistogram(croppedSceneColorMatrix.col(0), histLs, bins_, Lmin_, Lmax_);

    Eigen::VectorXf histLs_normalized(bins_), histLm_normalized (bins_);
    histLs_normalized = histLs.col(0);
    float Ls_sum = histLs_normalized.sum();
    histLs_normalized /= Ls_sum;
    histLm_normalized = histLm.col(0);
    float Lm_sum = histLm_normalized.sum();
    histLm_normalized /= Lm_sum;

    float best_corr = computeHistogramIntersection(histLs_normalized, histLm_normalized);
    int best_shift = 0;

    Eigen::VectorXf histLm_normalized_shifted_old = histLm_normalized;
    Eigen::VectorXf histLm_normalized_shifted;

    for(int shift=1; shift<std::floor(bins_/2); shift++) // shift right
    {
        shiftHistogram(histLm_normalized_shifted_old, histLm_normalized_shifted, true);
        float corr = computeHistogramIntersection( histLs_normalized,  histLm_normalized_shifted);
        if (corr>best_corr)
        {
            best_corr = corr;
            best_shift = shift;
        }
        histLm_normalized_shifted_old = histLm_normalized_shifted;
    }

    histLm_normalized_shifted_old = histLm_normalized;
    for(int shift=1; shift<std::floor(bins_/2); shift++) // shift left
    {
        shiftHistogram(histLm_normalized_shifted_old, histLm_normalized_shifted, false);
        float corr = computeHistogramIntersection( histLs_normalized,  histLm_normalized_shifted);
        if (corr>best_corr)
        {
            best_corr = corr;
            best_shift = -shift;
        }
        histLm_normalized_shifted_old = histLm_normalized_shifted;
    }

    rm.L_value_offset_ = best_shift * (Lmax_ - Lmin_) / bins_;
}

//######### VISUALIZATION FUNCTIONS #####################
template<>
void
GHV<pcl::PointXYZ, pcl::PointXYZ>::visualizeGOCuesForModel(const HVRecognitionModel<pcl::PointXYZ> &rm, int model_id) const
{
    (void)rm;
    (void)model_id;
    std::cerr << "The visualization function is not defined for the chosen Point Cloud Type!" << std::endl;
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::visualizeGOCuesForModel(const HVRecognitionModel<ModelT> &rm, int model_id) const
{
    if(!rm_vis_) {
        rm_vis_.reset (new pcl::visualization::PCLVisualizer ("model cues"));
        rm_vis_->createViewPort(0   , 0   , 0.25,0.5 , rm_v1);
        rm_vis_->createViewPort(0.25, 0   , 0.50,0.5 , rm_v2);
        rm_vis_->createViewPort(0.50, 0   , 0.75,0.5 , rm_v3);
        rm_vis_->createViewPort(0.75, 0   , 1   ,0.5 , rm_v4);

        rm_vis_->createViewPort(0   , 0.5 , 0.25,1   , rm_v5);
        rm_vis_->createViewPort(0.25, 0.5 , 0.50,1   , rm_v6);
        rm_vis_->createViewPort(0.50, 0.5 , 0.75,1   , rm_v7);

        rm_vis_->createViewPort(0.75, 0.5 , 0.875   , 0.75   , rm_v11);
        rm_vis_->createViewPort(0.875, 0.5 , 1   , 0.75   , rm_v12);
        rm_vis_->createViewPort(0.75, 0.75 , 0.875   ,1   , rm_v11);
        rm_vis_->createViewPort(0.875, 0.75 , 1   ,1   , rm_v12);

        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v1);
        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v2);
        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v3);
        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v4);
        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v5);
        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v6);
        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v7);
    }

    rm_vis_->removeAllPointClouds();
    rm_vis_->removeAllShapes();

    if(!param_.vis_for_paper_)
        rm_vis_->addText("scene",10,10,12,1,1,1,"scene",rm_v1);

    rm_vis_->addPointCloud(scene_cloud_downsampled_, "scene1",rm_v1);

#ifdef L_HIST
    // compute color histogram for visible model points
    {
        pcl::PointCloud<ModelT> m_cloud_orig, m_cloud_color_reg, m_cloud_color_reg2;
        m_cloud_orig.points.resize( rm.visible_indices_.size() );
        m_cloud_color_reg.points.resize( rm.visible_indices_.size() );
        m_cloud_color_reg2.points.resize( rm.visible_indices_.size() );

        Eigen::MatrixXf colorVisibleCloud (rm.visible_indices_.size(), rm.pt_color_.cols());
        for(size_t i=0; i<rm.visible_indices_.size(); i++)
        {
            int idx = rm.visible_indices_[i];
            colorVisibleCloud.row(i) = rm.pt_color_.row(idx);

            ModelT &m = m_cloud_orig.points[i];
            ModelT &m2 = m_cloud_color_reg2.points[i];
            m = rm.visible_cloud_->points[i];
            m2 = rm.visible_cloud_->points[i];
            ColorTransform::CIELAB2RGB( colorVisibleCloud(i,0), colorVisibleCloud(i,1), colorVisibleCloud(i,2), m.r, m.g, m.b);

            float l_w_offset = std::max(0.f, std::min(100.f, rm.L_value_offset_ + colorVisibleCloud(i,0)) );
            ColorTransform::CIELAB2RGB( l_w_offset, colorVisibleCloud(i,1), colorVisibleCloud(i,2), m2.r, m2.g, m2.b);
        }

        Eigen::MatrixXf histLm, histLs;
        computeHistogram(colorVisibleCloud.col(0), histLm, bins_, Lmin_, Lmax_);

        // pre-allocate memory
        size_t kept = 0;
        for(size_t sidx=0; sidx<scene_explained_weight_.rows(); sidx++)
        {
            if(scene_explained_weight_(sidx,model_id) > std::numeric_limits<float>::epsilon())
                kept++;
        }
        Eigen::MatrixXf croppedSceneColorMatrix (kept, scene_color_channels_.cols());
        pcl::PointCloud<SceneT> s_cloud_orig, s_cloud_color_reg;
        s_cloud_orig.points.resize( kept );
        s_cloud_color_reg.points.resize( kept );
        kept = 0;
        for(size_t sidx=0; sidx<scene_explained_weight_.rows(); sidx++)
        {
            if(scene_explained_weight_(sidx,model_id) > std::numeric_limits<float>::epsilon())
            {
                croppedSceneColorMatrix.row(kept) = scene_color_channels_.row(sidx);
                SceneT &s = s_cloud_orig.points[kept];
                s = scene_cloud_downsampled_->points[sidx];
                ColorTransform::CIELAB2RGB( croppedSceneColorMatrix(kept,0), croppedSceneColorMatrix(kept,1), croppedSceneColorMatrix(kept,2), s.r, s.g, s.b);
                kept++;
            }
        }

//        for(size_t idx=0; idx<rm.model_scene_c_.size(); idx++)
//        {
//            int scene_idx = rm.model_scene_c_[idx].index_match;
//            croppedSceneColorMatrix.row(idx) = scene_color_channels_.row(scene_idx);

//            SceneT &s = s_cloud_orig.points[idx];
//            s = scene_cloud_downsampled_->points[scene_idx];
//            ColorTransform::CIELAB2RGB( croppedSceneColorMatrix(idx,0), croppedSceneColorMatrix(idx,1), croppedSceneColorMatrix(idx,2), s.r, s.g, s.b);
//        }
        computeHistogram(croppedSceneColorMatrix.col(0), histLs, bins_, Lmin_, Lmax_);
        Eigen::VectorXf histLs_normalized(bins_), histLm_normalized (bins_);
        histLs_normalized = histLs.col(0);
        float Ls_sum = histLs_normalized.sum();
        histLs_normalized /= Ls_sum;
        histLm_normalized = histLm.col(0);
        float Lm_sum = histLm_normalized.sum();
        histLm_normalized /= Lm_sum;

        Eigen::MatrixXf lookUpTable;
        specifyHistogram(histLs_normalized, histLm_normalized, lookUpTable);
        float bin_size = (Lmax_-Lmin_) / bins_;
        double fitness_orig=0.f, fitness_reg = 0.f, fitness_reg2 = 0.f;
        for(size_t idx=0; idx<rm.model_scene_c_.size(); idx++)
        {
            const pcl::Correspondence &c = rm.model_scene_c_[idx];
            int sidx = c.index_match;
            int midx = c.index_query;

            if(sidx<0)
                continue;

            float Ls = scene_color_channels_(sidx,0);
            float Lm = colorVisibleCloud(midx, 0);

            int pos = std::floor( (Lm - Lmin_) / bin_size);

            if(pos < 0)
                pos = 0;

            if(pos > (int)bins_)
                pos = bins_ - 1;

            float Lm_reg = lookUpTable(pos,0);

            ModelT &m = m_cloud_color_reg.points[midx];
            m = rm.visible_cloud_->points[midx];
            ColorTransform::CIELAB2RGB( Lm_reg, colorVisibleCloud(midx,1), colorVisibleCloud(midx,2), m.r, m.g, m.b);

            float Lm_reg2 = std::max(0.f, std::min(100.f, rm.L_value_offset_ + Lm) );
            fitness_orig += exp(- (Ls - Lm)*(Ls - Lm)/ (param_.color_sigma_ab_ * param_.color_sigma_ab_) );
            fitness_reg += exp(- (Ls - Lm_reg)*(Ls - Lm_reg)/ (param_.color_sigma_ab_ * param_.color_sigma_ab_) );
            fitness_reg2 += exp(- (Ls - Lm_reg2)*(Ls - Lm_reg2)/ (param_.color_sigma_ab_ * param_.color_sigma_ab_) );
        }

        std::cout << "fitness original: " << fitness_orig/rm.visible_cloud_->points.size() << "; after registration: " <<
                     fitness_reg/rm.visible_cloud_->points.size() << "; after registration2: " <<
                     fitness_reg2/rm.visible_cloud_->points.size() << "." << std::endl;

        rm_vis_->addText("explained scene cloud original", 10, 10, 10, 1,1,1,"s_cloud_orig", rm_v9);
        rm_vis_->addText("model cloud color registered", 10, 10, 10, 1,1,1,"s_cloud_color_reg2", rm_v10);
        rm_vis_->addText("model cloud original", 10, 10, 10, 1,1,1,"m_cloud_orig", rm_v11);
        rm_vis_->addText("model cloud color registered", 10, 10, 10, 1,1,1,"m_cloud_reg", rm_v12);
        rm_vis_->addPointCloud(s_cloud_orig.makeShared(), "s_cloud_original", rm_v9);
        rm_vis_->addPointCloud(m_cloud_color_reg2.makeShared(), "m_cloud_color_registered2", rm_v10);
        rm_vis_->addPointCloud(m_cloud_orig.makeShared(), "m_cloud_original", rm_v11);
        rm_vis_->addPointCloud(m_cloud_color_reg.makeShared(), "m_cloud_color_registered", rm_v12);
    }
#endif
    typename pcl::PointCloud<ModelT>::Ptr visible_cloud_colored (new pcl::PointCloud<ModelT> (*rm.complete_cloud_));
    for(size_t i=0; i<visible_cloud_colored->points.size(); i++)
    {
        ModelT &mp = visible_cloud_colored->points[i];
        mp.r = mp.g = mp.b = 0.f;
    }

    for(size_t i=0; i<rm.visible_indices_.size(); i++)
    {
        int idx = rm.visible_indices_[i];
        ModelT &mp = visible_cloud_colored->points[idx];
        const ModelT &mp2 = rm.visible_cloud_->points[i];
        mp.r = 255.f;
        mp.g = 0.f;
        mp.b = 0.f;
    }

    std::stringstream txt; txt << "visible ratio: " << std::fixed << std::setprecision(2) << (float)rm.visible_cloud_->points.size() / (float)rm.complete_cloud_->points.size();

    if(!param_.vis_for_paper_)
        rm_vis_->addText(txt.str(),10,10,12,0,0,0,"visible model cloud",rm_v2);

    rm_vis_->addPointCloud(visible_cloud_colored, "model2",rm_v2);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             5, "model2", rm_v2);

    typename pcl::PointCloud<ModelT>::Ptr model_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
    for(size_t p=0; p < model_fit_cloud->points.size(); p++)
    {
        ModelT &mp = model_fit_cloud->points[p];
        mp.r = mp.b = 0.f;
        mp.g = 50.f;
    }
    for(size_t cidx=0; cidx < rm.model_scene_c_.size(); cidx++)
    {
        const pcl::Correspondence &c = rm.model_scene_c_[cidx];
        int sidx = c.index_match;
        int midx = c.index_query;
        float weight = c.weight;

        if(sidx<0)
            continue;

        ModelT &mp = model_fit_cloud->points[midx];
        mp.g = (255.f - mp.g) * weight;   // scale green channel with fitness score
    }
    txt.str(""); txt << "model cost: " << std::fixed << std::setprecision(4) << rm.model_fit_ <<
                        "; normalized: " << rm.model_fit_ / rm.visible_cloud_->points.size();
    rm_vis_->addText(txt.str(),10,10,12,0,0,0,"model cost",rm_v3);
    rm_vis_->addPointCloud(model_fit_cloud, "model cost", rm_v3);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             5, "model cost", rm_v3);


    // ---- VISUALIZE SMOOTH SEGMENTATION -------
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_smooth_labels_rgb (new pcl::PointCloud<pcl::PointXYZRGB>(*scene_cloud_downsampled_));
    if(!smooth_label_count_.empty())
    {
        Eigen::Matrix3Xf label_colors (3, smooth_label_count_.size());
        for(size_t i=0; i<smooth_label_count_.size(); i++)
        {
            float r,g,b;
            if( i==0 )
            {
                r = g = b = 255; // label 0 will be white
            }
            else
            {
                r = rand () % 255;
                g = rand () % 255;
                b = rand () % 255;
            }
            label_colors(0,i) = r;
            label_colors(1,i) = g;
            label_colors(2,i) = b;

            if(!param_.vis_for_paper_)
            {
                std::stringstream lbl_txt; lbl_txt << std::fixed << std::setprecision(2) << rm.explained_pts_per_smooth_cluster_[i] << " / " << smooth_label_count_[i];
                std::stringstream txt_id; txt_id << "smooth_cluster_txt " << i;
                rm_vis_->addText( lbl_txt.str(), 10, 10+12*i, 12, r/255, g/255, b/255, txt_id.str(), rm_v4);
            }
        }

        for(size_t i=0; i < scene_smooth_labels_.size(); i++)
        {
            int l = scene_smooth_labels_[i];
            pcl::PointXYZRGB &p = scene_smooth_labels_rgb->points[i];
            p.r = label_colors(0,l);
            p.g = label_colors(1,l);
            p.b = label_colors(2,l);
        }
        rm_vis_->addPointCloud(scene_smooth_labels_rgb, "smooth labels", rm_v4);
    }
    //---- END VISUALIZE SMOOTH SEGMENTATION-----------


    typename pcl::PointCloud<SceneT>::Ptr scene_fit_cloud (new pcl::PointCloud<SceneT> (*scene_cloud_downsampled_));

    for(size_t p=0; p<scene_explained_weight_.rows(); p++)
    {
        SceneT &sp = scene_fit_cloud->points[p];
        sp.r = sp.b = 0.f;

        sp.g = 255.f * scene_explained_weight_(p,model_id);
    }
    txt.str(""); txt << "scene pts explained (fitness: " << scene_explained_weight_.col(model_id).sum() <<
                        "; normalized: " << scene_explained_weight_.col(model_id).sum()/scene_cloud_downsampled_->points.size() << ")";
    rm_vis_->addText(txt.str(),10,10,12,0,0,0,"scene fitness",rm_v5);
    rm_vis_->addPointCloud(scene_fit_cloud, "scene fitness", rm_v5);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             5, "scene fitness", rm_v5);

    rm_vis_->addText("scene and visible model",10,10,12,1,1,1,"scene_and_model",rm_v6);
    rm_vis_->addPointCloud(scene_cloud_downsampled_, "scene_model_1", rm_v6);
    rm_vis_->addPointCloud(rm.visible_cloud_, "scene_model_2", rm_v6);

    rm_vis_->resetCamera();
    rm_vis_->spin();
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::visualizePairwiseIntersection() const
{
    if(!vis_pairwise_)
    {
        vis_pairwise_.reset( new pcl::visualization::PCLVisualizer("intersection") );
        vis_pairwise_->createViewPort(0,0,0.5,1, vp_pair_1_);
        vis_pairwise_->createViewPort(0.5,0,1,1, vp_pair_2_);
    }

    for(size_t i=1; i<recognition_models_.size(); i++)
    {
        const HVRecognitionModel<ModelT> &rm_a = *recognition_models_[i];

        for(size_t j=0; j<i; j++)
        {
            const HVRecognitionModel<ModelT> &rm_b = *recognition_models_[j];
            std::stringstream txt;
            txt <<  "intersection cost (" << i << ", " << j << "): " << intersection_cost_(j,i);

            vis_pairwise_->removeAllPointClouds();
            vis_pairwise_->removeAllShapes();
            vis_pairwise_->addText(txt.str(), 10, 10, 12, 1.f, 1.f, 1.f, "intersection_text", vp_pair_1_ );
            vis_pairwise_->addPointCloud(rm_a.visible_cloud_, "cloud_a", vp_pair_1_);
            vis_pairwise_->addPointCloud(rm_b.visible_cloud_, "cloud_b", vp_pair_1_);
//            vis.addPointCloud(rendered_vis_m_a.makeShared(), "cloud_ar",v2);
//            vis.addPointCloud(rendered_vis_m_b.makeShared(), "cloud_br",v2);
            vis_pairwise_->resetCamera();
            vis_pairwise_->spin();
        }
    }
}


template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::visualizeGOCues (const std::vector<bool> & active_solution, float cost, int times_evaluated) const
{
    (void)active_solution;
    (void)cost;
    (void)times_evaluated;
    std::cerr << "Visualizing GO Cues is only implemented for XYZRGB point clouds." << std::endl;
}


template<>
void
GHV<pcl::PointXYZRGB, pcl::PointXYZRGB>::visualizeGOCues (const std::vector<bool> & active_solution, float cost, int times_evaluated) const
{
    typedef pcl::PointXYZRGB ModelT;
    typedef pcl::PointXYZRGB SceneT;

    if(!vis_go_cues_) {
        vis_go_cues_.reset(new pcl::visualization::PCLVisualizer("visualizeGOCues"));
        vis_go_cues_->createViewPort(0, 0, 0.33, 0.5, vp_scene_);
        vis_go_cues_->createViewPort(0.33, 0, 0.66, 0.5, vp_active_hypotheses_);
        vis_go_cues_->createViewPort(0.66, 0, 1, 0.5, vp_model_fitness_);
        vis_go_cues_->createViewPort(0, 0.5, 0.33, 1, vp_scene_fitness_);
    }

    vis_go_cues_->removeAllPointClouds();
    vis_go_cues_->removeAllShapes();

    double model_fitness = 0.f;
    double pairwise_cost = 0.f;
    double scene_fitness = 0.f;

    // model uni term
    size_t num_active_hypotheses = 0;
    for(size_t i=0; i<active_solution.size(); i++)
    {
        if(active_solution[i]) {
            model_fitness += recognition_models_[i]->model_fit_;
            num_active_hypotheses++;
        }
    }
    if(!num_active_hypotheses)
        model_fitness = 0.f;

    // scene uni term
    Eigen::MatrixXf scene_explained_weight_for_active_hypotheses = scene_explained_weight_;
    for(size_t i=0; i<active_solution.size(); i++)
    {
        if(!active_solution[i]) {
            scene_explained_weight_for_active_hypotheses.col(i) = Eigen::VectorXf::Zero(
                        scene_explained_weight_for_active_hypotheses.rows());
        }
    }

    if ( scene_explained_weight_for_active_hypotheses.cols() ) {
        Eigen::VectorXf max = scene_explained_weight_for_active_hypotheses.rowwise().maxCoeff();
//        scene_fitness = max.sum() / scene_cloud_downsampled_->points.size();
        scene_fitness = max.sum();
    }


    // pairwise term
    for(size_t i=0; i<active_solution.size(); i++)
    {
        for(size_t j=0; j<i; j++)
        {
            if(active_solution[i] && active_solution[j])
                pairwise_cost += intersection_cost_(i,j);
        }
    }


    std::ostringstream out, model_fitness_txt, scene_fitness_txt;
    out << "Active Hypotheses" << std::endl;
    out << "Cost: " << std::setprecision(5) << cost << " , #Evaluations: " << times_evaluated;
    out << std::endl << "; pairwise cost: " << pairwise_cost << "; total cost: " << cost_ << std::endl;
    model_fitness_txt << "model fitness: " << model_fitness;
    scene_fitness_txt << "scene fitness: " << scene_fitness;


    vis_go_cues_->addText ("Scene", 1, 30, 16, 1, 1, 1, "inliers_outliers", vp_scene_);
    vis_go_cues_->addText (out.str(), 1, 30, 16, 1, 1, 1, "scene_cues", vp_active_hypotheses_);
    vis_go_cues_->addText (model_fitness_txt.str(), 1, 30, 16, 1, 1, 1, "model fitness", vp_model_fitness_);
    vis_go_cues_->addText (scene_fitness_txt.str(), 1, 30, 16, 1, 1, 1, "scene fitness", vp_scene_fitness_);
    vis_go_cues_->addPointCloud (scene_cloud_downsampled_, "scene_cloud", vp_scene_);

    //display active hypotheses
    for(size_t i=0; i < active_solution.size(); i++)
    {
        if(active_solution[i])
        {
            HVRecognitionModel<ModelT> &rm = *recognition_models_[i];
            std::stringstream model_name; model_name << "model_" << i;
            vis_go_cues_->addPointCloud(rm.visible_cloud_, model_name.str(), vp_active_hypotheses_);

            typename pcl::PointCloud<ModelT>::Ptr model_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
            for(size_t p=0; p < model_fit_cloud->points.size(); p++)
            {
                ModelT &mp = model_fit_cloud->points[p];
                mp.r = mp.g = 0.f;

                const pcl::Correspondence &c = rm.model_scene_c_[p];
                mp.b = 50.f + 205.f * c.weight;
            }

            model_name << "_fitness";
            vis_go_cues_->addPointCloud(model_fit_cloud, model_name.str(), vp_model_fitness_);
        }
    }

    if ( scene_explained_weight_for_active_hypotheses.cols() ) {
        Eigen::VectorXf max_fit = scene_explained_weight_for_active_hypotheses.rowwise().maxCoeff();
        typename pcl::PointCloud<SceneT>::Ptr scene_fit_cloud (new pcl::PointCloud<SceneT> (*scene_cloud_downsampled_));

        for(size_t p=0; p<scene_fit_cloud->points.size(); p++)
        {
            SceneT &sp = scene_fit_cloud->points[p];
            sp.r = sp.g = 0.f;

            sp.b = 50.f + 205.f * max_fit(p);
        }

        vis_go_cues_->addPointCloud(scene_fit_cloud, "scene fitness", vp_scene_fitness_);
    }

    vis_go_cues_->resetCamera();
    vis_go_cues_->spin();
}

}
