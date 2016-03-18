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
        for(size_t j=0; j<i; j++)
        {
            const HVRecognitionModel<ModelT> &rm_a = *recognition_models_[i];
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

    {
        pcl::ScopeTime t("pose refinement and computing visible model points");
        computeVisibleModelsAndRefinePose();
    }

    removeModelsWithLowVisibility();

    #pragma omp parallel sections
    {
        #pragma omp section
        if(!param_.ignore_color_even_if_exists_)
        {
            pcl::ScopeTime t("Converting scene color values");
            convertColor(*scene_cloud_downsampled_, scene_color_channels_);
        }

        #pragma omp section
        {
            pcl::ScopeTime t("Converting model color values");
            for (size_t i = 0; i < recognition_models_.size (); i++)
            {
                HVRecognitionModel<ModelT> &rm = *recognition_models_[i];
                removeNanNormals(rm);

                if(!param_.ignore_color_even_if_exists_)
                    convertColor(*rm.visible_cloud_, rm.pt_color_);
            }
        }
    }

    {
        pcl::ScopeTime t("Computing fitness score between models and scene");
        scene_explained_weight_ = Eigen::MatrixXf::Zero(scene_cloud_downsampled_->points.size(), recognition_models_.size());

        for (size_t i = 0; i < recognition_models_.size (); i++){
            HVRecognitionModel<ModelT> &rm = *recognition_models_[i];
            computeModel2SceneFitness(rm, i);
        }
    }

    // remove badly explained object hypotheses
    size_t kept=0;
    for (size_t i = 0; i < recognition_models_.size (); i++)
    {
        HVRecognitionModel<ModelT> &rm = *recognition_models_[i];
        if(rm.model_fit_ / rm.visible_cloud_->points.size() > param_.min_model_fitness_)
        {
            recognition_models_[kept] = recognition_models_[i]; //TODO: USE POINTER TO AVOID COPYING
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

    if(!kept)
        return;

    {
        pcl::ScopeTime t("Compressing scene explained matrix");
        // remove rows of scene explained matrix, whose point is not explained by any hypothesis. Because it is usually very sparse and would take a lot of computation time.
        scene_explained_weight_compressed_ = scene_explained_weight_;
        Eigen::VectorXf min_tmp = scene_explained_weight_.rowwise().maxCoeff();
        kept=0;
        for(size_t pt=0; pt<scene_cloud_downsampled_->points.size(); pt++)
        {
            if( min_tmp(pt) > std::numeric_limits<float>::epsilon() )
            {
                scene_explained_weight_compressed_.row(kept).swap(scene_explained_weight_compressed_.row(pt));
                kept++;
            }
        }
        scene_explained_weight_compressed_.conservativeResize(kept, scene_explained_weight_.cols());

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
template<typename PointT>
void
GHV<ModelT, SceneT>::convertColor(const typename pcl::PointCloud<PointT> &cloud, Eigen::MatrixXf &color_mat)
{
    throw std::runtime_error("This function is not implemented for the chosen Point Cloud Type!");
}

template<typename ModelT, typename SceneT>
template<pcl::PointXYZRGB>
void
GHV<ModelT, SceneT>::convertColor(const typename pcl::PointCloud<pcl::PointXYZRGB> &cloud, Eigen::MatrixXf &color_mat)
{
    size_t num_color_channels = 0;
    switch (param_.color_space_)
    {
        case ColorSpace::LAB: case ColorSpace::RGB: num_color_channels = 3; break;
        case ColorSpace::GRAYSCALE: num_color_channels = 1; break;
        default: throw std::runtime_error("Color space not implemented!");
    }

    color_mat = Eigen::MatrixXf::Zero ( cloud.points.size(), num_color_channels);

    #pragma omp parallel for schedule (dynamic)
    for(size_t j=0; j < cloud.points.size(); j++)
    {
        const PointT &p = cloud.points[j];

        switch (param_.color_space_)
        {
            case ColorSpace::LAB:
            {
                unsigned char r = (unsigned char)p.r;
                unsigned char g = (unsigned char)p.g;
                unsigned char b = (unsigned char)p.b;
                float LRefm, aRefm, bRefm;
                color_transf_omp_.RGB2CIELAB(r, g, b, LRefm, aRefm, bRefm);
                color_mat(j, 0) = LRefm;
                color_mat(j, 1) = aRefm;
                color_mat(j, 2) = bRefm;
                break;
            }
            case ColorSpace::RGB:
                color_mat(j, 0) = p.r/255.f;
                color_mat(j, 1) = p.g/255.f;
                color_mat(j, 2) = p.b/255.f;
                break;
            case ColorSpace::GRAYSCALE:
                color_mat(j, 0) = .2126 * p.r/255.f + .7152 * p.g/255.f + .0722 * p.b/255.f;
        }
    }
}

template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::computeModel2SceneFitness(HVRecognitionModel<ModelT> &rm, size_t model_idx)
{
    rm.model_fit_ = 0.;
    rm.model_scene_c_.resize( rm.visible_cloud_->points.size() );

    //Goes through the visible model points and finds scene points within a radius neighborhood
    //If in this neighborhood, there are no scene points, model point is considered outlier
    //If there are scene points, the model point is associated with the scene point, together with its distance
    //A scene point might end up being explained by multiple model points

    omp_lock_t scene_pt_lock[ scene_cloud_downsampled_->points.size() ];
    for(size_t i=0; i < scene_cloud_downsampled_->points.size(); i++)
        omp_init_lock( &scene_pt_lock[i] );

//    std::cout << "Color sigma ab " << param_.color_sigma_ab_ << ", inlier thresh: " << param_.inliers_threshold_ << std::endl;

    #pragma omp parallel for schedule(dynamic)
    for (size_t m_pt_id = 0; m_pt_id < rm.visible_cloud_->points.size (); m_pt_id++) {
//        float radius = param_.inliers_threshold_;

//        if ( param_.use_noise_model_ ) {
//            float min_radius = 0.01f, max_radius = 0.03f;
//            radius = 2*std::max(noise_term_visible_pt[pt][0], noise_term_visible_pt[pt][1]);
//            radius = std::max( std::min(radius, min_radius), max_radius);   // clip threshold to account for pose error etc.
//        }

        std::vector<int> nn_indices;
        std::vector<float> nn_distances;
        double w3d = 1 / (param_.inliers_threshold_ * param_.inliers_threshold_);
        double w_color = 1 / (2 * param_.color_sigma_ab_ * param_.color_sigma_ab_);

        octree_scene_downsampled_->nearestKSearch(rm.visible_cloud_->points[m_pt_id], param_.knn_inliers_,
                                                 nn_indices, nn_distances);

        if ( !nn_indices.empty() && !param_.ignore_color_even_if_exists_ )
        {
            int min_idx = 0;
            double max_score = 0.f;

            for (size_t k = 0; k < nn_indices.size(); k++)
            {
                  int sidx = nn_indices[ k ];
                  const Eigen::VectorXf &color_m = rm.pt_color_.row(m_pt_id);
                  const Eigen::VectorXf &color_s = scene_color_channels_.row( nn_indices[k] );
                  double sqr_3D_dist = nn_distances[k];
                  double score = w3d * sqr_3D_dist;

                  if(param_.color_space_ == ColorSpace::LAB)
                  {
                      double As = color_s(1);
                      double Bs = color_s(2);
                      double Am = color_m(1);
                      double Bm = color_m(2);

                      double sqr_color_dist = ( (As-Am)*(As-Am)+(Bs-Bm)*(Bs-Bm) );
                      score += w_color * sqr_color_dist;
                  }


                  score = exp( -score ); //1.f - exp( -dist );
                  if(score > max_score) {
                      max_score = score;
                      min_idx = k;
                  }

                  omp_set_lock(&scene_pt_lock[sidx]);
                  double old_scene_expl_weight = scene_explained_weight_(sidx, model_idx);
                  scene_explained_weight_(sidx, model_idx) = std::max(old_scene_expl_weight, score );    // 1 for perfect fit, 0 if if not
                  omp_unset_lock(&scene_pt_lock[sidx]);
            }

            pcl::Correspondence c;
            int sidx = nn_indices[ min_idx ];
//            double weight = max_score; //1.f - exp( -min_dist );
            c.index_query = m_pt_id;
            c.index_match = sidx;
            c.distance = max_score;
            rm.model_scene_c_[ m_pt_id] = c;
            rm.model_fit_ += max_score;
        }
    }

    for(size_t i=0; i < scene_cloud_downsampled_->points.size(); i++)
        omp_destroy_lock( &scene_pt_lock[i] );
}

//######### VISUALIZATION FUNCTIONS #####################
template<typename ModelT, typename SceneT>
void
GHV<ModelT, SceneT>::visualizeGOCuesForModel(const HVRecognitionModel<ModelT> &rm, int model_id) const
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

    rm_vis_->addText("scene",10,10,20,1,1,1,"scene",rm_v1);
    rm_vis_->addPointCloud(scene_cloud_downsampled_, "scene1",rm_v1);

    typename pcl::PointCloud<ModelT>::Ptr visible_cloud_colored (new pcl::PointCloud<ModelT> (*rm.complete_cloud_));
    for(size_t i=0; i<visible_cloud_colored->points.size(); i++)
    {
        ModelT &mp = visible_cloud_colored->points[i];
        mp.r = mp.g = mp.b = 128.f;
    }

    for(size_t i=0; i<rm.visible_indices_.size(); i++)
    {
        int idx = rm.visible_indices_[i];
        ModelT &mp = visible_cloud_colored->points[idx];
        const ModelT &mp2 = rm.visible_cloud_->points[i];
        mp.r = mp2.r;
        mp.g = mp2.g;
        mp.b = mp2.b;
    }

    std::stringstream txt; txt << "visible ratio: " << (float)rm.visible_cloud_->points.size() / (float)rm.complete_cloud_->points.size();
    rm_vis_->addText(txt.str(),10,10,20,1,1,1,"visible model cloud",rm_v2);
    rm_vis_->addPointCloud(visible_cloud_colored, "model2",rm_v2);

    typename pcl::PointCloud<ModelT>::Ptr model_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
    for(size_t p=0; p < model_fit_cloud->points.size(); p++)
    {
        ModelT &mp = model_fit_cloud->points[p];
        mp.r = mp.b = 0.f;
        const pcl::Correspondence &c = rm.model_scene_c_[p];
        mp.g = 50.f + 205.f * c.weight;
    }
    txt.str(""); txt << "model cost: " << rm.model_fit_ <<
                        "; normalized: " << rm.model_fit_ / rm.visible_cloud_->points.size();
    rm_vis_->addText(txt.str(),10,10,20,1,1,1,"model cost",rm_v3);
    rm_vis_->addPointCloud(model_fit_cloud, "model cost", rm_v3);


    typename pcl::PointCloud<SceneT>::Ptr scene_fit_cloud (new pcl::PointCloud<SceneT> (*scene_cloud_downsampled_));

    for(size_t p=0; p<scene_explained_weight_.rows(); p++)
    {
        SceneT &sp = scene_fit_cloud->points[p];
        sp.r = sp.b = 0.f;

        sp.g = 50.f + 205.f * scene_explained_weight_(p,model_id);
    }
    txt.str(""); txt << "scene pts explained (fitness: " << scene_explained_weight_.col(model_id).sum() <<
                        "; normalized: " << scene_explained_weight_.col(model_id).sum()/scene_cloud_downsampled_->points.size() << ")";    rm_vis_->addText(txt.str(),10,10,20,1,1,1,"scene fitness",rm_v4);
    rm_vis_->addPointCloud(scene_fit_cloud, "scene fitness", rm_v4);

    rm_vis_->addText("scene and visible model",10,10,20,1,1,1,"scene_and_model",rm_v5);
    rm_vis_->addPointCloud(scene_cloud_downsampled_, "scene_model_1", rm_v5);
    rm_vis_->addPointCloud(rm.visible_cloud_, "scene_model_2", rm_v5);

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
            vis_pairwise_->addText(txt.str(), 10, 10, 20, 1.f, 1.f, 1.f, "intersection_text", vp_pair_1_ );
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
