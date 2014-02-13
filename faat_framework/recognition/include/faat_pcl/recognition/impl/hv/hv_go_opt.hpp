/*
 * hv_go_bin_opt.hpp
 *
 *  Created on: Feb 27, 2013
 *      Author: aitor
 */

#include <faat_pcl/recognition/hv/hv_go_opt.h>

///////////////////////////////////////////////////////////////
///////////// HVGOBinaryOptimizer /////////////////////////////
///////////////////////////////////////////////////////////////

template<typename ModelT, typename SceneT>
float
faat_pcl::HVGOBinaryOptimizer<ModelT, SceneT>::computeBound(SAModel<ModelT, SceneT> & model, int d) {

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
faat_pcl::HVGOBinaryOptimizer<ModelT, SceneT>::search_recursive (mets::feasible_solution & sol, int d)
throw(mets::no_moves_error)
{
  if(d == (sol_length_))
    return;

  //do we need to expand this branch?
  //compute lower bound and compare with incumbent
  SAModel<ModelT, SceneT> model;
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
  typedef mets::abstract_search<move_manager<ModelT, SceneT> > base_t;
  move<ModelT, SceneT> m(d);
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
faat_pcl::HVGOBinaryOptimizer<ModelT, SceneT>::computeStructures
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
faat_pcl::HVGOBinaryOptimizer<ModelT, SceneT>::search ()
throw(mets::no_moves_error)
{
  typedef mets::abstract_search<faat_pcl::move_manager<ModelT, SceneT> > base_t;
  base_t::solution_recorder_m.accept(base_t::working_solution_m);

  best_cost_ =
    static_cast<mets::evaluable_solution&>(base_t::working_solution_m)
    .cost_function();

  std::cout << "Initial cost HVGOBinaryOptimizer:" << static_cast<float>(best_cost_) << std::endl;

  search_recursive(base_t::working_solution_m, 0);
}

///////////////////////////////////////////////////////////////
///////////// move manager ////////////////////////////////////
///////////////////////////////////////////////////////////////

template<typename ModelT, typename SceneT>
void
faat_pcl::move_manager<ModelT, SceneT>::refresh(mets::feasible_solution& s)
{
  for (iterator ii = begin (); ii != end (); ++ii)
    delete (*ii);

  SAModel<ModelT, SceneT>& model = dynamic_cast<SAModel<ModelT, SceneT>&> (s);
  moves_m.clear();
  moves_m.resize(model.solution_.size() + model.solution_.size()*model.solution_.size());
  for (int ii = 0; ii != model.solution_.size(); ++ii)
    moves_m[ii]  = new move<ModelT, SceneT> (ii);

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
          moves_m[model.solution_.size() + nm] = new replace_hyp_move<ModelT, SceneT> (active[i], inactive[j], model.solution_.size());
          nm++;
        }
      }
    }

    moves_m.resize(model.solution_.size() + nm);
  } else {
    moves_m.resize(model.solution_.size());
  }
  std::random_shuffle (moves_m.begin (), moves_m.end ());
}
