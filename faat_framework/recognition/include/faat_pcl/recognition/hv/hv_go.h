#ifndef FAAT_PCL_GO_H_
#define FAAT_PCL_GO_H_

#include <pcl/pcl_macros.h>
#include <faat_pcl/recognition/hv/hypotheses_verification.h>
#include <pcl/common/common.h>
#include "pcl/recognition/3rdparty/metslib/mets.hh"
#include <pcl/features/normal_3d.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <map>
#include <iostream>
#include <fstream>

#ifdef _MSC_VER
#ifdef FAAT_REC_EXPORTS
#define FAAT_REC_API __declspec(dllexport)
#else
#define FAAT_REC_API __declspec(dllimport)
#endif
#else
#define FAAT_REC_API
#endif

namespace faat_pcl
{

  /** \brief A hypothesis verification method proposed in
    * "A Global Hypotheses Verification Method for 3D Object Recognition", A. Aldoma and F. Tombari and L. Di Stefano and Markus Vincze, ECCV 2012
    * \author Aitor Aldoma
    * Extended with physical constraints and color information (see ICRA paper)
    */

  template<typename ModelT, typename SceneT>
  class FAAT_REC_API GlobalHypothesesVerification: public faat_pcl::HypothesisVerification<ModelT, SceneT>
  {
    private:

      //Helper classes
      struct RecognitionModel
      {
        public:
          std::vector<int> explained_; //indices vector referencing explained_by_RM_
          std::vector<float> explained_distances_; //closest distances to the scene for point i
          std::vector<int> unexplained_in_neighborhood; //indices vector referencing unexplained_by_RM_neighboorhods
          std::vector<float> unexplained_in_neighborhood_weights; //weights for the points not being explained in the neighborhood of a hypothesis
          std::vector<int> outlier_indices_; //outlier indices of this model
          std::vector<int> complete_cloud_occupancy_indices_;
          typename pcl::PointCloud<ModelT>::Ptr cloud_;
          typename pcl::PointCloud<ModelT>::Ptr complete_cloud_;
          typename pcl::PointCloud<pcl::Normal>::Ptr complete_cloud_normals_;
          float bad_information_;
          float outliers_weight_;
          pcl::PointCloud<pcl::Normal>::Ptr normals_;
          int id_;
          float extra_weight_; //descriptor distance weight for instance
          float model_constraints_value_;
          float color_similarity_;
          float median_;
          float mean_;
          Eigen::MatrixXf color_mapping_;
          std::string id_s_;
      };

      typedef GlobalHypothesesVerification<ModelT, SceneT> SAOptimizerT;
      class SAModel: public mets::evaluable_solution
      {
        public:
          std::vector<bool> solution_;
          SAOptimizerT * opt_;
          mets::gol_type cost_;

          //Evaluates the current solution
          mets::gol_type cost_function() const
          {
            return cost_;
          }

          void copy_from(const mets::copyable& o)
          {
            const SAModel& s = dynamic_cast<const SAModel&> (o);
            solution_ = s.solution_;
            opt_ = s.opt_;
            cost_ = s.cost_;
          }

          void copy_from(const mets::feasible_solution& o)
          {
            const SAModel& s = dynamic_cast<const SAModel&> (o);
            solution_ = s.solution_;
            opt_ = s.opt_;
            cost_ = s.cost_;
          }

          mets::gol_type what_if(int /*index*/, bool /*val*/) const
          {
            return static_cast<mets::gol_type>(0);
          }

          mets::gol_type apply_and_evaluate(int index, bool val)
          {
            solution_[index] = val;
            mets::gol_type sol = opt_->evaluateSolution (solution_, index); //this will update the state of the solution
            cost_ = sol;
            return sol;
          }

          void apply(int /*index*/, bool /*val*/)
          {

          }

          void unapply(int index, bool val)
          {
            solution_[index] = val;
            //update optimizer solution
            cost_ = opt_->evaluateSolution (solution_, index); //this will udpate the cost function in opt_
          }
          void setSolution(std::vector<bool> & sol)
          {
            solution_ = sol;
          }

          void setOptimizer(SAOptimizerT * opt)
          {
            opt_ = opt;
          }
      };

      /*
       * Represents a generic move from which all move types should inherit
       */

      class generic_move : public mets::mana_move
      {
        public:
          virtual mets::gol_type evaluate(const mets::feasible_solution& cs) const = 0;
          virtual mets::gol_type apply_and_evaluate(mets::feasible_solution& cs) = 0;
          virtual void apply(mets::feasible_solution& s) const = 0;
          virtual void unapply(mets::feasible_solution& s) const = 0;
          virtual mets::clonable* clone() const = 0;
          virtual size_t hash() const = 0;
          virtual bool operator==(const mets::mana_move&) const = 0;
      };

      /*
       * Represents a move that deactivates an active hypothesis replacing it by an inactive one
       * Such moves should be done when the temperature is low
       * It is based on the intersection of explained points between hypothesis
       */

      class replace_hyp_move: public generic_move {
        int i_, j_; //i_ is an active hypothesis, j_ is an inactive hypothesis
        int sol_size_;

        public:
          replace_hyp_move(int i, int j, int sol_size) : i_(i), j_(j), sol_size_(sol_size)
          {
          }

          mets::gol_type evaluate(const mets::feasible_solution& cs) const
          {
            SAModel model;
            model.copy_from(cs);
            model.apply_and_evaluate (i_, !model.solution_[i_]);
            mets::gol_type cost = model.apply_and_evaluate (j_, !model.solution_[j_]);
            //unapply moves now
            model.unapply (j_, !model.solution_[j_]);
            model.unapply (i_, !model.solution_[i_]);
            return cost;
          }

          mets::gol_type apply_and_evaluate(mets::feasible_solution& cs)
          {
            SAModel& model = dynamic_cast<SAModel&> (cs);
            assert(model.solution_[i_]);
            model.apply_and_evaluate (i_, !model.solution_[i_]);
            assert(!model.solution_[j_]);
            return model.apply_and_evaluate (j_, !model.solution_[j_]);
          }

          void apply(mets::feasible_solution& s) const
          {
            SAModel& model = dynamic_cast<SAModel&> (s);
            model.apply_and_evaluate (i_, !model.solution_[i_]);
            model.apply_and_evaluate (j_, !model.solution_[j_]);
          }

          void unapply(mets::feasible_solution& s) const
          {
            //go back
            SAModel& model = dynamic_cast<SAModel&> (s);
            model.unapply (j_, !model.solution_[j_]);
            model.unapply (i_, !model.solution_[i_]);
          }

          mets::clonable* clone() const {
            replace_hyp_move * m = new replace_hyp_move(i_, j_, sol_size_);
            return static_cast<mets::clonable*>(m);
          }

          size_t hash() const {
            return static_cast<size_t>(sol_size_ + sol_size_ * i_ + j_);
          }

          bool operator==(const mets::mana_move& m) const {
            const replace_hyp_move& mm = dynamic_cast<const replace_hyp_move&> (m);
            return (mm.i_ == i_) && (mm.j_ == j_);
          }
      };

      /*
       * Represents a move, activate/deactivate an hypothesis
       */

      class move: public generic_move
      {
          int index_;
        public:
          move(int i) :
              index_ (i)
          {
          }

          int getIndex() {
            return index_;
          }

          mets::gol_type evaluate(const mets::feasible_solution& cs) const
          {
            //mets::copyable copyable = dynamic_cast<mets::copyable> (&cs);
            SAModel model;
            model.copy_from(cs);
            mets::gol_type cost = model.apply_and_evaluate (index_, !model.solution_[index_]);
            model.apply_and_evaluate (index_, !model.solution_[index_]);
            return cost;
          }

          mets::gol_type apply_and_evaluate(mets::feasible_solution& cs)
          {
            SAModel& model = dynamic_cast<SAModel&> (cs);
            return model.apply_and_evaluate (index_, !model.solution_[index_]);
          }

          void apply(mets::feasible_solution& s) const
          {
            SAModel& model = dynamic_cast<SAModel&> (s);
            model.apply_and_evaluate (index_, !model.solution_[index_]);
          }

          void unapply(mets::feasible_solution& s) const
          {
            SAModel& model = dynamic_cast<SAModel&> (s);
            model.unapply (index_, !model.solution_[index_]);
          }

          mets::clonable* clone() const {
            move * m = new move(index_);
            return static_cast<mets::clonable*>(m);
          }

          size_t hash() const {
            return static_cast<size_t>(index_);
          }

          bool operator==(const mets::mana_move& m) const {
            const move& mm = dynamic_cast<const move&> (m);
            return mm.index_ == index_;
          }
      };

      class move_manager
      {
          bool use_replace_moves_;
        public:
          std::vector<generic_move*> moves_m;
          boost::shared_ptr< std::map< std::pair<int, int>, bool > > intersections_;
          typedef typename std::vector<generic_move*>::iterator iterator;
          iterator begin()
          {
            return moves_m.begin ();
          }
          iterator end()
          {
            return moves_m.end ();
          }

          move_manager(int problem_size, bool rp_moves = true)
          {
            use_replace_moves_ = rp_moves;
            for (int ii = 0; ii != problem_size; ++ii)
              moves_m.push_back (new move (ii));
          }

          ~move_manager()
          {
            // delete all moves
            for (iterator ii = begin (); ii != end (); ++ii)
              delete (*ii);
          }

          void setExplainedPointIntersections(boost::shared_ptr<std::map<std::pair<int, int>, bool > > & intersections) {
            intersections_ = intersections;
          }

          void refresh(mets::feasible_solution& s);
      };

      class CostFunctionLogger : public mets::solution_recorder {
        std::vector<float> costs_;
        std::vector<float> costs_each_time_evaluated_;
        int times_evaluated_;

        public:
          CostFunctionLogger();

          CostFunctionLogger(mets::evaluable_solution& best) :
                mets::solution_recorder(),
                best_ever_m(best)
              {
                times_evaluated_ = 0;
                costs_.resize(0);
              }

          void writeToLog(std::ofstream & of) {
            const SAModel& ss = static_cast<const SAModel&> (best_ever_m);
            of << times_evaluated_ << "\t\t";
            of << costs_.size() << "\t\t";
            of << costs_[costs_.size() - 1] << std::endl;
          }

          void writeEachCostToLog(std::ofstream & of) {
            for(size_t i=0; i < costs_each_time_evaluated_.size(); i++) {
              of << costs_each_time_evaluated_[i] << "\t";
            }
            of << std::endl;
          }

          void addCost(float c)
          {
            costs_.push_back(c);
          }

          void addCostEachTimeEvaluated(float c)
          {
            costs_each_time_evaluated_.push_back(c);
          }

          void increaseEvaluated()
          {
            times_evaluated_++;
          }

          int getTimesEvaluated() {
            return times_evaluated_;
          }

          size_t getAcceptedMovesSize() {
            return costs_.size();
          }

          bool accept(const mets::feasible_solution& sol)
          {
            const mets::evaluable_solution& s = dynamic_cast<const mets::evaluable_solution&>(sol);
            if(s.cost_function() <= best_ever_m.cost_function())
            {
              best_ever_m.copy_from(s);
              const SAModel& ss = static_cast<const SAModel&> (sol);
              costs_.push_back(ss.cost_);
              std::cout << "Move accepted:" << ss.cost_ << std::endl;
              return true;
            }
            return false;
          }

          /// @brief Returns the best solution found since the beginning.
          const mets::evaluable_solution& best_seen() const
          {
            return best_ever_m;
          }

          mets::gol_type best_cost() const
          {
            return best_ever_m.cost_function();
          }

          void setBestSolution(std::vector<bool> & sol)
          {
            SAModel& ss = static_cast<SAModel&> (best_ever_m);
            for(size_t i=0; i < sol.size(); i++) {
              ss.solution_[i] = sol[i];
              //std::cout << "setBestSolution" << ss.solution_[i] << " " << sol[i] << std::endl;
            }
          }

        protected:
            /// @brief Records the best solution
            mets::evaluable_solution& best_ever_m;
      };

      class HVGOBinaryOptimizer : public mets::abstract_search<move_manager>
      {

      private:
        void
        search_recursive (mets::feasible_solution & sol, int d)
        throw(mets::no_moves_error);

        mets::gol_type best_cost_;
        int sol_length_;
        std::vector<boost::shared_ptr<RecognitionModel> > recognition_models__;
        std::vector<float> intersection_full_;
        std::vector<float> intersection_;

        float incumbent_;
        SAOptimizerT * opt_;

      public:
        HVGOBinaryOptimizer (mets::evaluable_solution& starting_point,
                               mets::solution_recorder& recorder,
                               move_manager& moveman,
                               int sol_length)
                            : mets::abstract_search<move_manager>(starting_point, recorder, moveman),
                              sol_length_(sol_length)
        {
         typedef mets::abstract_search<move_manager> base_t;
         base_t::step_m = 0;
        }

        void setRecogModels(std::vector<boost::shared_ptr<RecognitionModel> > & recog_models) {
          recognition_models__ = recog_models;
        }

        void computeStructures(int size_full_occupancy, int size_explained);

        float computeBound(SAModel & model, int d);

        void
        search ()
        throw(mets::no_moves_error);

        void
        setIncumbent(float inc)
        {
          incumbent_ = inc;
        }

        void setOptimizer(SAOptimizerT * opt)
        {
          opt_ = opt;
        }
      };

      //inherited class attributes
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::mask_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::scene_cloud_downsampled_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::scene_downsampled_tree_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::visible_models_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::visible_normal_models_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::visible_indices_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::complete_models_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::resolution_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::inliers_threshold_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::normals_set_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::requires_normals_;

      //class attributes
      typedef typename pcl::NormalEstimation<SceneT, pcl::Normal> NormalEstimator_;
      pcl::PointCloud<pcl::Normal>::Ptr scene_normals_;
      pcl::PointCloud<pcl::PointXYZI>::Ptr clusters_cloud_;

      std::vector<int> complete_cloud_occupancy_by_RM_;
      float res_occupancy_grid_;
      float w_occupied_multiple_cm_;

      std::vector<int> explained_by_RM_; //represents the points of scene_cloud_ that are explained by the recognition models
      std::vector<double> explained_by_RM_distance_weighted; //represents the points of scene_cloud_ that are explained by the recognition models
      std::vector<double> unexplained_by_RM_neighboorhods; //represents the points of scene_cloud_ that are not explained by the active hypotheses in the neighboorhod of the recognition models
      std::vector<boost::shared_ptr<RecognitionModel> > recognition_models_;
      std::vector<size_t> indices_;

      float regularizer_;
      float clutter_regularizer_;
      bool detect_clutter_;
      float radius_neighborhood_GO_;
      float radius_normals_;

      float previous_explained_value;
      int previous_duplicity_;
      int previous_duplicity_complete_models_;
      float previous_bad_info_;
      float previous_unexplained_;

      int max_iterations_; //max iterations without improvement
      SAModel best_seen_;
      float initial_temp_;
      bool use_replace_moves_;

      //conflict graph stuff
      bool use_conflict_graph_;
      std::vector<float> extra_weights_;
      int n_cc_;
      std::vector<std::vector<int> > cc_;

      std::map<int, int> graph_id_model_map_;
      typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::shared_ptr<RecognitionModel> > Graph;
      Graph conflict_graph_;
      std::vector<std::vector<boost::shared_ptr<RecognitionModel> > > points_explained_by_rm_; //if inner size > 1, conflict

      //general model constraints, they get a model cloud and return the number of points that do not fulfill the condition
      std::vector<boost::function<int
      (const typename pcl::PointCloud<ModelT>::Ptr)> > model_constraints_;
      std::vector<float> model_constraints_weights_;
      bool ignore_color_even_if_exists_;
      float color_sigma_;
      int opt_type_;
      float active_hyp_penalty_;

      std::vector<std::string> object_ids_;

      float getOccupiedMultipleW() {
        return w_occupied_multiple_cm_;
      }

      void setPreviousBadInfo(float f)
      {
        previous_bad_info_ = f;
      }

      float getPreviousBadInfo()
      {
        return previous_bad_info_;
      }

      void setPreviousExplainedValue(float v)
      {
        previous_explained_value = v;
      }

      void setPreviousDuplicity(int v)
      {
        previous_duplicity_ = v;
      }

      void setPreviousDuplicityCM(int v)
      {
        previous_duplicity_complete_models_ = v;
      }

      void setPreviousUnexplainedValue(float v)
      {
        previous_unexplained_ = v;
      }

      float getPreviousUnexplainedValue()
      {
        return previous_unexplained_;
      }

      float getExplainedValue()
      {
        return previous_explained_value;
      }

      int getDuplicity()
      {
        return previous_duplicity_;
      }

      int getDuplicityCM()
      {
        return previous_duplicity_complete_models_;
      }

      float getHypPenalty() {
        return active_hyp_penalty_;
      }

      double getExplainedByIndices(std::vector<int> & indices, std::vector<float> & explained_values,
                                     std::vector<double> & explained_by_RM, std::vector<int> & indices_to_update_in_RM_local);

      void getExplainedByRM(std::vector<double> & explained_by_rm) {
        explained_by_rm = explained_by_RM_distance_weighted;
      }

      void updateUnexplainedVector(std::vector<int> & unexplained_, std::vector<float> & unexplained_distances, std::vector<double> & unexplained_by_RM,
          std::vector<int> & explained, std::vector<int> & explained_by_RM, float val)
      {
        {

          float add_to_unexplained = 0.f;

          for (size_t i = 0; i < unexplained_.size (); i++)
          {

            bool prev_unexplained = (unexplained_by_RM[unexplained_[i]] > 0) && (explained_by_RM[unexplained_[i]] == 0);
            unexplained_by_RM[unexplained_[i]] += val * unexplained_distances[i];

            if (val < 0) //the hypothesis is being removed
            {
              if (prev_unexplained)
              {
                //decrease by 1
                add_to_unexplained -= unexplained_distances[i];
              }
            } else //the hypothesis is being added and unexplains unexplained_[i], so increase by 1 unless its explained by another hypothesis
            {
              if (explained_by_RM[unexplained_[i]] == 0)
                add_to_unexplained += unexplained_distances[i];
            }
          }

          for (size_t i = 0; i < explained.size (); i++)
          {
            if (val < 0)
            {
              //the hypothesis is being removed, check that there are no points that become unexplained and have clutter unexplained hypotheses
              if ((explained_by_RM[explained[i]] == 0) && (unexplained_by_RM[explained[i]] > 0))
              {
                add_to_unexplained += unexplained_by_RM[explained[i]]; //the points become unexplained
              }
            } else
            {
              //std::cout << "being added..." << add_to_unexplained << " " << unexplained_by_RM[explained[i]] << std::endl;
              if ((explained_by_RM[explained[i]] == 1) && (unexplained_by_RM[explained[i]] > 0))
              { //the only hypothesis explaining that point
                add_to_unexplained -= unexplained_by_RM[explained[i]]; //the points are not unexplained any longer because this hypothesis explains them
              }
            }
          }

          //std::cout << add_to_unexplained << std::endl;
          previous_unexplained_ += add_to_unexplained;
        }
      }

      void updateExplainedVector(std::vector<int> & vec, std::vector<float> & vec_float, std::vector<int> & explained_,
          std::vector<double> & explained_by_RM_distance_weighted, float sign)
      {
        float add_to_explained = 0.f;
        int add_to_duplicity_ = 0;

        for (size_t i = 0; i < vec.size (); i++)
        {
          bool prev_dup = explained_[vec[i]] > 1;
          bool prev_explained = explained_[vec[i]] == 1;
          float prev_explained_value = explained_by_RM_distance_weighted[vec[i]];

          explained_[vec[i]] += static_cast<int> (sign);
          explained_by_RM_distance_weighted[vec[i]] += vec_float[i] * sign;

          //add_to_explained += vec_float[i] * sign;
          if(explained_[vec[i]] == 1 && !prev_explained) {
            if(sign > 0) {
              add_to_explained += vec_float[i];
            } else {
              add_to_explained += explained_by_RM_distance_weighted[vec[i]];
            }
          }

          //hypotheses being removed, now the point is not explained anymore and was explained before by this hypothesis
          if((sign < 0) && (explained_[vec[i]] == 0) && prev_explained) {
            //assert(prev_explained_value == vec_float[i]);
            add_to_explained -= prev_explained_value;
          }

          //this hypothesis was added and now the point is not explained anymore, remove previous value
          if((sign > 0) && (explained_[vec[i]] == 2) && prev_explained)
            add_to_explained -= prev_explained_value;

          if ((explained_[vec[i]] > 1) && prev_dup)
          { //its still a duplicate, we are adding
            add_to_duplicity_ += static_cast<int> (sign); //so, just add or remove one
          } else if ((explained_[vec[i]] == 1) && prev_dup)
          { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
            add_to_duplicity_ -= 2;
          } else if ((explained_[vec[i]] > 1) && !prev_dup)
          { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
            add_to_duplicity_ += 2;
          }
        }

        //update explained and duplicity values...
        previous_explained_value += add_to_explained;
        previous_duplicity_ += add_to_duplicity_;
      }

      void updateCMDuplicity(std::vector<int> & vec, std::vector<int> & occupancy_vec, float sign) {
        int add_to_duplicity_ = 0;
        for (size_t i = 0; i < vec.size (); i++)
        {
          bool prev_dup = occupancy_vec[vec[i]] > 1;
          occupancy_vec[vec[i]] += static_cast<int> (sign);
          if ((occupancy_vec[vec[i]] > 1) && prev_dup)
          { //its still a duplicate, we are adding
            add_to_duplicity_ += static_cast<int> (sign); //so, just add or remove one
          } else if ((occupancy_vec[vec[i]] == 1) && prev_dup)
          { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
            add_to_duplicity_ -= 2;
          } else if ((occupancy_vec[vec[i]] > 1) && !prev_dup)
          { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
            add_to_duplicity_ += 2;
          }
        }

        previous_duplicity_complete_models_ += add_to_duplicity_;
      }

      float getTotalExplainedInformation(std::vector<int> & explained_, std::vector<double> & explained_by_RM_distance_weighted, int * duplicity_)
      {
        float explained_info = 0;
        int duplicity = 0;

        for (size_t i = 0; i < explained_.size (); i++)
        {
          //if (explained_[i] > 0)
          if (explained_[i] == 1)
            explained_info += explained_by_RM_distance_weighted[i];

          if (explained_[i] > 1)
            duplicity += explained_[i];
        }

        *duplicity_ = duplicity;

        return explained_info;
      }

      float getTotalBadInformation(std::vector<boost::shared_ptr<RecognitionModel> > & recog_models)
      {
        float bad_info = 0;
        for (size_t i = 0; i < recog_models.size (); i++)
          bad_info += recog_models[i]->outliers_weight_ * static_cast<float> (recog_models[i]->bad_information_);

        return bad_info;
      }

      float getUnexplainedInformationInNeighborhood(std::vector<double> & unexplained, std::vector<int> & explained)
      {
        float unexplained_sum = 0.f;
        for (size_t i = 0; i < unexplained.size (); i++)
        {
          if (unexplained[i] > 0 && explained[i] == 0)
            unexplained_sum += unexplained[i];
        }

        return unexplained_sum;
      }

      float
      getModelConstraintsValueForActiveSolution (const std::vector<bool> & active)
      {
        float bad_info = 0;
        for (size_t i = 0; i < recognition_models_.size (); i++)
        {
          if (active[i])
            bad_info += recognition_models_[i]->model_constraints_value_;
        }

        return bad_info;
      }

      float
      getModelConstraintsValue (typename pcl::PointCloud<ModelT>::Ptr & cloud)
      {
        float under = 0;
        for (int i = 0; i < model_constraints_.size (); i++)
        {
          under += model_constraints_[i] (cloud) * model_constraints_weights_[i];
        }

        return under;
      }

      //Performs smooth segmentation of the scene cloud and compute the model cues
      void
      initialize();

      mets::gol_type
      evaluateSolution(const std::vector<bool> & active, int changed);

      /*bool
      addModel(typename pcl::PointCloud<ModelT>::ConstPtr & model,
                typename pcl::PointCloud<ModelT>::ConstPtr & complete_model,
                pcl::PointCloud<pcl::Normal>::ConstPtr & model_normals,
                boost::shared_ptr<RecognitionModel> & recog_model,
                std::vector<int> & visible_indices,
                float extra_weight = 1.f);*/

      bool addModel(int i, boost::shared_ptr<RecognitionModel> & recog_model);

      void
      computeClutterCue(boost::shared_ptr<RecognitionModel> & recog_model);

      void
      SAOptimize(std::vector<int> & cc_indices, std::vector<bool> & sub_solution);

      void
      fill_structures(std::vector<int> & cc_indices, std::vector<bool> & sub_solution, SAModel & model);

      void
      clear_structures();

      float
      countActiveHypotheses (const std::vector<bool> & sol);

      boost::shared_ptr<CostFunctionLogger> cost_logger_;
      bool initial_status_;

      void computeYUVHistogram(std::vector<Eigen::Vector3f> & yuv_values, Eigen::VectorXf & histogram);

      void computeHueHistogram(std::vector<Eigen::Vector3f> & hsv_values, Eigen::VectorXf & histogram);

      void convertToHSV(int ri, int gi, int bi, Eigen::Vector3f & hsv) {
        float r = ri / 255.f;
        float g = gi / 255.f;
        float b = bi / 255.f;
        //std::cout << "rgb:" << r << " " << g << " " << b << std::endl;
        float max_color = std::max(r,std::max(g,b));
        float min_color = std::min(r,std::min(g,b));
        float h,s,v;
        h = 0;
        if(min_color == max_color)
        {
          h = 0;
        }
        else
        {

          if(max_color == r) {
            h = 60.f * (0 + (g-b) / (max_color - min_color));
          } else if (max_color == g) {
            h = 60.f * (2 + (b-r) / (max_color - min_color));
          } else if(max_color == b) {
            h = 60.f * (4 + (r-g) / (max_color - min_color));
          }

          if(h < 0) {
            h += 360.f;
          }

        }

        hsv[0] = h / 360.f;
        if(max_color == 0.f)
        {
          hsv[1] = 0.f;
        }
        else
        {
          hsv[1] = (max_color - min_color) / max_color;
        }

        hsv[2] = (max_color - min_color) / 2.f;
      }

      std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr > normals_for_visibility_;
      double eps_angle_threshold_;
      int min_points_;
      float curvature_threshold_;
      float cluster_tolerance_;

    public:
      GlobalHypothesesVerification() : faat_pcl::HypothesisVerification<ModelT, SceneT>()
      {
        resolution_ = 0.005f;
        max_iterations_ = 5000;
        regularizer_ = 1.f;
        radius_normals_ = 0.01f;
        initial_temp_ = 1000;
        detect_clutter_ = true;
        radius_neighborhood_GO_ = 0.03f;
        clutter_regularizer_ = 5.f;
        res_occupancy_grid_ = 0.005f;
        w_occupied_multiple_cm_ = 4.f;
        use_conflict_graph_ = false;
        ignore_color_even_if_exists_ = true;
        color_sigma_ = 50.f;
        opt_type_ = 2;
        use_replace_moves_ = true;
        active_hyp_penalty_ = 0.f;
        requires_normals_ = false;
        initial_status_ = false;

        eps_angle_threshold_ = 0.25;
        min_points_ = 20;
        curvature_threshold_ = 0.04f;
        cluster_tolerance_ = 0.015f;
      }

      void setSmoothSegParameters(float t_eps, float curv_t, float dist_t, int min_points = 20)
      {
        eps_angle_threshold_ = t_eps;
        min_points_ = min_points;
        curvature_threshold_ = curv_t;
        cluster_tolerance_ = dist_t;
      }

      void setObjectIds(std::vector<std::string> & ids) {
        object_ids_ = ids;
      }

      void writeToLog(std::ofstream & of, bool all_costs_=false) {
        cost_logger_->writeToLog(of);
        if(all_costs_) {
          cost_logger_->writeEachCostToLog(of);
        }
      }

      void setHypPenalty(float p) {
        active_hyp_penalty_ = p;
      }

      void
      setInitialStatus(bool b) {
        initial_status_ = b;
      }

      /*void logCosts() {
        cost_logger_.reset(new CostFunctionLogger());
      }*/

      pcl::PointCloud<pcl::PointXYZI>::Ptr getSmoothClusters()
      {
        return clusters_cloud_;
      }

      float getResolution() {
        return resolution_;
      }

      void setRequiresNormals(bool b) {
        requires_normals_ = b;
      }

      void
      setUseReplaceMoves(bool u) {
        use_replace_moves_ = u;
      }

      void
      setOptimizerType(int t) {
        opt_type_ = t;
      }

      void
      verify();

      void
      addModelConstraint (boost::function<int
      (const typename pcl::PointCloud<ModelT>::Ptr)> & f, float weight = 1.f)
      {
        model_constraints_.push_back (f);
        model_constraints_weights_.push_back (weight);
      }

      void
      clearModelConstraints ()
      {
        model_constraints_.clear ();
        model_constraints_weights_.clear ();
      }

      void
      setIgnoreColor (bool i)
      {
        ignore_color_even_if_exists_ = i;
      }

      void setColorSigma(float s) {
        color_sigma_ = s;
      }

      void
      setUseConflictGraph (bool u)
      {
        use_conflict_graph_ = u;
      }

      void setRadiusNormals(float r)
      {
        radius_normals_ = r;
      }

      void setMaxIterations(int i)
      {
        max_iterations_ = i;
      }

      void setInitialTemp(float t)
      {
        initial_temp_ = t;
      }

      void setRegularizer(float r)
      {
        regularizer_ = r;
        w_occupied_multiple_cm_ = regularizer_;
      }

      void setRadiusClutter(float r)
      {
        radius_neighborhood_GO_ = r;
      }

      void setClutterRegularizer(float cr)
      {
        clutter_regularizer_ = cr;
      }

      void setDetectClutter(bool d)
      {
        detect_clutter_ = d;
      }

      //Same length as the recognition models
      void
      setExtraWeightVectorForInliers (std::vector<float> & weights) {
        extra_weights_.clear();
        extra_weights_ = weights;
      }

      void addNormalsForVisibility(std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr > & complete_normals_for_visibility) {
        normals_for_visibility_ = complete_normals_for_visibility;
      }
  };
}

#endif /* FAAT_PCL_ */
