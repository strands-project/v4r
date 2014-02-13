/*
 * bbox_optimizer.h
 *
 *  Created on: Aug 7, 2012
 *      Author: aitor
 */

#ifndef BBOX_OPTIMIZER_H_
#define BBOX_OPTIMIZER_H_

#include "faat_pcl/segmentation/objectness_3d/objectness_common.h"
#include "pcl/recognition/3rdparty/metslib/mets.hh"
#include "pcl/filters/crop_box.h"
#include "pcl/common/time.h"

template<typename PointInT>
  class BBoxOptimizer
  {
  private:
    typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;

    struct RecognitionModel
    {
      BBox box_;
      std::vector<int> explained_; //indices vector referencing explained_by_RM_
      std::vector<int> explained_fullcoud; //indices vector referencing explained_by_RM_
      std::vector<double> explained_distances_;
      float free_space_;
      std::vector<int> unexplained_in_neighborhood; //indices vector referencing unexplained_by_RM_neighboorhods
    };

    typedef BBoxOptimizer<PointInT> SAOptimizerT;
    class SAModel : public mets::evaluable_solution
    {
    public:
      std::vector<bool> solution_;
      SAOptimizerT * opt_;
      mets::gol_type cost_;

      //Evaluates the current solution
      mets::gol_type
      cost_function () const
      {
        return cost_;
      }

      void
      copy_from (const mets::copyable& o)
      {
        const SAModel& s = dynamic_cast<const SAModel&> (o);
        solution_ = s.solution_;
        opt_ = s.opt_;
        cost_ = s.cost_;
      }

      void
      copy_from (const mets::feasible_solution& o)
      {
        const SAModel& s = dynamic_cast<const SAModel&> (o);
        solution_ = s.solution_;
        opt_ = s.opt_;
        cost_ = s.cost_;
      }

      mets::gol_type
      what_if (int index, bool val) const
      {
        std::cout << "what_if being called..." << std::endl;
        return static_cast<mets::gol_type>(0);
      }

      mets::gol_type
      apply_and_evaluate (int index, bool val)
      {
        solution_[index] = val;
        mets::gol_type sol = opt_->evaluateSolution (solution_, index); //this will update the state of the solution
        cost_ = sol;

        /*std::cout << "Cost with incremental update:" << cost_ << std::endl;
        opt_->evaluateSolution (solution_);
        sleep(1);*/

        return sol;
      }

      void
      apply (int index, bool val)
      {
        std::cout << "Apply being called..." << std::endl;
        /*solution_[index] = val;
        cost_ = opt_->evaluateSolution (solution_, index); //this will udpate the cost function in opt_*/
      }

      void
      unapply (int index, bool val)
      {
        //std::cout << "Unapply being called..." << std::endl;
        solution_[index] = val;
        //update optimizer solution
        //std::cout << "Move is being unapplied" << std::endl;
        cost_ = opt_->evaluateSolution (solution_, index); //this will udpate the cost function in opt_

        /*std::cout << "Cost with incremental update:" << cost_ << std::endl;
        opt_->evaluateSolution (solution_);*/
        //sleep(1);
      }
      void
      setSolution (std::vector<bool> & sol)
      {
        solution_ = sol;
      }

      void
      setOptimizer (SAOptimizerT * opt)
      {
        opt_ = opt;
      }
    };

    /*
     * Represents a generic move from which all move types should inherit
     */

    class generic_move : public mets::move
    {
      public:
        virtual mets::gol_type evaluate(const mets::feasible_solution& cs) const = 0;
        virtual mets::gol_type apply_and_evaluate(mets::feasible_solution& cs) = 0;
        virtual void apply(mets::feasible_solution& s) const = 0;
        virtual void unapply(mets::feasible_solution& s) const = 0;
        //virtual mets::clonable* clone() const = 0;
        //virtual size_t hash() const = 0;
        //virtual bool operator==(const mets::mana_move&) const = 0;
    };

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
     * Represents a move, deactivate a hypothesis
     */

    class move : public generic_move
    {
    public:
      int index_;
      int applied_;

      move (int i) :
        index_ (i)
      {
        applied_ = 0;
      }

      mets::gol_type
      evaluate (const mets::feasible_solution& cs) const
      {
        //std::cout << "What if... " << index_ << std::endl;
        SAModel model;
        model.copy_from(cs);
        mets::gol_type cost = model.apply_and_evaluate (index_, !model.solution_[index_]);
        model.apply_and_evaluate (index_, !model.solution_[index_]);
        return cost;
      }

      mets::gol_type
      apply_and_evaluate (mets::feasible_solution& cs)
      {
        //std::cout << "What if... " << index_ << std::endl;
        SAModel& model = dynamic_cast<SAModel&> (cs);
        applied_++;
        return model.apply_and_evaluate (index_, !model.solution_[index_]);
      }

      void
      apply (mets::feasible_solution& s) const
      {
        //std::cout << "Apply move... " << index_ << std::endl;
        SAModel& model = dynamic_cast<SAModel&> (s);
        model.apply_and_evaluate (index_, !model.solution_[index_]);
      }

      void
      unapply (mets::feasible_solution& s) const
      {
        SAModel& model = dynamic_cast<SAModel&> (s);
        model.unapply (index_, !model.solution_[index_]);
      }
    };

    class move_manager
    {
    public:
      std::vector<generic_move*> moves_m;

      typedef typename std::vector<generic_move*>::iterator iterator;
      iterator
      begin ()
      {
        return moves_m.begin ();
      }
      iterator
      end ()
      {
        return moves_m.end ();
      }

      move_manager (int problem_size)
      {
        srand (time (NULL));
        for (int ii = 0; ii != problem_size; ++ii)
        {
          moves_m.push_back (new move (ii));
        }
      }

      ~move_manager ()
      {
        // delete all moves
        for (iterator ii = begin (); ii != end (); ++ii)
          delete (*ii);
      }

      void
      refresh (mets::feasible_solution& s)
      {
        for (iterator ii = begin (); ii != end (); ++ii)
          delete (*ii);

        SAModel& model = dynamic_cast<SAModel&> (s);
        moves_m.clear ();
        moves_m.resize (model.solution_.size () + model.solution_.size () * model.solution_.size ());
        for (int ii = 0; ii != model.solution_.size (); ++ii)
          moves_m[ii] = new move (ii);

        std::vector<int> active, inactive;
        active.resize (model.solution_.size ());
        inactive.resize (model.solution_.size ());
        int nact, ninact;
        nact = ninact = 0;
        for (int i = 0; i < static_cast<int> (model.solution_.size ()); i++)
        {
          if (model.solution_[i])
          {
            active[nact] = i;
            nact++;
          }
          else
          {
            inactive[ninact] = i;
            ninact++;
          }
        }

        active.resize (nact);
        inactive.resize (ninact);

        int nm = 0;
        for (size_t i = 0; i < active.size (); ++i)
        {
          for (size_t j = (i + 1); j < inactive.size (); ++j)
          {
            moves_m[model.solution_.size () + nm] = new replace_hyp_move (active[i], inactive[j], model.solution_.size ());
            nm++;
          }
        }

        moves_m.resize (model.solution_.size () + nm);

        std::random_shuffle (moves_m.begin (), moves_m.end ());
      }
    };

    SAModel best_seen_;

    std::vector<int> explained_by_RM_;
    std::vector<int> full_cloud_explained_by_RM_;
    std::vector<double> explained_by_RM_objectness_weighted;
    std::vector<int> unexplained_by_RM_neighboorhods;
    std::vector<boost::shared_ptr<RecognitionModel> > recognition_models_;
    std::vector<bool> mask_;

    //this is already the cloud_on_plane_
    PointInTPtr cloud_;

    float min_z;
    float max_z;

    float min_x;
    float max_x;

    float min_y;
    float max_y;

    float resolution;

    double previous_explained_value;
    int previous_duplicity_;
    int previous_bad_info_;
    float previous_objectness_;
    int previous_duplicity_complete_models_;
    float previous_unexplained_;

    float bad_info_weight_;
    float unexplained_weight_;
    float duplicity_cm_weight_;
    float duplicity_weight_;

    int opt_type_;

    void
    setPreviousBadInfo (float f)
    {
      previous_bad_info_ = f;
    }

    float
    getPreviousBadInfo ()
    {
      return previous_bad_info_;
    }

    void
    setPreviousExplainedValue (double v)
    {
      previous_explained_value = v;
    }

    void
    setPreviousDuplicity (int v)
    {
      previous_duplicity_ = v;
    }

    double
    getExplainedValue ()
    {
      return previous_explained_value;
    }

    void setPreviousUnExplainedValue(float v)
    {
      previous_unexplained_ = v;
    }

    float getPreviousUnexplainedValue()
    {
      return previous_unexplained_;
    }

    int
    getDuplicity ()
    {
      return previous_duplicity_;
    }

    int getDuplicityCM()
    {
      return previous_duplicity_complete_models_;
    }

    void setPreviousDuplicityCM(int v)
    {
      previous_duplicity_complete_models_ = v;
    }

    double
    getTotalExplainedInformation (std::vector<int> & explained_, std::vector<double> & explained_by_RM_distance_weighted, int * duplicity_)
    {
      double explained_info = 0;
      int duplicity = 0;

      for (size_t i = 0; i < explained_.size (); i++)
      {
        if (explained_[i] == 1)
        {
          //explained_info++;
          explained_info += explained_by_RM_distance_weighted[i];
        }

        if (explained_[i] > 1)
          duplicity += explained_[i];
      }

      *duplicity_ = duplicity;

      return explained_info;
    }

    float
    getTotalFreeSpace (std::vector<boost::shared_ptr<RecognitionModel> > & recog_models, const std::vector<bool> & active)
    {
      float bad_info = 0;
      for (size_t i = 0; i < recog_models.size (); i++)
      {
        if (active[i])
          bad_info += recog_models[i]->free_space_ * bad_info_weight_;
      }

      return bad_info;
    }

    float
    countActiveHypotheses (const std::vector<bool> & sol)
    {
      int c = 0;
      for (size_t i = 0; i < sol.size (); i++)
      {
        if (sol[i])
          c++;
      }

      return static_cast<float> (c) * (0.f);
    }

    void
    updateCMDuplicity (std::vector<int> & vec, std::vector<int> & occupancy_vec, float sign)
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

    void
    updateUnexplainedVector (std::vector<int> & unexplained_, std::vector<int> & unexplained_by_RM,
                                std::vector<int> & explained, std::vector<int> & explained_by_RM, float val)
    {
      {

        int add_to_unexplained = 0;

        for (size_t i = 0; i < unexplained_.size (); i++)
        {

          bool prev_unexplained = (unexplained_by_RM[unexplained_[i]] > 0) && (explained_by_RM[unexplained_[i]] == 0);
          int prev_unexplained_value = unexplained_by_RM[unexplained_[i]];
          unexplained_by_RM[unexplained_[i]] += val;

          if (val < 0) //the hypothesis is being removed
          {
            if (prev_unexplained)
            {
                add_to_unexplained -= 1;
            }
          }
          else //the hypothesis is being added and unexplains unexplained_[i], so increase by 1 unless its explained by another hypothesis
          {
            if (explained_by_RM[unexplained_[i]] == 0)
              add_to_unexplained += 1;
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
          }
          else
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

    float getUnexplainedInformationInNeighborhood(std::vector<int> & unexplained, std::vector<int> & explained)
    {
      float unexplained_sum = 0.f;
      for (size_t i = 0; i < unexplained.size (); i++)
      {
        if (unexplained[i] > 0 && explained[i] == 0)
          unexplained_sum += unexplained[i];
      }

      return unexplained_sum;
    }

    void
    updateExplainedVector (std::vector<int> & vec, std::vector<double> & vec_float, std::vector<int> & explained_,
                           std::vector<double> & explained_by_RM_distance_weighted, int sign)
    {
      float add_to_explained = 0.f;
      int add_to_duplicity_ = 0;

      for (size_t i = 0; i < vec.size (); i++)
      {
        bool prev_dup = explained_[vec[i]] > 1;
        bool prev_explained = explained_[vec[i]] == 1;
        float prev_explained_value = explained_by_RM_distance_weighted[vec[i]];

        explained_[vec[i]] += sign;
        explained_by_RM_distance_weighted[vec[i]] += vec_float[i] * sign;

        //hypothesis added/removed, now point is uniquely explained
        //it could happen that by removing this hypothesis, the point gets explained by the other hypothesis explaining it..

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
          add_to_duplicity_ += sign; //so, just add or remove one
        }
        else if ((explained_[vec[i]] == 1) && prev_dup)
        { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
          add_to_duplicity_ -= 2;
        }
        else if ((explained_[vec[i]] > 1) && !prev_dup)
        { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
          add_to_duplicity_ += 2;
        }
      }

      //update explained and duplicity values...
      previous_explained_value += add_to_explained;
      previous_duplicity_ += add_to_duplicity_;
    }

  public:

    int angle_incr_;

    BBoxOptimizer (float w = 2.f)
    {
      bad_info_weight_ = w;
      unexplained_weight_ = 2.f;
      duplicity_cm_weight_ = 0.f;
      duplicity_weight_ = 0.5f;
    }

    void
    setCloud (PointInTPtr & cloud)
    {
      cloud_ = cloud;
    }

    void
    setResolution (float r)
    {
      resolution = r;
    }

    void
    setMinMaxValues (float minx, float maxx, float miny, float maxy, float minz, float maxz)
    {
      min_z = minz;
      max_z = maxz;

      min_y = miny;
      max_y = maxy;

      min_x = minx;
      max_x = maxx;

    }

    void
    addModels (std::vector<BBox> & bbox_models, std::vector<float> & free_space)
    {
      recognition_models_.clear ();

      float max_objectness = 0.f;
      for (size_t i = 0; i < bbox_models.size (); i++)
      {
        boost::shared_ptr<RecognitionModel> rm (new RecognitionModel);
        rm->box_ = bbox_models[i];
        rm->free_space_ = free_space[i];
        recognition_models_.push_back (rm);

        if (max_objectness < bbox_models[i].score)
          max_objectness = bbox_models[i].score;
      }

      /*for (size_t i = 0; i < recognition_models_.size (); i++)
      {
        recognition_models_[i]->box_.score /= max_objectness;
      }*/

      std::cout << recognition_models_.size() << " models added to the optimizer..." << std::endl;
    }

    /*mets::gol_type
    evaluateSolution (const std::vector<bool> & active) {
      int duplicity;
      //float free_space = getTotalFreeSpace (recognition_models_, active);
      float free_space = 0;
      std::vector<int> explained_by_RM__local;
      std::vector<int> full_cloud_explained_by_RM__local;
      std::vector<float> explained_by_RM_objectness_weighted_local;

      explained_by_RM__local.resize (explained_by_RM_.size (), 0);
      explained_by_RM_objectness_weighted_local.resize (explained_by_RM_objectness_weighted.size (), 0.f);
      full_cloud_explained_by_RM__local.resize(full_cloud_explained_by_RM_.size(), 0);
      //unexplained_by_RM_neighboorhods.clear();
      //unexplained_by_RM_neighboorhods.resize (cloud_->points.size (), 0.f);
      for (size_t i = 0; i < active.size (); i++)
      {
        if (active[i])
        {
          for (size_t j = 0; j < recognition_models_[i]->explained_.size (); j++)
          {
            explained_by_RM__local[recognition_models_[i]->explained_[j]]++;
            explained_by_RM_objectness_weighted_local[recognition_models_[i]->explained_[j]] += recognition_models_[i]->box_.score;
          }

          for (size_t j = 0; j < recognition_models_[i]->explained_fullcoud.size (); j++) {
            full_cloud_explained_by_RM__local[recognition_models_[i]->explained_fullcoud[j]]++;
          }

          //for (size_t j = 0; j < recognition_models_[i]->unexplained_in_neighborhood.size (); j++)
            //unexplained_by_RM_neighboorhods[recognition_models_[i]->unexplained_in_neighborhood[j]]++;
        }
      }

      int occupied_multiple = 0;
      for (size_t i = 0; i < full_cloud_explained_by_RM__local.size (); i++)
      {
        if (full_cloud_explained_by_RM__local[i] > 1)
        {
          occupied_multiple += full_cloud_explained_by_RM__local[i];
        }
      }

      float good_information = getTotalExplainedInformation (explained_by_RM__local, explained_by_RM_objectness_weighted_local, &duplicity);

      std::cout << "cost evaluating the whole thing...:"
          << static_cast<mets::gol_type> (
              (static_cast<float> (good_information) - static_cast<float> (duplicity)
                          - free_space - static_cast<float> (occupied_multiple) - countActiveHypotheses (active) - unexplained_in_neighboorhod * unexplained_weight_) * -1.f) << std::endl;

      std::cout << "the values when reevaluating are:" << good_information << " " << duplicity << " " << free_space << " " << countActiveHypotheses (active) << " " << unexplained_in_neighboorhod << occupied_multiple << std::endl;
    }*/

    mets::gol_type
    evaluateSolution (const std::vector<bool> & active, int changed)
    {
      boost::posix_time::ptime start_time (boost::posix_time::microsec_clock::local_time ());
      float sign = 1.f;
      //update explained_by_RM
      if (active[changed])
      {
        //it has been activated
        updateExplainedVector (recognition_models_[changed]->explained_, recognition_models_[changed]->explained_distances_, explained_by_RM_,
                               explained_by_RM_objectness_weighted, 1);

        //updateCMDuplicity (recognition_models_[changed]->explained_fullcoud, full_cloud_explained_by_RM_, 1.f);

        updateUnexplainedVector (recognition_models_[changed]->unexplained_in_neighborhood, unexplained_by_RM_neighboorhods,
                                 recognition_models_[changed]->explained_, explained_by_RM_, 1.f);
      }
      else
      {
        //it has been deactivated
        updateExplainedVector (recognition_models_[changed]->explained_, recognition_models_[changed]->explained_distances_, explained_by_RM_,
                               explained_by_RM_objectness_weighted, -1);

        //updateCMDuplicity (recognition_models_[changed]->explained_fullcoud, full_cloud_explained_by_RM_, -1.f);
        updateUnexplainedVector (recognition_models_[changed]->unexplained_in_neighborhood, unexplained_by_RM_neighboorhods,
                                 recognition_models_[changed]->explained_, explained_by_RM_, -1.f);
        sign = -1.f;
      }

      int duplicity = getDuplicity ();
      double good_info = getExplainedValue ();
      float unexplained = getPreviousUnexplainedValue();

      /*float bad_info = static_cast<float> (getPreviousBadInfo ()) + static_cast<float> (recognition_models_[changed]->free_space_) * bad_info_weight_
          * sign;*/

      float bad_info = getTotalFreeSpace(recognition_models_, active);

      setPreviousBadInfo (bad_info);

      float duplicity_cm = static_cast<float> (getDuplicityCM ()) * duplicity_cm_weight_;

      //std::cout << good_info << " " << duplicity << " " << bad_info << " " << countActiveHypotheses (active) << " " << /*" " << unexplained_in_neighboorhod <<*/ duplicity_cm << std::endl;

      float cost = (static_cast<float> (good_info) -
          bad_info - static_cast<float> (duplicity) * duplicity_weight_ -
          countActiveHypotheses (active) - duplicity_cm - unexplained * unexplained_weight_) * -1.f;

      boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time ();
      //std::cout << (end_time - start_time).total_microseconds () << " microsecs" << std::endl;
      return static_cast<mets::gol_type> (cost); //return the dual to our max problem
    }

    inline void getInsideBox(BBox & bb, PointInTPtr & cloud, std::vector<int> & inside_box) {
      int v = bb.angle;
      Eigen::Affine3f incr_rot_trans;
      incr_rot_trans.setIdentity ();

      Eigen::Vector4f minxyz, maxxyz;
      minxyz[0] = min_x + (bb.x) * resolution - resolution / 2.f;
      minxyz[1] = min_y + (bb.y) * resolution - resolution / 2.f;
      minxyz[2] = min_z + (bb.z) * resolution - resolution / 2.f;
      minxyz[3] = 1.f;

      maxxyz[0] = min_x + (bb.sx + bb.x) * resolution + resolution / 2.f;
      maxxyz[1] = min_y + (bb.sy + bb.y) * resolution + resolution / 2.f;
      maxxyz[2] = min_z + (bb.sz + bb.z) * resolution + resolution / 2.f;
      maxxyz[3] = 1.f;

      /*Eigen::Vector4f minxyz, maxxyz;
      minxyz[0] = min_x + bb.x * resolution;
      minxyz[1] = min_y + bb.y * resolution;
      minxyz[2] = min_z + bb.z * resolution;
      minxyz[3] = 1.f;

      maxxyz[0] = min_x + (bb.sx + bb.x) * resolution;
      maxxyz[1] = min_y + (bb.sy + bb.y) * resolution;
      maxxyz[2] = min_z + (bb.sz + bb.z) * resolution;
      maxxyz[3] = 1.f;*/

      if (v != 0)
      {
        float rot_rads = pcl::deg2rad (static_cast<float> (angle_incr_ * v));
        incr_rot_trans = Eigen::Affine3f (Eigen::AngleAxisf (static_cast<float> (rot_rads), Eigen::Vector3f::UnitZ ()));
      }

      {
        pcl::CropBox<PointInT> cb;
        cb.setInputCloud (cloud);
        cb.setMin (minxyz);
        cb.setMax (maxxyz);
        cb.setTransform (incr_rot_trans);
        cb.filter (inside_box);
      }
    }

    void setOptType(int t) {
      opt_type_ = t;
    }

    void
    optimize ()
    {
      full_cloud_explained_by_RM_.clear ();
      unexplained_by_RM_neighboorhods.clear ();
      explained_by_RM_.clear ();
      explained_by_RM_objectness_weighted.clear ();
      mask_.clear ();

      float resolution_full_cloud_ = 0.01f;
      int GRIDSIZE_X = (int)((max_x - min_x) / resolution_full_cloud_);
      int GRIDSIZE_Y = (int)((max_y - min_y) / resolution_full_cloud_);
      int GRIDSIZE_Z = (int)((max_z - min_z) / resolution_full_cloud_);

      PointInTPtr full_cloud (new pcl::PointCloud<PointInT>);
      full_cloud->width = (GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z);
      full_cloud->height = 1;
      full_cloud->points.resize ((GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z));

      for (int xx = 0; xx < GRIDSIZE_X; xx++)
      {
        for (int yy = 0; yy < GRIDSIZE_Y; yy++)
        {
          for (int zz = 0; zz < GRIDSIZE_Z; zz++)
          {
            Eigen::Vector4f vec;
            float x = min_x + xx * resolution_full_cloud_ + resolution_full_cloud_ / 2.f;
            float y = min_y + yy * resolution_full_cloud_ + resolution_full_cloud_ / 2.f;
            float z = min_z + zz * resolution_full_cloud_ + resolution_full_cloud_ / 2.f;
            int idx = zz * GRIDSIZE_X * GRIDSIZE_Y + yy * GRIDSIZE_X + xx;
            vec = Eigen::Vector4f (x, y, z, 0);
            full_cloud->points[idx].getVector4fMap () = vec;
          }
        }
      }

      //fill explained_by_RM_ using the recognition_models_
      full_cloud_explained_by_RM_.resize (full_cloud->points.size (), 0);
      explained_by_RM_.resize (cloud_->points.size (), 0);
      unexplained_by_RM_neighboorhods.resize (cloud_->points.size (), 0);
      explained_by_RM_objectness_weighted.resize (cloud_->points.size (), 0);

      std::vector<bool> initial_solution (recognition_models_.size (), true);

      {
        pcl::ScopeTime t ("cropping models...");

        for (size_t i = 0; i < recognition_models_.size (); i++)
        {
          BBox bb = recognition_models_[i]->box_;
          std::vector<int> inside_box;
          std::vector<int> inside_box_fullcloud;
          getInsideBox (bb, cloud_, inside_box);

          BBox bb_shrinked;
          bb_shrinked.sx = std::max(bb.sx - 2,1);
          bb_shrinked.sy = std::max(bb.sy - 2,1);
          bb_shrinked.sz = std::max(bb.sz - 2,1);

          bb_shrinked.x = bb.x + std::max (static_cast<int> (std::floor ((bb.sx - bb_shrinked.sx) / 2.f)), 1);
          bb_shrinked.y = bb.y + std::max (static_cast<int> (std::floor ((bb.sy - bb_shrinked.sy) / 2.f)), 1);
          bb_shrinked.z = bb.z + std::max (static_cast<int> (std::floor ((bb.sz - bb_shrinked.sz) / 2.f)), 1);

          getInsideBox (bb, full_cloud, inside_box_fullcloud);
          //getInsideBox (bb_shrinked, full_cloud, inside_box_fullcloud);

          float wo = 1.f;
          if (initial_solution[i])
          {
            for (size_t j = 0; j < inside_box.size (); j++)
            {
              explained_by_RM_[inside_box[j]]++;
              explained_by_RM_objectness_weighted[inside_box[j]] += recognition_models_[i]->box_.score * wo;
            }

            for (size_t j = 0; j < inside_box_fullcloud.size (); j++)
            {
              full_cloud_explained_by_RM_[inside_box_fullcloud[j]]++;
            }
          }

          recognition_models_[i]->explained_ = inside_box;
          recognition_models_[i]->explained_fullcoud = inside_box_fullcloud;

          recognition_models_[i]->explained_distances_.clear ();
          for (size_t j = 0; j < inside_box.size (); j++)
          {
            recognition_models_[i]->explained_distances_.push_back (recognition_models_[i]->box_.score * wo);
          }

          //unexplained values in neighborhood
          BBox bb_extended;
          float expand_factor_ = 1.25f;
          int size_ring = 4;
          bb_extended.sx = static_cast<int> (pcl_round (bb.sx * expand_factor_));
          bb_extended.sy = static_cast<int> (pcl_round (bb.sy * expand_factor_));
          bb_extended.sz = static_cast<int> (pcl_round (bb.sz * expand_factor_));

          bb_extended.sx = bb.sx + size_ring * 2;
          bb_extended.sy = bb.sy + size_ring * 2;
          bb_extended.sz = bb.sz + size_ring * 2;

          bb_extended.x = bb.x - static_cast<int> (pcl_round ((bb_extended.sx - bb.sx) / 2.f));
          bb_extended.y = bb.y - static_cast<int> (pcl_round ((bb_extended.sy - bb.sy) / 2.f));
          bb_extended.z = bb.z - static_cast<int> (pcl_round ((bb_extended.sz - bb.sz) / 2.f));

          bb_extended.x = std::max (bb_extended.x, 1);
          bb_extended.y = std::max (bb_extended.y, 1);
          bb_extended.z = std::max (bb_extended.z, 1);

          bb_extended.sx = std::min (GRIDSIZE_X - 1, bb_extended.x + bb_extended.sx) - bb_extended.x;
          bb_extended.sy = std::min (GRIDSIZE_Y - 1, bb_extended.y + bb_extended.sy) - bb_extended.y;
          bb_extended.sz = std::min (GRIDSIZE_Z - 1, bb_extended.z + bb_extended.sz) - bb_extended.z;
          bb_extended.angle = bb.angle;

          //get indices inside the extended box
          std::vector<int> inside_extended_box;
          getInsideBox (bb_extended, cloud_, inside_extended_box);

          std::vector<int> inside_extended_and_not_extended;
          inside_extended_and_not_extended.insert (inside_extended_and_not_extended.begin (), inside_box.begin (), inside_box.end ());
          inside_extended_and_not_extended.insert (inside_extended_and_not_extended.begin(), inside_extended_box.begin (), inside_extended_box.end ());
          std::sort (inside_extended_and_not_extended.begin (), inside_extended_and_not_extended.end ());

          //keep all indices that appear only once representing the outer ring...
          std::vector<int> in_outer_ring;
          std::map<int, int> map_count;
          std::map<int, int>::iterator it;
          for (size_t j = 0; j < inside_box.size (); j++) {
            it = map_count.find(inside_box[j]);
            if(it == map_count.end()) {
              map_count[inside_box[j]] = 1;
            } else {
              it->second++;
            }
          }

          for (size_t j = 0; j < inside_extended_box.size (); j++) {
            it = map_count.find(inside_extended_box[j]);
            if(it == map_count.end()) {
              in_outer_ring.push_back(inside_extended_box[j]);
            }
          }

          std::cout << "Explained points:" << inside_box.size () << " " << inside_box_fullcloud.size() << " " << in_outer_ring.size() << std::endl;

          recognition_models_[i]->unexplained_in_neighborhood = in_outer_ring;

          //check that unexplained_in_neighborhood and explained_ are disjoint sets...
          {
            std::vector<int> explained_points = recognition_models_[i]->explained_;
            std::vector<int> unexplained_points = recognition_models_[i]->unexplained_in_neighborhood;

            std::sort(explained_points.begin(),explained_points.end());
            std::sort(unexplained_points.begin(),unexplained_points.end());

            std::vector<int> together;
            together.insert(together.begin(), explained_points.begin(), explained_points.end());
            together.insert(together.begin(), unexplained_points.begin(), unexplained_points.end());
            std::sort(together.begin(), together.end());
            together.erase (std::unique (together.begin (), together.end ()), together.end ());
            std::cout << "together size:" << together.size() << " " << (explained_points.size() + unexplained_points.size()) << std::endl;
            assert(together.size() == (explained_points.size() + unexplained_points.size()));
          }

          if (initial_solution[i])
          {
            for (size_t j = 0; j < in_outer_ring.size (); j++)
            {
              unexplained_by_RM_neighboorhods[in_outer_ring[j]] += 1.f;
            }
          }
        }
      }

      mask_.resize (recognition_models_.size ());
      for (size_t i = 0; i < recognition_models_.size (); i++)
        mask_[i] = initial_solution[i];

      int duplicity;
      float free_space = getTotalFreeSpace (recognition_models_, initial_solution);
      double good_information = getTotalExplainedInformation (explained_by_RM_, explained_by_RM_objectness_weighted, &duplicity);
      float unexplained_in_neighboorhod = getUnexplainedInformationInNeighborhood (unexplained_by_RM_neighboorhods, explained_by_RM_);

      int occupied_multiple = 0;
      for (size_t i = 0; i < full_cloud_explained_by_RM_.size (); i++)
      {
        if (full_cloud_explained_by_RM_[i] > 1)
        {
          occupied_multiple += full_cloud_explained_by_RM_[i];
        }
      }

      //occupied_multiple = 0;
      setPreviousUnExplainedValue(unexplained_in_neighboorhod);
      setPreviousDuplicityCM (occupied_multiple);
      setPreviousExplainedValue (good_information);
      setPreviousDuplicity (duplicity);
      setPreviousBadInfo (free_space);

      std::cout << good_information << " " << duplicity << " " << free_space << " " << countActiveHypotheses (mask_) << /*" " << unexplained_in_neighboorhod <<*/ " " << occupied_multiple << std::endl;

      SAModel model;
      model.cost_ = static_cast<mets::gol_type> ((
                        static_cast<float> (good_information) -
                        static_cast<float> (duplicity) * duplicity_weight_ - free_space -
                        countActiveHypotheses (initial_solution) -
                        static_cast<float> (occupied_multiple) * duplicity_cm_weight_ -
                        unexplained_in_neighboorhod * unexplained_weight_)
                        * -1.f);

      model.setSolution (initial_solution);
      model.setOptimizer (this);
      SAModel best (model);

      std::cout << "Final cost:" << model.cost_ << std::endl;

      move_manager neigh (static_cast<int> (recognition_models_.size ()));

      mets::best_ever_solution best_recorder (best);
      mets::noimprove_termination_criteria noimprove (15000);

      switch(opt_type_)
      {
        case 0:
        {
          mets::local_search<move_manager> local ( model, best_recorder, neigh, 0, false);
          {
            pcl::ScopeTime t ("local search...");
            local.search ();
          }
          break;
        }
        default:
        {
          mets::linear_cooling linear_cooling;
          mets::simulated_annealing<move_manager> sa (model, best_recorder, neigh, noimprove, linear_cooling, 1500, 1e-7, 1);
          sa.setApplyAndEvaluate (true);

          {
            pcl::ScopeTime t ("SA search...");
            sa.search ();
          }
        }
      }

      //neigh.printStatistics ();
      best_seen_ = static_cast<const SAModel&> (best_recorder.best_seen ());
      std::cout << "final solution:" << std::endl;
      for (size_t i = 0; i < best_seen_.solution_.size (); i++)
      {
        mask_[i] = (best_seen_.solution_[i]);
        std::cout << mask_[i] << " ";
      }

      std::cout << std::endl;

      std::cout << "Final cost after optimization:" << best_seen_.cost_ << std::endl;

      {
        int duplicity;
        float free_space = getTotalFreeSpace (recognition_models_, mask_);

        explained_by_RM_.clear ();
        explained_by_RM_.resize (cloud_->points.size (), 0);
        explained_by_RM_objectness_weighted.clear ();
        explained_by_RM_objectness_weighted.resize (cloud_->points.size (), 0.f);
        full_cloud_explained_by_RM_.clear();
        full_cloud_explained_by_RM_.resize(full_cloud->points.size(), 0);
        unexplained_by_RM_neighboorhods.clear();
        unexplained_by_RM_neighboorhods.resize (cloud_->points.size (), 0.f);
        for (size_t i = 0; i < mask_.size (); i++)
        {
          if (mask_[i])
          {
            for (size_t j = 0; j < recognition_models_[i]->explained_.size (); j++)
            {
              explained_by_RM_[recognition_models_[i]->explained_[j]]++;
              explained_by_RM_objectness_weighted[recognition_models_[i]->explained_[j]] += recognition_models_[i]->box_.score;
            }

            for (size_t j = 0; j < recognition_models_[i]->explained_fullcoud.size (); j++) {
              full_cloud_explained_by_RM_[recognition_models_[i]->explained_fullcoud[j]]++;
            }

            for (size_t j = 0; j < recognition_models_[i]->unexplained_in_neighborhood.size (); j++)
            {
              unexplained_by_RM_neighboorhods[recognition_models_[i]->unexplained_in_neighborhood[j]]++;
            }
          }
        }

        int occupied_multiple = 0;
        for (size_t i = 0; i < full_cloud_explained_by_RM_.size (); i++)
        {
          if (full_cloud_explained_by_RM_[i] > 1)
          {
            occupied_multiple += full_cloud_explained_by_RM_[i];
          }
        }

        double good_information = getTotalExplainedInformation (explained_by_RM_, explained_by_RM_objectness_weighted, &duplicity);
        float unexplained_in_neighboorhod = getUnexplainedInformationInNeighborhood (unexplained_by_RM_neighboorhods, explained_by_RM_);

        std::cout << good_information << " " << duplicity << " " << free_space << " " << countActiveHypotheses (mask_) << " " << unexplained_in_neighboorhod << " " << occupied_multiple << std::endl;
        std::cout << "Final cost after optimization:"
            << static_cast<mets::gol_type> (
                (static_cast<float> (good_information) - static_cast<float> (duplicity) * duplicity_weight_
                    - free_space - static_cast<float> (occupied_multiple) * duplicity_cm_weight_ - countActiveHypotheses (mask_) - unexplained_in_neighboorhod * unexplained_weight_) * -1.f) << std::endl;

      }
    }

    void
    getMask (std::vector<bool> & mask)
    {
      mask = mask_;
    }

  };

#endif /* BBOX_OPTIMIZER_H_ */
