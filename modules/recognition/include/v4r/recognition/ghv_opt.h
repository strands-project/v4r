/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/


#ifndef V4R_GHV_OPT_H_
#define V4R_GHV_OPT_H_

#include <pcl/pcl_macros.h>
#include <pcl/common/common.h>
#include <boost/function.hpp>
#include <boost/random.hpp>
#include <metslib/mets.hh>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <map>
#include <iostream>
#include <fstream>
#include <v4r/core/macros.h>
#include <v4r/recognition/hypotheses_verification.h>

namespace v4r
{
  template<typename ModelT, typename SceneT> class V4R_EXPORTS HypothesisVerification;

  template<typename ModelT, typename SceneT>
  class V4R_EXPORTS GHVSAModel : public mets::evaluable_solution
  {
    typedef HypothesisVerification<ModelT, SceneT> SAOptimizerT;

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
      const GHVSAModel& s = dynamic_cast<const GHVSAModel&> (o);
      solution_ = s.solution_;
      opt_ = s.opt_;
      cost_ = s.cost_;
    }

    void
    copy_from (const mets::feasible_solution& o)
    {
      const GHVSAModel& s = dynamic_cast<const GHVSAModel&> (o);
      solution_ = s.solution_;
      opt_ = s.opt_;
      cost_ = s.cost_;
    }

    mets::gol_type
    what_if (int /*index*/, bool /*val*/) const
    {
      return static_cast<mets::gol_type> (0);
    }

    mets::gol_type
    apply_and_evaluate (int index, bool val)
    {
      solution_[index] = val;
      mets::gol_type sol = opt_->evaluateSolution (solution_, index); //this will update the state of the solution
      cost_ = sol;
      return sol;
    }

    void
    apply (int /*index*/, bool /*val*/)
    {

    }

    void
    unapply (int index, bool val)
    {
      solution_[index] = val;
      //update optimizer solution
      cost_ = opt_->evaluateSolution (solution_, index); //this will udpate the cost function in opt_
    }
    void
    setSolution (const std::vector<bool> & sol)
    {
      solution_ = sol;
    }

    void
    setOptimizer (SAOptimizerT *opt)
    {
      opt_ = opt;
    }
  };

  /**
   * @brief Represents a generic move from which all move types should inherit
   */
  class V4R_EXPORTS GHVgeneric_move : public mets::mana_move
  {
  public:
    virtual mets::gol_type
    evaluate (const mets::feasible_solution& cs) const = 0;
    virtual mets::gol_type
    apply_and_evaluate (mets::feasible_solution& cs) = 0;
    virtual void
    apply (mets::feasible_solution& s) const = 0;
    virtual void
    unapply (mets::feasible_solution& s) const = 0;
    virtual mets::clonable*
    clone () const = 0;
    virtual size_t
    hash () const = 0;
    virtual bool
    operator== (const mets::mana_move&) const = 0;
  };

  /**
   * @brief Represents a move that deactivates an active hypothesis replacing it by an inactive one
   * Such moves should be done when the temperature is low
   * It is based on the intersection of explained points between hypothesis
   */
  template<typename ModelT, typename SceneT>
  class V4R_EXPORTS GHVreplace_hyp_move : public GHVgeneric_move
  {
    size_t i_, j_; //i_ is an active hypothesis, j_ is an inactive hypothesis
    size_t sol_size_;

  public:
    GHVreplace_hyp_move (size_t i, size_t j, size_t sol_size) :
      i_ (i), j_ (j), sol_size_ (sol_size)
    {
    }

    mets::gol_type
    evaluate (const mets::feasible_solution& cs) const
    {
      GHVSAModel<ModelT, SceneT> model;
      model.copy_from (cs);
      model.apply_and_evaluate (i_, !model.solution_[i_]);
      mets::gol_type cost = model.apply_and_evaluate (j_, !model.solution_[j_]);
      //unapply moves now
      model.unapply (j_, !model.solution_[j_]);
      model.unapply (i_, !model.solution_[i_]);
      return cost;
    }

    mets::gol_type
    apply_and_evaluate (mets::feasible_solution& cs)
    {
      GHVSAModel<ModelT, SceneT>& model = dynamic_cast<GHVSAModel<ModelT, SceneT>&> (cs);
      assert (model.solution_[i_]);
      model.apply_and_evaluate (i_, !model.solution_[i_]);
      assert (!model.solution_[j_]);
      return model.apply_and_evaluate (j_, !model.solution_[j_]);
    }

    void
    apply (mets::feasible_solution& s) const
    {
      GHVSAModel<ModelT, SceneT>& model = dynamic_cast<GHVSAModel<ModelT, SceneT>&> (s);
      model.apply_and_evaluate (i_, !model.solution_[i_]);
      model.apply_and_evaluate (j_, !model.solution_[j_]);
    }

    void
    unapply (mets::feasible_solution& s) const
    {
      //go back
      GHVSAModel<ModelT, SceneT>& model = dynamic_cast<GHVSAModel<ModelT, SceneT>&> (s);
      model.unapply (j_, !model.solution_[j_]);
      model.unapply (i_, !model.solution_[i_]);
    }

    mets::clonable*
    clone () const
    {
      GHVreplace_hyp_move * m = new GHVreplace_hyp_move (i_, j_, sol_size_);
      return static_cast<mets::clonable*> (m);
    }

    size_t
    hash () const;
    /*{
      return static_cast<size_t> (sol_size_ + sol_size_ * i_ + j_);
    }*/

    bool
    operator== (const mets::mana_move& m) const;
    /*{
      const replace_hyp_move& mm = dynamic_cast<const replace_hyp_move&> (m);
      return (mm.i_ == i_) && (mm.j_ == j_);
    }*/
  };


  /**
   * @brief Represents a move, activate/deactivate an hypothesis
   */
  template<typename ModelT, typename SceneT>
  class GHVmove : public GHVgeneric_move
  {
    int index_;
  public:
    GHVmove (int i) :
      index_ (i)
    {
    }

    int
    getIndex ()
    {
      return index_;
    }

    mets::gol_type
    evaluate (const mets::feasible_solution& cs) const
    {
      //mets::copyable copyable = dynamic_cast<mets::copyable> (&cs);
      GHVSAModel<ModelT, SceneT> model;
      model.copy_from (cs);
      mets::gol_type cost = model.apply_and_evaluate (index_, !model.solution_[index_]);
      model.apply_and_evaluate (index_, !model.solution_[index_]);
      return cost;
    }

    mets::gol_type
    apply_and_evaluate (mets::feasible_solution& cs)
    {
      GHVSAModel<ModelT, SceneT>& model = dynamic_cast<GHVSAModel<ModelT, SceneT>&> (cs);
      return model.apply_and_evaluate (index_, !model.solution_[index_]);
    }

    void
    apply (mets::feasible_solution& s) const
    {
      GHVSAModel<ModelT, SceneT>& model = dynamic_cast<GHVSAModel<ModelT, SceneT>&> (s);
      model.apply_and_evaluate (index_, !model.solution_[index_]);
    }

    void
    unapply (mets::feasible_solution& s) const
    {
      GHVSAModel<ModelT, SceneT>& model = dynamic_cast<GHVSAModel<ModelT, SceneT>&> (s);
      model.unapply (index_, !model.solution_[index_]);
    }

    mets::clonable*
    clone () const
    {
      GHVmove * m = new GHVmove (index_);
      return static_cast<mets::clonable*> (m);
    }

    size_t
    hash () const;

    bool
    operator== (const mets::mana_move& m) const;
  };

  template<typename ModelT, typename SceneT>
    class V4R_EXPORTS GHVmove_activate : public GHVgeneric_move
    {
      size_t index_;
    public:
      GHVmove_activate (size_t i) :
        index_ (i)
      {
      }

      size_t
      getIndex ()
      {
        return index_;
      }

      mets::gol_type
      evaluate (const mets::feasible_solution& cs) const
      {
        //mets::copyable copyable = dynamic_cast<mets::copyable> (&cs);
        GHVSAModel<ModelT, SceneT> model;
        model.copy_from (cs);
        mets::gol_type cost = model.apply_and_evaluate (index_, true);
        model.apply_and_evaluate (index_, false);
        return cost;
      }

      mets::gol_type
      apply_and_evaluate (mets::feasible_solution& cs)
      {
        GHVSAModel<ModelT, SceneT>& model = dynamic_cast<GHVSAModel<ModelT, SceneT>&> (cs);
        return model.apply_and_evaluate (index_, true);
      }

      void
      apply (mets::feasible_solution& s) const
      {
        GHVSAModel<ModelT, SceneT>& model = dynamic_cast<GHVSAModel<ModelT, SceneT>&> (s);
        model.apply_and_evaluate (index_, true);
      }

      void
      unapply (mets::feasible_solution& s) const
      {
        GHVSAModel<ModelT, SceneT>& model = dynamic_cast<GHVSAModel<ModelT, SceneT>&> (s);
        model.unapply (index_, false);
      }

      mets::clonable*
      clone () const
      {
        GHVmove_activate * m = new GHVmove_activate (index_);
        return static_cast<mets::clonable*> (m);
      }

      size_t
      hash () const;

      bool
      operator== (const mets::mana_move& m) const;
    };

    template<typename ModelT, typename SceneT>
      class GHVmove_deactivate : public GHVgeneric_move
      {
        size_t index_;
        size_t problem_size_;
      public:
        GHVmove_deactivate (size_t i, size_t problem_size) :
            index_ (i), problem_size_(problem_size)
        {
        }

        size_t
        getIndex ()
        {
          return index_;
        }

        mets::gol_type
        evaluate (const mets::feasible_solution& cs) const
        {
          //mets::copyable copyable = dynamic_cast<mets::copyable> (&cs);
          GHVSAModel<ModelT, SceneT> model;
          model.copy_from (cs);
          mets::gol_type cost = model.apply_and_evaluate (index_, false);
          model.apply_and_evaluate (index_, true);
          return cost;
        }

        mets::gol_type
        apply_and_evaluate (mets::feasible_solution& cs)
        {
          GHVSAModel<ModelT, SceneT>& model = dynamic_cast<GHVSAModel<ModelT, SceneT>&> (cs);
          return model.apply_and_evaluate (index_, false);
        }

        void
        apply (mets::feasible_solution& s) const
        {
          GHVSAModel<ModelT, SceneT>& model = dynamic_cast<GHVSAModel<ModelT, SceneT>&> (s);
          model.apply_and_evaluate (index_, false);
        }

        void
        unapply (mets::feasible_solution& s) const
        {
          GHVSAModel<ModelT, SceneT>& model = dynamic_cast<GHVSAModel<ModelT, SceneT>&> (s);
          model.unapply (index_, true);
        }

        mets::clonable*
        clone () const
        {
          GHVmove_deactivate * m = new GHVmove_deactivate (index_, problem_size_);
          return static_cast<mets::clonable*> (m);
        }

        size_t
        hash () const;

        bool
        operator== (const mets::mana_move& m) const;
      };

    template<typename ModelT, typename SceneT>
      class V4R_EXPORTS GHVmove_manager
      {
        bool use_replace_moves_;
      public:
        std::vector<GHVgeneric_move*> moves_m;
        Eigen::MatrixXf intersection_cost_;
        typedef typename std::vector<GHVgeneric_move*>::iterator iterator;
        size_t problem_size_;
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

        GHVmove_manager (size_t problem_size, bool rp_moves = true)
        {
          use_replace_moves_ = rp_moves;
          problem_size_ = problem_size;

          /*for (int ii = 0; ii != problem_size; ++ii)
            moves_m.push_back (new move<ModelT, SceneT> (ii));*/
        }

        ~GHVmove_manager ()
        {
          // delete all moves
          for (iterator ii = begin (); ii != end (); ++ii)
            delete (*ii);
        }

        void
        setExplainedPointIntersections (Eigen::MatrixXf &intersection_cost)
        {
            intersection_cost_ = intersection_cost;
        }

        void
        refresh (mets::feasible_solution& s);
      };

  template<typename ModelT, typename SceneT>
  class V4R_EXPORTS GHVCostFunctionLogger : public mets::solution_recorder
  {
    std::vector<float> costs_;
    std::vector<float> costs_each_time_evaluated_;
    int times_evaluated_;
    boost::function<void (const std::vector<bool> &, float, int)> visualize_function_;

  public:
    GHVCostFunctionLogger ();

    GHVCostFunctionLogger (mets::evaluable_solution& best) :
      mets::solution_recorder (), best_ever_m (best)
    {
      times_evaluated_ = 0;
      costs_.resize (1);
      costs_[0] = 0.f;

      // costs_.resize (0); before merge it was like this ---> What is correct?
    }

    void setVisualizeFunction(boost::function<void (const std::vector<bool> &, float, int)> & f)
    {
        visualize_function_ = f;
    }

    void
    writeToLog (std::ofstream & of)
    {
      const GHVSAModel<ModelT, SceneT>& ss = static_cast<const GHVSAModel<ModelT, SceneT>&> (best_ever_m);
      of << times_evaluated_ << "\t\t";
      of << costs_.size () << "\t\t";
      of << costs_[costs_.size () - 1] << std::endl;
    }

    void
    writeEachCostToLog (std::ofstream & of)
    {
      for (size_t i = 0; i < costs_each_time_evaluated_.size (); i++)
      {
        of << costs_each_time_evaluated_[i] << "\t";
      }
      of << std::endl;
    }

    void
    addCost (float c)
    {
      costs_.push_back (c);
    }

    void
    addCostEachTimeEvaluated (float c)
    {
      costs_each_time_evaluated_.push_back (c);
    }

    void
    increaseEvaluated ()
    {
      times_evaluated_++;
    }

    int
    getTimesEvaluated ()
    {
      return times_evaluated_;
    }

    size_t
    getAcceptedMovesSize ()
    {
      return costs_.size ();
    }

    bool
    accept (const mets::feasible_solution& sol)
    {
      const mets::evaluable_solution& s = dynamic_cast<const mets::evaluable_solution&> (sol);
      if (s.cost_function () < best_ever_m.cost_function ())
      {
        best_ever_m.copy_from (s);
        const GHVSAModel<ModelT, SceneT>& ss = static_cast<const GHVSAModel<ModelT, SceneT>&> (sol);
        costs_.push_back (ss.cost_);
//        std::cout << "Move accepted:" << ss.cost_ << std::endl;

        if(visualize_function_)
            visualize_function_(ss.solution_, ss.cost_, times_evaluated_);

        return true;
      }
      return false;
    }

    /// @brief Returns the best solution found since the beginning.
    const mets::evaluable_solution&
    best_seen () const
    {
      return best_ever_m;
    }

    mets::gol_type
    best_cost () const
    {
      return best_ever_m.cost_function ();
    }

    void
    setBestSolution (std::vector<bool> & sol)
    {
      GHVSAModel<ModelT, SceneT>& ss = static_cast<GHVSAModel<ModelT, SceneT>&> (best_ever_m);
      for (size_t i = 0; i < sol.size (); i++)
      {
        ss.solution_[i] = sol[i];
        //std::cout << "setBestSolution" << ss.solution_[i] << " " << sol[i] << std::endl;
      }
    }

  protected:
    mets::evaluable_solution& best_ever_m;   /// @brief Records the best solution
  };
}
#endif /* FAAT_PCL_HV_GO_OPT_H_ */
