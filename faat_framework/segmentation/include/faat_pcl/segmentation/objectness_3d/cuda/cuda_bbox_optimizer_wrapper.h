/*
 * cuda_bbox_optimizer.h
 *
 *  Created on: Nov 15, 2012
 *      Author: aitor
 */

#ifndef CUDA_BBOX_OPTIMIZER_WRAPPER_H_
#define CUDA_BBOX_OPTIMIZER_WRAPPER_H_

#include "faat_pcl/segmentation/objectness_3d/objectness_common.h"
#include "pcl/recognition/3rdparty/metslib/mets.hh"
#include "pcl/filters/crop_box.h"

#include "faat_pcl/segmentation/objectness_3d/cuda/cuda_bbox_optimizer.h"

namespace faat_pcl
{
  namespace cuda
  {
    namespace segmentation
    {
      template<typename PointInT>
        class CudaBBoxOptimizerWrapper
        {
          typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
          typedef CudaBBoxOptimizerWrapper<PointInT> SAOptimizerT;

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

            mets::gol_type
            what_if (int index, bool val) const
            {
              std::vector<bool> tmp (solution_);
              tmp[index] = val;
              mets::gol_type sol = opt_->evaluateSolution (solution_, index); //evaluate without updating status
              return sol;
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
            apply (int index, bool val)
            {
              //apply does nothing as the solution has been modified, unapply will change it
              solution_[index] = val;
              //update optimizer solution
              cost_ = opt_->evaluateSolution (solution_, index); //this will udpate the cost function in opt_
            }

            void
            unapply (int index, bool val)
            {
              solution_[index] = val;
              //update optimizer solution
              cost_ = opt_->evaluateSolution (solution_, index); //this will udpate the cost function in opt_
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
           * Represents a move, deactivate a hypothesis
           */

          class move : public mets::move
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
              const SAModel& model = dynamic_cast<const SAModel&> (cs);
              return model.what_if (index_, !model.solution_[index_]);
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
              model.apply (index_, !model.solution_[index_]);
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
            std::vector<move*> moves_m;

            typedef typename std::vector<move*>::iterator iterator;
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
            refresh (mets::feasible_solution& /*s*/)
            {
              std::random_shuffle (moves_m.begin (), moves_m.end ());
            }
          };

          SAModel best_seen_;
          std::vector<bool> mask_;

          PointInTPtr cloud_;

          faat_pcl::cuda::segmentation::CudaBBoxOptimizer * cuda_optimizer_;
          int size_recog_models_;

        public:
          CudaBBoxOptimizerWrapper (float w = 2.f)
          {
            cuda_optimizer_ = new CudaBBoxOptimizer(w);
          }

          mets::gol_type
          evaluateSolution (const std::vector<bool> & active, int changed)
          {
            boost::posix_time::ptime start_time (boost::posix_time::microsec_clock::local_time ());
            //call cuda_optimizer_ to evaluate solution and update the internal status
            float cost = cuda_optimizer_->evaluateSolution(active, changed);
            boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time ();
            //std::cout << (end_time - start_time).total_microseconds () << " microsecs" << std::endl;
            return static_cast<mets::gol_type>(cost);
          }

          void
          setCloud (PointInTPtr & cloud)
          {
            cloud_ = cloud;
            //pass the cloud to the optimizer in some cuda format
            thrust::host_vector<float> x_points, y_points, z_points;
            x_points.resize(cloud->points.size());
            y_points.resize(cloud->points.size());
            z_points.resize(cloud->points.size());
            for(size_t i=0; i < cloud->points.size(); i++) {
              x_points[i] = cloud->points[i].x;
              y_points[i] = cloud->points[i].y;
              z_points[i] = cloud->points[i].z;
            }

            cuda_optimizer_->setCloud(x_points, y_points, z_points);
          }

          void
          setResolution (float r)
          {
            cuda_optimizer_->setResolution(r);
          }

          void
          setMinMaxValues (float minx, float maxx, float miny, float maxy, float minz, float maxz)
          {
            cuda_optimizer_->setMinMaxValues(minx, maxx, miny, maxy, minz, maxz);
          }

          void
          addModels (std::vector<BBox> & bbox_models, std::vector<float> & free_space) {
            pcl::ScopeTime t("time adding models to cuda...");
            size_recog_models_ = static_cast<int>(bbox_models.size());
            cuda_optimizer_->addModels(bbox_models, free_space);
            std::cout << "Adding models in wrapper" << " " << size_recog_models_ << std::endl;
          }

          void
          setAngleIncr(int incr) {
            cuda_optimizer_->setAngleIncr(incr);
          }

          void optimize() {
            std::vector<bool> initial_solution (size_recog_models_, true);
            mask_.resize (initial_solution.size ());
            for (size_t i = 0; i < initial_solution.size (); i++)
              mask_[i] = initial_solution[i];

            //call the cuda optimizier with the initial solution to initialize arrays handling the optimization
            {
              pcl::ScopeTime time_init("initializeOptimization");
              cuda_optimizer_->initializeOptimization(initial_solution);
            }

            //define simulated annealing
            SAModel model;
            model.cost_ = static_cast<mets::gol_type> (cuda_optimizer_->getCost());

            model.setSolution (initial_solution);
            model.setOptimizer (this);
            SAModel best (model);

            move_manager neigh (size_recog_models_);

            mets::best_ever_solution best_recorder (best);
            mets::noimprove_termination_criteria noimprove (15000);
            mets::linear_cooling linear_cooling;
            mets::simulated_annealing<move_manager> sa (model, best_recorder, neigh, noimprove, linear_cooling, 1500, 1e-7, 1);
            sa.setApplyAndEvaluate (true);

            {
              pcl::ScopeTime t ("SA search...");
              sa.search ();
            }

            //get final mask
            best_seen_ = static_cast<const SAModel&> (best_recorder.best_seen ());
            std::cout << "Final cost:" << best_seen_.cost_ << std::endl;

            for (size_t i = 0; i < best_seen_.solution_.size (); i++)
            {
              mask_[i] = (best_seen_.solution_[i]);
            }
          }

          void
          getMask (std::vector<bool> & mask)
          {
            mask = mask_;
          }

        };
    }
  }
}

#endif /* CUDA_BBOX_OPTIMIZER_H_ */
