/*
 * cuda_bbox_optimizer.cu
 *
 *  Created on: Nov 15, 2012
 *      Author: aitor
 */

#include "faat_pcl/segmentation/objectness_3d/cuda/cuda_bbox_optimizer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/count.h>

namespace faat_pcl
{
  namespace cuda
  {
    namespace segmentation
    {

      template<typename T>
      struct addToArray {
        T * raw_ptr_;
        addToArray(T * ptr) : raw_ptr_(ptr) {
        }

        __device__
        void
        operator() (thrust::tuple<int, T> tp) {
          raw_ptr_[thrust::get<0>(tp)] += thrust::get<1>(tp);
        }

      };

      struct good_info
      {

        float * raw_ptr;
        good_info(float * ptr) : raw_ptr(ptr) {

        }

        __host__ __device__
        float operator()(thrust::tuple<int, int> tp)
        {

          if(thrust::get<0>(tp) == 1) {
            return raw_ptr[thrust::get<1>(tp)];
          } else {
            return 0.f;
          }
          /*if(thrust::get<0>(tp) <= 0) {
            return 0.f;
          } else {
            return raw_ptr[thrust::get<1>(tp)];
          }*/
        }

      };

      struct unexplainedKernel
      {
        __host__ __device__
        int operator()(thrust::tuple<int, int> tp)
        {
          if (thrust::get<0>(tp) > 0 && thrust::get<1>(tp) == 0) {
            return thrust::get<0>(tp);
          }

          return 0;
        }

      };

      struct duplicityKernel
      {
        __host__ __device__
        int operator()(int x)
        {
          if(x > 1) {
            return x;
          }

          return 0;
        }
      };

      void
      CudaBBoxOptimizer::initializeOptimization(std::vector<bool> & initial_sol) {

        explained_by_RM_.resize(x_points_.size(),0);
        explained_by_RM_objectness_weighted.resize(x_points_.size(),0.f);
        full_cloud_explained_by_RM_.resize(x_points_full_cloud_.size(),0);
        unexplained_by_RM_neighboorhods.resize(x_points_.size(),0);

        previous_active_hypotheses_ = 0;
        for (size_t i = 0; i < recognition_models_.size (); i++) {
          if(initial_sol[i]) {

            previous_active_hypotheses_++;
            thrust::constant_iterator<int> first(1);
            thrust::constant_iterator<float> first_float(recognition_models_[i].box_.score);

            {
              int * raw_ptr = thrust::raw_pointer_cast(explained_by_RM_.data());
              thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(recognition_models_[i].explained_.begin(), first)),
                thrust::make_zip_iterator(thrust::make_tuple(recognition_models_[i].explained_.end(), first + recognition_models_[i].explained_.size())),
                             addToArray<int>(raw_ptr));
            }

            {
              int * raw_ptr = thrust::raw_pointer_cast(full_cloud_explained_by_RM_.data());
              thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(recognition_models_[i].explained_fullcoud.begin(), first)),
                thrust::make_zip_iterator(thrust::make_tuple(recognition_models_[i].explained_fullcoud.end(), first + recognition_models_[i].explained_fullcoud.size())),
                             addToArray<int>(raw_ptr));
            }

            {
              int * raw_ptr = thrust::raw_pointer_cast(unexplained_by_RM_neighboorhods.data());
              thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(recognition_models_[i].unexplained_in_neighborhood.begin(), first)),
                thrust::make_zip_iterator(thrust::make_tuple(recognition_models_[i].unexplained_in_neighborhood.end(), first + recognition_models_[i].unexplained_in_neighborhood.size())),
                             addToArray<int>(raw_ptr));
            }

            {
              float * raw_ptr = thrust::raw_pointer_cast(explained_by_RM_objectness_weighted.data());
              thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(recognition_models_[i].explained_.begin(), first_float)),
                thrust::make_zip_iterator(thrust::make_tuple(recognition_models_[i].explained_.end(), first_float + recognition_models_[i].explained_.size())),
                             addToArray<float>(raw_ptr));
            }
          }
        }

        thrust::counting_iterator<int> idx_iterator (0);
        int * raw_ptr = thrust::raw_pointer_cast(explained_by_RM_.data());
        float * raw_ptr_explained_objectness = thrust::raw_pointer_cast(explained_by_RM_objectness_weighted.data());

        float ginfo = thrust::transform_reduce(
                                      thrust::make_zip_iterator(thrust::make_tuple(explained_by_RM_.begin(), idx_iterator)),
                                      thrust::make_zip_iterator(thrust::make_tuple(explained_by_RM_.end(), idx_iterator + explained_by_RM_.size())),
                                      good_info(raw_ptr_explained_objectness),
                                      0.f, thrust::plus<float>());

        int duplicity = thrust::transform_reduce(explained_by_RM_.begin(),
                                                 explained_by_RM_.end(),
                                                 duplicityKernel(),
                                                 0,
                                                 thrust::plus<int>());

        int duplicityCM = thrust::transform_reduce(full_cloud_explained_by_RM_.begin(),
                                                   full_cloud_explained_by_RM_.end(),
                                                   duplicityKernel(),
                                                   0,
                                                   thrust::plus<int>());

        /*int unexplained = thrust::transform_reduce(
                                              thrust::make_zip_iterator(thrust::make_tuple(unexplained_by_RM_neighboorhods.begin(), explained_by_RM_.begin())),
                                              thrust::make_zip_iterator(thrust::make_tuple(unexplained_by_RM_neighboorhods.end(), explained_by_RM_.end())),
                                              unexplainedKernel(),
                                              0.f, thrust::plus<int>());*/

        //TODO: free space still missing
        solution_so_far_ = initial_sol;
        float free_space = 0;
        std::cout << ginfo << " " << duplicity << " " /*<< unexplained << " "*/ << duplicityCM << std::endl;
        model_cost_ = (static_cast<float> (ginfo) -
            static_cast<float> (duplicity) - free_space -
            countActiveHypotheses (solution_so_far_) -
            static_cast<float> (duplicityCM) /*-
            unexplained * unexplained_weight_*/)
            * -1.f;

        std::cout << "Initial cost:" << model_cost_ << std::endl;

        //setPreviousUnExplainedValue(unexplained);
        setPreviousDuplicityCM (duplicityCM);
        setPreviousExplainedValue (ginfo);
        setPreviousDuplicity (duplicity);
        setPreviousBadInfo (free_space);

        //assign raw pointers
        raw_ptr_explained_ = thrust::raw_pointer_cast(explained_by_RM_.data());
        raw_ptr_explained_objectness_ = thrust::raw_pointer_cast(explained_by_RM_objectness_weighted.data());
        raw_ptr_duplicity_cm_ = thrust::raw_pointer_cast(full_cloud_explained_by_RM_.data());
        raw_ptr_unexplained_ = thrust::raw_pointer_cast(unexplained_by_RM_neighboorhods.data());
      }

      struct updateExplainedVectors
      {

        float * raw_ptr_explained_objectness_;
        int * raw_ptr_explained_;
        int sign_;
        updateExplainedVectors(float * ptr_eo, int * ptr_e, int sign)
            : raw_ptr_explained_objectness_(ptr_eo), raw_ptr_explained_(ptr_e), sign_(sign)
        { }

        template <typename Tup>
        __host__ __device__
        void operator()(Tup tp)
        {
          bool prev_dup = raw_ptr_explained_[thrust::get<0>(tp)] > 1;
          raw_ptr_explained_[thrust::get<0>(tp)] += sign_;
          raw_ptr_explained_objectness_[thrust::get<0>(tp)] += thrust::get<1>(tp) * sign_;

          int val_exp = raw_ptr_explained_[thrust::get<0>(tp)];
          if(val_exp == 1)
            thrust::get<2>(tp) = thrust::get<1>(tp);

           //hypotheses being removed, now the point is not explained anymore and was explained before
           if((sign_ < 0) && (val_exp == 0))
             thrust::get<2>(tp) = -thrust::get<1>(tp);

           //one hypohteses was added and now the point is not explained anymore
           if((sign_ > 0) && (val_exp == 2))
             thrust::get<2>(tp) = -thrust::get<1>(tp);

          if ((val_exp > 1) && prev_dup)
          { //its still a duplicate, so, just add or remove one
            thrust::get<3>(tp) = sign_;
            return;
          }
          else if ((val_exp == 1) && prev_dup)
          { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
            thrust::get<3>(tp) = -2;
            return;
          }
          else if ((val_exp > 1) && !prev_dup)
          { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
            thrust::get<3>(tp) = 2;
            return;
          }

          thrust::get<3>(tp) = 0;
        }

      };

      struct updateDuplicityCM
      {

        int * raw_ptr_explained_;
        int sign_;
        updateDuplicityCM(int * ptr_e, int sign)
            : raw_ptr_explained_(ptr_e), sign_(sign)
        { }

        template <typename Tup>
        __host__ __device__
        void operator()(Tup tp)
        {
          thrust::get<1>(tp) = 0;
          int val_exp = raw_ptr_explained_[thrust::get<0>(tp)];
          bool prev_dup = val_exp > 1;
          raw_ptr_explained_[thrust::get<0>(tp)] += sign_;
          val_exp += sign_;

           if ((val_exp > 1) && prev_dup)
           { //its still a duplicate, we are adding
             thrust::get<1>(tp) = sign_; //so, just add or remove one
           }
           else if ((val_exp == 1) && prev_dup)
           { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
             thrust::get<1>(tp) = -2;
           }
           else if ((val_exp > 1) && !prev_dup)
           { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
             thrust::get<1>(tp) = 2;
           }
        }
      };

      struct updateDuplicityCMTR
      {

        int * raw_ptr_explained_;
        int sign_;
        updateDuplicityCMTR(int * ptr_e, int sign)
            : raw_ptr_explained_(ptr_e), sign_(sign)
        { }

        __host__ __device__
        int operator()(int idx)
        {
          int val_exp = raw_ptr_explained_[idx];
          bool prev_dup = val_exp > 1;
          raw_ptr_explained_[idx] += sign_;
          val_exp += sign_;

           if ((val_exp > 1) && prev_dup)
           { //its still a duplicate, we are adding
             //thrust::get<1>(tp) = static_cast<int> (sign_); //so, just add or remove one
             return sign_;
           }
           else if ((val_exp == 1) && prev_dup)
           { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
             //thrust::get<1>(tp) = -2;
             return -2;
           }
           else if ((val_exp > 1) && !prev_dup)
           { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
             return 2;
             //thrust::get<1>(tp) = 2;
           }

           return 0;
        }
      };

      struct updateUnexplained
      {

        int * raw_ptr_unexplained_;
        int * raw_ptr_explained_;
        int sign_;
        updateUnexplained(int * ptr_ue, int * ptr_e, int sign)
            : raw_ptr_unexplained_(ptr_ue), raw_ptr_explained_(ptr_e), sign_(sign)
        { }

        __device__
        void operator()(thrust::tuple<int, int> tp)
        {
          thrust::get<1>(tp) = 0;
          int unexp_idx = thrust::get<0>(tp);
          bool prev_unexplained = (raw_ptr_unexplained_[unexp_idx] > 0) && (raw_ptr_explained_[unexp_idx] == 0);
          raw_ptr_unexplained_[unexp_idx] += sign_;

          if(sign_ < 0) {
            if (prev_unexplained)
            {
              //decrease by 1
              thrust::get<1>(tp) -= 1;
            }
          } else {
            if (raw_ptr_explained_[unexp_idx] == 0)
              thrust::get<1>(tp) += 1;
          }
        }

      };

      struct getUnexplained
      {

        int * raw_ptr_unexplained_;
        int * raw_ptr_explained_;
        int sign_;
        getUnexplained(int * ptr_ue, int * ptr_e, int sign)
            : raw_ptr_unexplained_(ptr_ue), raw_ptr_explained_(ptr_e), sign_(sign)
        { }

        __device__
        int operator()(int exp_idx)
        {
          if(sign_ < 0) {
            if ((raw_ptr_explained_[exp_idx] == 0) && (raw_ptr_unexplained_[exp_idx] > 0))
            {
              return raw_ptr_unexplained_[exp_idx]; //the points become unexplained
            }
          } else {
            if ((raw_ptr_explained_[exp_idx] == 1) && (raw_ptr_unexplained_[exp_idx] > 0)) {
              return raw_ptr_unexplained_[exp_idx] * (-1);
            }
          }

          return 0;
        }
      };

      float CudaBBoxOptimizer::evaluateSolution (const std::vector<bool> & active, int changed) {
        //std::cout << "CudaBBoxOptimizer::evaluateSolution called..." << changed << std::endl;
        float sign = 1.f;

        RecognitionModel * recog_model = &(recognition_models_[0]) + changed;
        if (!active[changed])
          sign = -1.f;

        //update explained_by_RM_, full_cloud_explained_by_RM_, explained_by_RM_objectness_weighted
        // and unexplained_by_RM_neighboorhods accordingly

        thrust::for_each(recog_model->start_exp_it_, recog_model->end_exp_it_,
            updateExplainedVectors(raw_ptr_explained_objectness_, raw_ptr_explained_, sign));

        thrust::for_each(
            recog_model->start_dup_cm_it_, recog_model->end_dup_cm_it_,
            updateDuplicityCM(raw_ptr_duplicity_cm_, sign));

        thrust::host_vector<float> add_to_explained_host = recog_model->add_to_explained_;
        thrust::host_vector<int> add_to_duplicity_host = recog_model->add_to_duplicity_;
        thrust::host_vector<int> add_to_duplicity_CM_host = recog_model->add_to_duplicity_CM_;

        //device reductions
        /*previous_explained_value += thrust::reduce ( recog_model->add_to_explained_.begin (),
                                                             recog_model->add_to_explained_.end (),
                                                             0.f, thrust::plus<float> ());

        previous_duplicity_ += thrust::reduce ( recog_model->add_to_duplicity_.begin (),
                                                recog_model->add_to_duplicity_.end (),
                                                0, thrust::plus<int> ());

        previous_duplicity_complete_models_ += thrust::reduce (recognition_models_[changed].add_to_duplicity_CM_.begin (),
                                                               recognition_models_[changed].add_to_duplicity_CM_.end (),
                                                               0, thrust::plus<int> ());*/

        //host reductions
        previous_explained_value += thrust::reduce ( add_to_explained_host.begin (),
                                                     add_to_explained_host.end (),
                                                     0.f, thrust::plus<float> ());

        previous_duplicity_ += thrust::reduce ( add_to_duplicity_host.begin (),
                                                add_to_duplicity_host.end (),
                                                0, thrust::plus<int> ());

        previous_duplicity_complete_models_ += thrust::reduce (add_to_duplicity_CM_host.begin (),
                                                               add_to_duplicity_CM_host.end (),
                                                               0, thrust::plus<int> ());

        /*previous_duplicity_complete_models_ += thrust::transform_reduce(recog_model->explained_fullcoud.begin(),
                                                                        recog_model->explained_fullcoud.end(),
                                                                        updateDuplicityCMTR(raw_ptr_duplicity_cm_, sign),
                                                                        0,
                                                                        thrust::plus<int> ());*/

        //update unexplained
        /*thrust::device_vector<int> add_to_unexplained(recognition_models_[changed].unexplained_in_neighborhood.size(), 0);
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(recognition_models_[changed].unexplained_in_neighborhood.begin(),
                                                         recognition_models_[changed].add_to_unexplained_in_neighborhood_.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(recognition_models_[changed].unexplained_in_neighborhood.end(),
                                                         recognition_models_[changed].add_to_unexplained_in_neighborhood_.end())),
                         updateUnexplained(raw_ptr_unexplained_, raw_ptr_explained_, sign));

        previous_unexplained_ += static_cast<float>(thrust::reduce (recognition_models_[changed].add_to_unexplained_in_neighborhood_.begin (), recognition_models_[changed].add_to_unexplained_in_neighborhood_.end (), 0, thrust::plus<int> ()));
        previous_unexplained_ += thrust::transform_reduce(
                         recognition_models_[changed].explained_.begin(),
                         recognition_models_[changed].explained_.end(),
                         getUnexplained(raw_ptr_unexplained_, raw_ptr_explained_, sign),
                         0, thrust::plus<int>());*/

        {
          /*float bad_info = static_cast<float> (getPreviousBadInfo ()) + static_cast<float> (recognition_models_[changed].free_space_) * bad_info_weight_
              * sign;*/

          previous_active_hypotheses_ += sign;

          float bad_info = 0.f;
          setPreviousBadInfo (bad_info);

          //cudaDeviceSynchronize ();

          //float duplicity_cm = static_cast<float> (previous_duplicity_complete_models_);
          return (static_cast<float> (previous_explained_value) -
              bad_info - static_cast<float> (previous_duplicity_) -
              previous_active_hypotheses_ * active_hyp_weight_ - previous_duplicity_complete_models_ /*- previous_unexplained_ * unexplained_weight_*/) * -1.f;
        }
      }

      //Requires a sorted sequence...
      struct copyUniqueElements {
        thrust::device_ptr<int> indices_;
        int size_;

        copyUniqueElements(thrust::device_ptr<int> indices, int size) : indices_(indices), size_(size) {

        }

        __device__
        bool
        operator() (const int j) {
          if (j == 0)
          {
            if (indices_[j] != indices_[j + 1])
            {
              return true;
            }
          }
          else if (j == (size_ - 1))
          {
            if (indices_[j] != indices_[j - 1])
            {
              return true;
            }
          }
          else
          {
            if (indices_[j] != indices_[j - 1] && indices_[j]
                != indices_[j + 1])
            {
              return true;
            }
          }

          return false;
        }
      };

      void
      CudaBBoxOptimizer::addModels (std::vector<BBox> & bbox_models, std::vector<float> & free_space) {
        std::cout << "Adding models in CudaBBoxOptimizer" << std::endl;

        int GRIDSIZE_X = (int)((max_x - min_x) / resolution);
        int GRIDSIZE_Y = (int)((max_y - min_y) / resolution);
        int GRIDSIZE_Z = (int)((max_z - min_z) / resolution);

        {
          //create full cloud
          thrust::host_vector<float> x_points, y_points, z_points;
          int n_points = GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z;
          x_points.resize (n_points);
          y_points.resize (n_points);
          z_points.resize (n_points);

          for (int xx = 0; xx < GRIDSIZE_X; xx++)
          {
            for (int yy = 0; yy < GRIDSIZE_Y; yy++)
            {
              for (int zz = 0; zz < GRIDSIZE_Z; zz++)
              {
                float x = min_x + xx * resolution + resolution / 2.f;
                float y = min_y + yy * resolution + resolution / 2.f;
                float z = min_z + zz * resolution + resolution / 2.f;
                int idx = zz * GRIDSIZE_X * GRIDSIZE_Y + yy * GRIDSIZE_X + xx;
                x_points[idx] = x;
                y_points[idx] = y;
                z_points[idx] = z;
              }
            }
          }

          x_points_full_cloud_ = x_points;
          y_points_full_cloud_ = y_points;
          z_points_full_cloud_ = z_points;
        }

        //go through the models and save information
        recognition_models_.resize(bbox_models.size());

        /*float max_objectness = 0.f;
        for (size_t i = 0; i < bbox_models.size (); i++)
        {
          if (max_objectness < bbox_models[i].score)
            max_objectness = bbox_models[i].score;
        }

        for (size_t i = 0; i < recognition_models_.size (); i++)
        {
          recognition_models_[i].box_.score /= max_objectness;
        }*/

        for (size_t i = 0; i < bbox_models.size (); i++)
        {
          BBox bb = bbox_models[i];
          recognition_models_[i].box_ = bbox_models[i];
          thrust::device_vector<int> inside_box;
          thrust::device_vector<int> inside_box_full;

          insideBBox(bbox_models[i], x_points_, y_points_, z_points_, inside_box);
          recognition_models_[i].explained_ = inside_box;

          insideBBox(bbox_models[i], x_points_full_cloud_, y_points_full_cloud_, z_points_full_cloud_, inside_box_full);
          recognition_models_[i].explained_fullcoud = inside_box_full;

          recognition_models_[i].explained_distances_.resize(inside_box.size());
          std::cout << recognition_models_[i].box_.score << std::endl;
          thrust::fill(recognition_models_[i].explained_distances_.begin(), recognition_models_[i].explained_distances_.end(), recognition_models_[i].box_.score);

          //extended bounding box
          BBox bb_extended;
          float expand_factor_ = 1.25f;
          bb_extended.sx = static_cast<int> (round (bb.sx * expand_factor_));
          bb_extended.sy = static_cast<int> (round (bb.sy * expand_factor_));
          bb_extended.sz = static_cast<int> (round (bb.sz * expand_factor_));

          bb_extended.x = bb.x - static_cast<int> (round ((bb_extended.sx - bb.sx) / 2.f));
          bb_extended.y = bb.y - static_cast<int> (round ((bb_extended.sy - bb.sy) / 2.f));
          bb_extended.z = bb.z - static_cast<int> (round ((bb_extended.sz - bb.sz) / 2.f));

          bb_extended.x = std::max (bb_extended.x, 1);
          bb_extended.y = std::max (bb_extended.y, 1);
          bb_extended.z = std::max (bb_extended.z, 1);

          bb_extended.sx = std::min (GRIDSIZE_X - 1, bb_extended.x + bb_extended.sx) - bb_extended.x;
          bb_extended.sy = std::min (GRIDSIZE_Y - 1, bb_extended.y + bb_extended.sy) - bb_extended.y;
          bb_extended.sz = std::min (GRIDSIZE_Z - 1, bb_extended.z + bb_extended.sz) - bb_extended.z;
          bb_extended.angle = bb.angle;

          thrust::device_vector<int> inside_extended_box;
          insideBBox (bb_extended, x_points_, y_points_, z_points_, inside_extended_box);

          thrust::device_vector<int> inside_extended_and_not_extended;
          inside_extended_and_not_extended.resize(inside_extended_box.size() + inside_box.size());
          thrust::copy(inside_box.begin(), inside_box.end(), inside_extended_and_not_extended.begin());
          thrust::copy(inside_extended_box.begin(), inside_extended_box.end(), inside_extended_and_not_extended.begin() + inside_box.size());
          thrust::sort(inside_extended_and_not_extended.begin(), inside_extended_and_not_extended.end());

          thrust::counting_iterator<int> idx_iterator (0);

          thrust::device_vector<int>::iterator it = thrust::copy_if(idx_iterator,
                                                                    idx_iterator + inside_extended_and_not_extended.size(),
                                                                    inside_extended_and_not_extended.begin(),
                                                                    copyUniqueElements(inside_extended_and_not_extended.data(),
                                                                                       inside_extended_and_not_extended.size()));

          inside_extended_and_not_extended.resize(it - inside_extended_and_not_extended.begin());
          recognition_models_[i].unexplained_in_neighborhood = inside_extended_and_not_extended;
          recognition_models_[i].free_space_ = free_space[i];
          //std::cout << inside_box.size() << " " << x_points_.size() << " " << inside_box_full.size() << " " << inside_extended_and_not_extended.size() << std::endl;

          recognition_models_[i].add_to_explained_.resize(recognition_models_[i].explained_.size(), 0.f);
          recognition_models_[i].add_to_duplicity_.resize(recognition_models_[i].explained_.size(), 0.f);
          recognition_models_[i].add_to_duplicity_CM_.resize(recognition_models_[i].explained_fullcoud.size(), 0);
          recognition_models_[i].add_to_unexplained_in_neighborhood_.resize(recognition_models_[i].unexplained_in_neighborhood.size(), 0);

          recognition_models_[i].start_exp_it_ = thrust::make_zip_iterator(
                                                        thrust::make_tuple(recognition_models_[i].explained_.begin(),
                                                                           recognition_models_[i].explained_distances_.begin(),
                                                                           recognition_models_[i].add_to_explained_.begin(),
                                                                           recognition_models_[i].add_to_duplicity_.begin()));

          recognition_models_[i].end_exp_it_ = thrust::make_zip_iterator(
                                                        thrust::make_tuple(recognition_models_[i].explained_.end(),
                                                                           recognition_models_[i].explained_distances_.end(),
                                                                           recognition_models_[i].add_to_explained_.end(),
                                                                           recognition_models_[i].add_to_duplicity_.end()));

          recognition_models_[i].start_dup_cm_it_ = thrust::make_zip_iterator(
              thrust::make_tuple(recognition_models_[i].explained_fullcoud.begin(),
                                 recognition_models_[i].add_to_duplicity_CM_.begin()));

          recognition_models_[i].end_dup_cm_it_ = thrust::make_zip_iterator(
                        thrust::make_tuple(recognition_models_[i].explained_fullcoud.begin(),
                                         recognition_models_[i].add_to_duplicity_CM_.begin()));
          //recognition_models_[i].add_to_unexplained_in_neighborhood_from_explained_indices_.resize(recognition_models_[i].explained_.size(), 0);
        }
      }

      struct Mat3f {
        float mat[3][3];
      };

      struct Vector3f {
        float x,y,z;
      };

      struct insideBoundingBox {
        Mat3f rot_;
        Vector3f trans_;
        Vector3f min_p, max_p;
        thrust::device_ptr<float> x_points_;
        thrust::device_ptr<float> y_points_;
        thrust::device_ptr<float> z_points_;
        insideBoundingBox(BBox bb, float min_x, float min_y, float min_z,
                            int angle_incr_, float resolution,
                            thrust::device_ptr<float> x_points,
                            thrust::device_ptr<float> y_points,
                            thrust::device_ptr<float> z_points) : x_points_(x_points),
                                                                   y_points_(y_points),
                                                                   z_points_(z_points) {

          min_p.x = min_x + bb.x * resolution;
          min_p.y = min_y + bb.y * resolution;
          min_p.z = min_z + bb.z * resolution;

          max_p.x = min_x + (bb.sx + bb.x) * resolution;
          max_p.y = min_y + (bb.sy + bb.y) * resolution;
          max_p.z = min_z + (bb.sz + bb.z) * resolution;

          for(size_t i=0; i < 3; i++) {
            for(size_t j=0; j < 3; j++) {
              if(i == j)
                rot_.mat[i][j] = 1.f;
              else
                rot_.mat[i][j] = 0.f;
            }
          }

          int v = bb.angle;
          if (v != 0)
          {
            double rot_rads = (static_cast<double> (angle_incr_ * v)) * 0.0174532925;
            rot_.mat[0][0] = cos(rot_rads);
            rot_.mat[1][1] = rot_.mat[0][0];
            rot_.mat[1][0] = sin(rot_rads);
            rot_.mat[0][1] = rot_.mat[1][0] * -1.f;
          }

          /*for(size_t i=0; i < 3; i++) {
            for(size_t j=0; j < 3; j++) {
              printf("%f ", rot_.mat[i][j]);
            }

            printf("\n");
          }*/
        }

        __device__
        bool
        operator() (const int idx)
        {
          //transform point into local crop coordinate system
          Vector3f p;
          p.x = *(x_points_ + idx);
          p.y = *(y_points_ + idx);
          p.z = *(z_points_ + idx);

          Vector3f tp;
          tp.x = tp.y = tp.z = 0;
          tp.x += rot_.mat[0][0] * p.x + rot_.mat[0][1] * p.y + rot_.mat[0][2] * p.z;
          tp.y += rot_.mat[1][0] * p.x + rot_.mat[1][1] * p.y + rot_.mat[1][2] * p.z;
          tp.z += rot_.mat[2][0] * p.x + rot_.mat[2][1] * p.y + rot_.mat[2][2] * p.z;

          //check its inside the bound, if yes return true
          //otherwise false
          if ( (tp.x < min_p.x || tp.y < min_p.y || tp.z < min_p.z)
             ||
               (tp.x > max_p.x || tp.y > max_p.y || tp.z > max_p.z))
          {
            return false;
          }

          return true;

        }
      };

      void CudaBBoxOptimizer::insideBBox(BBox & bb,
                                         thrust::device_vector<float> & xpoints,
                                         thrust::device_vector<float> & ypoints,
                                         thrust::device_vector<float> & zpoints,
                                         thrust::device_vector<int> & inside_box) {
        /*zip_iterator beg, end;
        beg = zip_cloud_begin();
        end = zip_cloud_end();*/

        //Mat3f rot;

        /*int v = bb.angle;
        if (v != 0)
        {
          float rot_rads = pcl::deg2rad (static_cast<float> (angle_incr_ * v));
          incr_rot_trans = Eigen::Affine3f (Eigen::AngleAxisf (static_cast<float> (rot_rads), Eigen::Vector3f::UnitZ ()));
        }*/

        //std::cout << "Angle icnr is:" << angle_incr_ << std::endl;
        inside_box.resize(xpoints.size());
        thrust::counting_iterator<int> idx_iterator (0);
        thrust::device_vector<int>::iterator it = thrust::copy_if(idx_iterator, idx_iterator + xpoints.size(),
                                                                  inside_box.begin(),
                                                                  insideBoundingBox(bb, min_x, min_y, min_z,
                                                                                    angle_incr_, resolution,
                                                                                    xpoints.data(), ypoints.data(), zpoints.data()));

        inside_box.resize(it - inside_box.begin());
      }

    }
  }
}
