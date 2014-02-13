/*
 * cuda_bbox_optimizer.h
 *
 *  Created on: Nov 15, 2012
 *      Author: aitor
 */

#ifndef CUDA_BBOX_OPTIMIZER_H_
#define CUDA_BBOX_OPTIMIZER_H_

#include "faat_pcl/segmentation/objectness_3d/objectness_common.h"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace faat_pcl
{
  namespace cuda
  {
    namespace segmentation
    {
        class CudaBBoxOptimizer
        {
          typedef thrust::device_vector<float>::iterator FloatIteratorThrust;
          typedef thrust::device_vector<int>::iterator IntIteratorThrust;
          typedef thrust::tuple<IntIteratorThrust, FloatIteratorThrust, FloatIteratorThrust, IntIteratorThrust> ExplainedIteratorTuple;
          typedef thrust::tuple<IntIteratorThrust, IntIteratorThrust> DupCMIteratorTuple;
          typedef thrust::zip_iterator<ExplainedIteratorTuple> ZipIterExpTuple;
          typedef thrust::zip_iterator<DupCMIteratorTuple> ZipIterDupCMTuple;

          struct RecognitionModel
          {
            BBox box_;
            thrust::device_vector<int> explained_;
            thrust::device_vector<int> explained_fullcoud;
            thrust::device_vector<float> explained_distances_;
            float free_space_;
            thrust::device_vector<int> unexplained_in_neighborhood;

            //for incremental updates...
            thrust::device_vector<float> add_to_explained_;
            thrust::device_vector<int> add_to_duplicity_;
            thrust::device_vector<int> add_to_duplicity_CM_;
            thrust::device_vector<int> add_to_unexplained_in_neighborhood_;

            ZipIterExpTuple start_exp_it_;
            ZipIterExpTuple end_exp_it_;
            ZipIterDupCMTuple start_dup_cm_it_;
            ZipIterDupCMTuple end_dup_cm_it_;
            //thrust::device_vector<int> add_to_unexplained_in_neighborhood_from_explained_indices_;
          };

          thrust::device_vector<int> explained_by_RM_;
          thrust::device_vector<int> full_cloud_explained_by_RM_;
          thrust::device_vector<float> explained_by_RM_objectness_weighted;
          thrust::device_vector<int> unexplained_by_RM_neighboorhods;
          std::vector<RecognitionModel> recognition_models_;
          std::vector<bool> mask_;

          float min_z;
          float max_z;
          float min_x;
          float max_x;
          float min_y;
          float max_y;

          float resolution;
          int angle_incr_;

          float previous_explained_value;
          int previous_duplicity_;
          int previous_bad_info_;
          float previous_objectness_;
          int previous_duplicity_complete_models_;
          float previous_unexplained_;
          int previous_active_hypotheses_;

          float bad_info_weight_;
          float unexplained_weight_;
          float active_hyp_weight_;

          int * raw_ptr_explained_; //= thrust::raw_pointer_cast(explained_by_RM_.data());
          float * raw_ptr_explained_objectness_; // = thrust::raw_pointer_cast(explained_by_RM_objectness_weighted.data());
          int * raw_ptr_duplicity_cm_; // = thrust::raw_pointer_cast(full_cloud_explained_by_RM_.data());
          int * raw_ptr_unexplained_; //= thrust::raw_pointer_cast(unexplained_by_RM_neighboorhods.data());

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
          setPreviousExplainedValue (float v)
          {
            previous_explained_value = v;
          }

          void
          setPreviousDuplicity (int v)
          {
            previous_duplicity_ = v;
          }

          float
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

          float model_cost_;
          std::vector<bool> solution_so_far_;

          thrust::device_vector<float> x_points_, y_points_, z_points_;
          thrust::device_vector<float> x_points_full_cloud_, y_points_full_cloud_, z_points_full_cloud_;

          typedef thrust::tuple<float, float, float> tuple_type;
          typedef thrust::device_vector<float>::iterator float_iterator;
          typedef thrust::tuple<float_iterator, float_iterator, float_iterator> iterator_tuple;
          typedef thrust::zip_iterator<iterator_tuple> zip_iterator;

          zip_iterator
          zip_cloud_begin ()
          {
            return (thrust::make_zip_iterator (thrust::make_tuple (x_points_.begin (),
                                                                   y_points_.begin (),
                                                                   z_points_.begin ())));
          }

          zip_iterator
          zip_cloud_end ()
          {
            return (thrust::make_zip_iterator (thrust::make_tuple (x_points_.end (),
                                                                   y_points_.end (),
                                                                   z_points_.end ())));
          }

          zip_iterator
          zip_full_cloud_begin ()
          {
            return (thrust::make_zip_iterator (thrust::make_tuple (x_points_full_cloud_.begin (),
                                                                   y_points_full_cloud_.begin (),
                                                                   z_points_full_cloud_.begin ())));
          }

          zip_iterator
          zip_full_cloud_end ()
          {
            return (thrust::make_zip_iterator (thrust::make_tuple (x_points_full_cloud_.end (),
                                                                   y_points_full_cloud_.end (),
                                                                   z_points_full_cloud_.end ())));
          }

          void insideBBox(BBox & bb, thrust::device_vector<float> & xpoints,
                          thrust::device_vector<float> & ypoints,
                          thrust::device_vector<float> & zpoints,
                          thrust::device_vector<int> & inside_box);

          float countActiveHypotheses(const std::vector<bool> & sol) {
            int c = 0;
            for (size_t i = 0; i < sol.size (); i++)
            {
              if (sol[i])
                c++;
            }

            return static_cast<float> (c) * 5.f;
          }

        public:
          CudaBBoxOptimizer (float w = 2.f)
          {
            bad_info_weight_ = w;
            unexplained_weight_ = 0.f;
            active_hyp_weight_ = 5.f;
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
          setAngleIncr(int incr) {
            angle_incr_ = incr;
            std::cout << "Setting angle incr:" << angle_incr_ << " " << incr << std::endl;
          }

          void
          setCloud(thrust::host_vector<float> x_points, thrust::host_vector<float> y_points, thrust::host_vector<float> z_points) {
            x_points_ = x_points;
            y_points_ = y_points;
            z_points_ = z_points;
            std::cout << "Setting device cloud" << std::endl;
          }


          void
          addModels (std::vector<BBox> & bbox_models, std::vector<float> & free_space);

          void
          initializeOptimization(std::vector<bool> & initial_sol);

          float evaluateSolution (const std::vector<bool> & active, int changed);

          float getCost() {
            return model_cost_;
          }

        };
    }
  }
}

#endif /* CUDA_BBOX_OPTIMIZER_H_ */
