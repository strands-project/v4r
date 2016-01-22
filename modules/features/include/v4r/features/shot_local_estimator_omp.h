/*
 * shot_local_estimator.h
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */

#ifndef V4R_SHOT_LOCAL_ESTIMATOR_OMP_H_
#define V4R_SHOT_LOCAL_ESTIMATOR_OMP_H_

#include "local_estimator.h"
#include <pcl/features/shot_omp.h>
#include <pcl/io/pcd_io.h>
#include <v4r/features/types.h>
#include <v4r/common/normals.h>

namespace v4r
{
    template<typename PointT>
      class V4R_EXPORTS SHOTLocalEstimationOMP : public LocalEstimator<PointT>
      {
        typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;
        typedef pcl::PointCloud<pcl::PointXYZ> KeypointCloud;

        using LocalEstimator<PointT>::keypoint_extractor_;
        using LocalEstimator<PointT>::normals_;
        using LocalEstimator<PointT>::keypoint_indices_;

        std::vector<int> indices_;

      public:
        class Parameter : public LocalEstimator<PointT>::Parameter
        {
        public:
            using LocalEstimator<PointT>::Parameter::adaptative_MLS_;
            using LocalEstimator<PointT>::Parameter::normal_computation_method_;
            using LocalEstimator<PointT>::Parameter::support_radius_;

            Parameter():LocalEstimator<PointT>::Parameter()
            {}
        }param_;

        SHOTLocalEstimationOMP (const Parameter &p = Parameter()) : LocalEstimator<PointT>(p)
        {
            this->descr_name_ = "shot_omp";
            param_ = p;
        }

        size_t getFeatureType() const
        {
            return SHOT;
        }

        void
        setIndices(const std::vector<int> & indices)
        {
          indices_ = indices;
        }

        bool acceptsIndices() const
        {
          return true;
        }

        bool
        estimate (const pcl::PointCloud<PointT> & in, pcl::PointCloud<PointT> & processed, pcl::PointCloud<PointT> & keypoints, std::vector<std::vector<float> > & signatures);

        bool
        needNormals ()
        {
          return true;
        }
      };
}

#endif /* REC_FRAMEWORK_SHOT_LOCAL_ESTIMATOR_OMP_H_ */
