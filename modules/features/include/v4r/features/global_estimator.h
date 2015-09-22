/*
 * estimator.h
 *
 *  Created on: Mar 22, 2012
 *      Author: aitor
 */

#ifndef FAAT_PCL_REC_FRAMEWORK_GLOBAL_ESTIMATOR_H_
#define FAAT_PCL_REC_FRAMEWORK_GLOBAL_ESTIMATOR_H_

#include <v4r/core/macros.h>
#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/normal_estimator.h>

namespace v4r
{
    template <typename PointInT, typename FeatureT>
    class V4R_EXPORTS GlobalEstimator {
      protected:
        bool computed_normals_;
        typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
        typedef typename pcl::PointCloud<FeatureT>::Ptr FeatureTPtr;

        typename boost::shared_ptr<PreProcessorAndNormalEstimator<PointInT, pcl::Normal> > normal_estimator_;

        pcl::PointCloud<pcl::Normal>::Ptr normals_;

      public:
        virtual bool
        estimate (const PointInTPtr & in, PointInTPtr & processed, std::vector<pcl::PointCloud<FeatureT>, Eigen::aligned_allocator<
            pcl::PointCloud<FeatureT> > > & signatures, std::vector<Eigen::Vector3f> & centroids)=0;

        virtual bool computedNormals() = 0;

        void setNormalEstimator(boost::shared_ptr<PreProcessorAndNormalEstimator<PointInT, pcl::Normal> > & ne) {
          normal_estimator_ = ne;
        }

        void getNormals(pcl::PointCloud<pcl::Normal>::Ptr & normals) {
          normals = normals_;
        }

        virtual bool acceptsIndices()
        {
            return false;
        }

        virtual void
        setIndices(pcl::PointIndices & p_indices)
        {
            (void) p_indices;
            PCL_ERROR("This function is not implemented!");
        }

        virtual void
        setIndices(std::vector<int> & p_indices)
        {
            (void) p_indices;
            PCL_ERROR("This function is not implemented!");
        }

    };
}


#endif /* REC_FRAMEWORK_ESTIMATOR_H_ */
