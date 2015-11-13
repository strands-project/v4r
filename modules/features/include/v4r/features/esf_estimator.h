/*
 * vfh_estimator.h
 *
 *  Created on: Mar 22, 2012
 *      Author: aitor
 */

#ifndef REC_FRAMEWORK_ESF_ESTIMATOR_H_
#define REC_FRAMEWORK_ESF_ESTIMATOR_H_

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/core/macros.h>
#include "global_estimator.h"
#include <pcl/features/esf.h>
#include <glog/logging.h>

namespace v4r
{
    template<typename PointInT>
      class V4R_EXPORTS ESFEstimation : public GlobalEstimator<PointInT>
      {

        typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;

      public:
//          bool
//          estimate (const PointInTPtr & in, PointInTPtr & processed,
//                    typename pcl::PointCloud<FeatureT>::CloudVectorType & signatures,
//                    std::vector<Eigen::Vector3f> & centroids)
//          {

//            if(!in)
//            {
//                PCL_ERROR("ESFEstimation, input is empty!");
//                return false;
//            }

//            if(in->points.size() == 0)
//            {
//                PCL_ERROR("ESFEstimation, input has no points!");
//                return false;
//            }

//            typedef typename pcl::ESFEstimation<PointInT, FeatureT> ESFEstimation;
//            pcl::PointCloud<FeatureT> ESF_signature;

//            ESFEstimation esf;
//            esf.setInputCloud (in);
//            esf.compute (ESF_signature);

//            signatures.resize (1);
//            centroids.resize (1);

//            signatures[0] = ESF_signature;

//            Eigen::Vector4f centroid4f;
//            pcl::compute3DCentroid (*in, centroid4f);
//            centroids[0] = Eigen::Vector3f (centroid4f[0], centroid4f[1], centroid4f[2]);

//            pcl::copyPointCloud(*in, *processed);

//            return true;
//          }

          bool
          estimate (const PointInTPtr & in,
                    std::vector<float> & signature)
          {
            CHECK(in && !in->points.empty());
            typedef typename pcl::ESFEstimation<PointInT, pcl::ESFSignature640> ESFEstimation;
            pcl::PointCloud<pcl::ESFSignature640> ESF_signature;
            ESFEstimation esf;
            esf.setInputCloud (in);
            esf.compute (ESF_signature);

            const pcl::ESFSignature640 &pt = ESF_signature.points[0];
            const size_t feat_dim = (size_t)pt.descriptorSize();

            for(size_t i=0; i<feat_dim; i++)
                signature[i] = ESF_signature.points[0].histogram[i];

            return true;
          }

        bool
        computedNormals ()
        {
          return false;
        }
      };
}

#endif /* REC_FRAMEWORK_ESF_ESTIMATOR_H_ */
