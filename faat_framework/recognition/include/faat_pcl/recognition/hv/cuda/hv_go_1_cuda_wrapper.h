/*
 * hv_go_1_cuda_wrapper.h
 *
 *  Created on: Feb 27, 2013
 *      Author: aitor
 */

#ifndef HV_GO_1_CUDA_WRAPPER_H_
#define HV_GO_1_CUDA_WRAPPER_H_

#include "hv_go_1_cuda.h"
#include "pcl/common/common.h"

namespace faat_pcl
{
  namespace cuda
  {
    namespace recognition
    {
        template<typename PointT>
        class HVGOCudaWrapper
        {
          private:
            HVGo1Cuda * cuda_go_;
          public:
            HVGOCudaWrapper()
            {
              cuda_go_ = new HVGo1Cuda();
            }

            void setSceneCloud(typename pcl::PointCloud<PointT>::Ptr & cloud)
            {

            }

            void setSmoothClusters(pcl::PointCloud<pcl::PointXYZI>::Ptr & clusters_cloud)
            {

            }
        };
    }
  }
}

#endif /* HV_GO_1_CUDA_WRAPPER_H_ */
