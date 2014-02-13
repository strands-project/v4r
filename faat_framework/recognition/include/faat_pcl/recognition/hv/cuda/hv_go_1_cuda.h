/*
 * hv_go_1_cuda.h
 *
 *  Created on: Feb 27, 2013
 *      Author: aitor
 */

#ifndef HV_GO_1_CUDA_H_
#define HV_GO_1_CUDA_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

struct Vector3f
{
  float x, y, z;
  __host__  __device__
  Vector3f (float _x, float _y, float _z)
  {
    x = _x;
    y = _y;
    z = _z;
  }
};

namespace faat_pcl
{
  namespace cuda
  {
    namespace recognition
    {
      class HVGo1Cuda
      {
      public:
        HVGo1Cuda ()
        {

        }
      };
    }
  }
}

#endif /* HV_GO_1_CUDA_H_ */
