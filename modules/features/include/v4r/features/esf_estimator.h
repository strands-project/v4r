/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma, Thomas Faeulhammer
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

#ifndef REC_FRAMEWORK_ESF_ESTIMATOR_H_
#define REC_FRAMEWORK_ESF_ESTIMATOR_H_

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/core/macros.h>
#include <v4r/features/global_estimator.h>

#include <pcl/features/esf.h>
#include <glog/logging.h>

namespace v4r
{
    template<typename PointT>
      class V4R_EXPORTS ESFEstimation : public GlobalEstimator<PointT>
      {
      private:
          using GlobalEstimator<PointT>::indices_;
          using GlobalEstimator<PointT>::input_cloud_;

          typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;
          PointInTPtr processed_;

      public:
          std::vector<float>
          estimate ()
          {
            std::vector<float> signature;

            CHECK(input_cloud_ && !input_cloud_->points.empty());

            if(!indices_.empty()) {
                processed_.reset(new pcl::PointCloud<PointT>);
                pcl::copyPointCloud(*input_cloud_, indices_, *processed_);
            }
            else
                processed_ = input_cloud_;


            typedef typename pcl::ESFEstimation<PointT, pcl::ESFSignature640> ESFEstimation;
            pcl::PointCloud<pcl::ESFSignature640> ESF_signature;
            ESFEstimation esf;
            esf.setInputCloud (processed_);
            esf.compute (ESF_signature);

            const pcl::ESFSignature640 &pt = ESF_signature.points[0];
            const size_t feat_dim = (size_t) ((double)(sizeof(pt.histogram)) / sizeof(pt.histogram[0]));
            signature.resize(feat_dim);

            for(size_t i=0; i<feat_dim; i++)
                signature[i] = pt.histogram[i];

            indices_.clear();

            return signature;
          }
      };
}

#endif
