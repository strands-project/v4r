/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma
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

#ifndef V4R_OPENCV_SIFT_ESTIMATOR_H_
#define V4R_OPENCV_SIFT_ESTIMATOR_H_

#include "local_estimator.h"
#include <v4r/features/types.h>
#include <pcl/io/pcd_io.h>
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <v4r/common/pcl_opencv.h>

namespace v4r
{
    template<typename PointT>
      class V4R_EXPORTS OpenCVSIFTLocalEstimation : public LocalEstimator<PointT>
      {
        typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;

        using LocalEstimator<PointT>::keypoint_extractor_;
        using LocalEstimator<PointT>::keypoint_indices_;

        std::vector<int> indices_;
        boost::shared_ptr<cv::SIFT> sift_;

      public:
        OpenCVSIFTLocalEstimation ()
        {
          this->descr_name_ = "sift_opencv";
          double threshold = 0.03;
          double edge_threshold = 10.0;
          sift_.reset(new cv::SIFT(0, 3, threshold, edge_threshold));
        }

        bool
        estimate (const pcl::PointCloud<PointT> & in, pcl::PointCloud<PointT> & processed, pcl::PointCloud<PointT> & keypoints, std::vector<std::vector<float> > & signatures);

        void
        setIndices(const std::vector<int> & indices)
        {
          indices_ = indices;
        }

        bool acceptsIndices() const
        {
            return true;
        }

        size_t getFeatureType() const
        {
            return SIFT;
        }

      };
}

#endif /* REC_FRAMEWORK_SHOT_LOCAL_ESTIMATOR_H_ */
