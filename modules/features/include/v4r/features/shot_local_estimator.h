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

#ifndef V4R_SHOT_LOCAL_ESTIMATOR_H_
#define V4R_SHOT_LOCAL_ESTIMATOR_H_

#include <pcl/io/pcd_io.h>
#include <v4r/features/local_estimator.h>
#include <v4r/features/types.h>

namespace v4r
{
    template<typename PointT>
      class V4R_EXPORTS SHOTLocalEstimation : public LocalEstimator<PointT>
      {
          using LocalEstimator<PointT>::indices_;
          using LocalEstimator<PointT>::cloud_;
          using LocalEstimator<PointT>::normals_;
          using LocalEstimator<PointT>::processed_;
          using LocalEstimator<PointT>::keypoint_indices_;
          using LocalEstimator<PointT>::keypoints_;
          using LocalEstimator<PointT>::descr_name_;
          using LocalEstimator<PointT>::descr_type_;
          using LocalEstimator<PointT>::descr_dims_;
          float support_radius_;

      public:
        SHOTLocalEstimation ( float support_radius = 0.04f ): support_radius_( support_radius )
        {
            descr_name_ = "shot";
            descr_type_ = FeatureType::SHOT;
            descr_dims_ = 352;
        }

        void
        setSupportRadius(float radius)
        {
            support_radius_ = radius;
        }

        bool
        acceptsIndices() const
        {
          return true;
        }

        void
        compute(std::vector<std::vector<float> > & signatures);

        bool
        needNormals () const
        {
            return true;
        }

        typedef boost::shared_ptr< SHOTLocalEstimation<PointT> > Ptr;
        typedef boost::shared_ptr< SHOTLocalEstimation<PointT> const> ConstPtr;
      };
}

#endif
