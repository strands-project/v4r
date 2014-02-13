/*
 * visibility_reasoning.h
 *
 *  Created on: Mar 19, 2013
 *      Author: aitor
 */

#ifndef VISIBILITY_REASONING_H_
#define VISIBILITY_REASONING_H_

#include "pcl/common/common.h"

namespace faat_pcl
{
  namespace object_modelling
  {
    template<typename PointT>
    class VisibilityReasoning
    {
      typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
      float focal_length_; float cx_; float cy_;

      public:
        VisibilityReasoning(float fc, float cx, float cy)
        {
          focal_length_ = fc;
          cx_ = cx;
          cy_ = cy;
        }

        void
        computeRangeDifferencesWhereObserved(PointCloudPtr & im1, PointCloudPtr & im2, std::vector<float> & range_diff);

        float computeFSV(PointCloudPtr & im1,
                           PointCloudPtr & im2,
                           Eigen::Matrix4f pose_2_to_1 = Eigen::Matrix4f::Identity());

        float computeFocalLength(int cx, int cy, PointCloudPtr & cloud);

        void computeRangeImage(int cx, int cy, float fl, PointCloudPtr & cloud, PointCloudPtr & range_image);
    };
  }
}

#endif /* VISIBILITY_REASONING_H_ */
