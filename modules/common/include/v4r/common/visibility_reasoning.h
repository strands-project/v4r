/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma, Thomas Faeulhammer
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


#ifndef V4R_REGISTRATION_VISIBILITY_REASONING_H_
#define V4R_REGISTRATION_VISIBILITY_REASONING_H_

#include <pcl/common/common.h>
#include <v4r/core/macros.h>

namespace v4r
{

/**
 * visibility_reasoning.h
 *
 *  @date Mar 19, 2013
 *  @author Aitor Aldoma, Thomas Faeulhammer
 */
    template<typename PointT>
    class V4R_EXPORTS VisibilityReasoning
    {
      typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
      float focal_length_; float cx_; float cy_;
      float tss_;
      int fsv_used_;
      public:
        VisibilityReasoning(float fc, float cx, float cy)
        {
          focal_length_ = fc;
          cx_ = cx;
          cy_ = cy;
          tss_ = 0.01f;
          fsv_used_ = 0;
        }

        int getFSVUsedPoints()
        {
          return fsv_used_;
        }

        int
        computeRangeDifferencesWhereObserved(const typename pcl::PointCloud<PointT>::ConstPtr & im1, const typename pcl::PointCloud<PointT>::ConstPtr & im2, std::vector<float> & range_diff);

        int
        computeRangeDifferencesWhereObservedWithIndicesBack(const typename pcl::PointCloud<PointT>::ConstPtr & im1, const typename pcl::PointCloud<PointT>::ConstPtr & im2, std::vector<float> & range_diff, std::vector<int> & indices);

        float computeFSV(const typename pcl::PointCloud<PointT>::ConstPtr &im1,
                           const typename pcl::PointCloud<PointT>::ConstPtr &im2,
                           Eigen::Matrix4f pose_2_to_1 = Eigen::Matrix4f::Identity());

        float computeFSVWithNormals(const typename pcl::PointCloud<PointT>::ConstPtr &im1,
                                    const typename pcl::PointCloud<PointT>::ConstPtr &im2,
                                    pcl::PointCloud<pcl::Normal>::Ptr & normals);

        float computeOSV(const typename pcl::PointCloud<PointT>::ConstPtr &im1,
                           const typename pcl::PointCloud<PointT>::ConstPtr &im2,
                           Eigen::Matrix4f pose_2_to_1 = Eigen::Matrix4f::Identity());

        void setThresholdTSS(float t)
        {
          tss_ = t;
        }

        float computeFocalLength(int width, int height, const typename pcl::PointCloud<PointT>::ConstPtr & cloud);

        static void computeRangeImage(int width, int height, float fl, const typename pcl::PointCloud<PointT>::ConstPtr & cloud, typename pcl::PointCloud<PointT>::Ptr & range_image);
    };
}
#endif
