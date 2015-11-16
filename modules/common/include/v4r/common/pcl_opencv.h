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


#include <pcl/common/common.h>
#include <opencv2/opencv.hpp>
#include <v4r/core/macros.h>

#ifndef V4R_PCL_OPENCV_H_
#define V4R_PCL_OPENCV_H_

namespace v4r
{
  template<class PointT>
  V4R_EXPORTS void
  ConvertPCLCloud2Image (const typename pcl::PointCloud<PointT>::Ptr &pcl_cloud, cv::Mat_<cv::Vec3b> &image, bool crop = false);

  template<class PointT>
  V4R_EXPORTS void
  ConvertPCLCloud2DepthImage (const typename pcl::PointCloud<PointT>::Ptr &pcl_cloud, cv::Mat_<float> &image);

  template<class PointT>
  V4R_EXPORTS void
  ConvertUnorganizedPCLCloud2Image (const typename pcl::PointCloud<PointT>::Ptr &pcl_cloud,
                                    cv::Mat_<cv::Vec3b> &image,
                                    bool crop = false,
                                    float bg_r = 255.0f,
                                    float bg_g = 255.0f,
                                    float bg_b = 255.0f,
                                    int width = 640,
                                    int height = 480,
                                    float f = 525.5f,
                                    float cx = 319.5f,
                                    float cy = 239.5f);


  template<typename PointT>
  V4R_EXPORTS cv::Mat
  ConvertPCLCloud2Image(const typename pcl::PointCloud<PointT> &cloud,
                        const std::vector<int> &cluster_idx,
                        size_t out_height, size_t out_width);


 /**
   * @brief transforms an RGB-D point cloud into an image with fixed size
   * @param RGB-D cloud
   * @param indices of the points belonging to the object
   * @param out_height
   * @param out_width
   * @return image
   */
  template<typename PointT>
  V4R_EXPORTS
  cv::Mat
  pcl2imageFixedSize(const pcl::PointCloud<PointT> &cloud, const std::vector<int> &cluster_idx, size_t out_height, size_t out_width);
}

#endif /* PCL_OPENCV_H_ */
