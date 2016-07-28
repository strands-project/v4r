/******************************************************************************
 * Copyright (c) 2015 Aitor Aldoma, Thomas Faeulhammer
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

#ifndef V4R_REGISTRATION_METRICS_H__
#define V4R_REGISTRATION_METRICS_H__

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <v4r_config.h>
#include <v4r/core/macros.h>

namespace v4r
{

/**
 * @brief This method computes a cost function for the pairwise alignment of two point clouds.
 * It is computed using fast ICP
 * @param[in] cloud_src source cloud
 * @param[in] cloud_dst target cloud
 * @param[in] transform homogenous transformation matrix aligning the two point clouds
 * @param[out] registration cost ( the lower the better the alignment - weight range [0, 0.75] )
 * @param[out] refined_transform refined homogenous transformation matrix aligning the two point clouds based on ICP
 */
template<typename PointT> V4R_EXPORTS void
calcEdgeWeightAndRefineTf (const typename pcl::PointCloud<PointT>::ConstPtr &cloud_src,
                           const typename pcl::PointCloud<PointT>::ConstPtr &cloud_dst,
                           const Eigen::Matrix4f &transform,
                           float &registration_quality,
                           Eigen::Matrix4f &refined_transform);


/**
 * @brief This method computes a cost function for the pairwise alignment of two point clouds.
 * It is computed using fast ICP
 * @param[in] cloud_src source cloud
 * @param[in] cloud_dst target cloud (already aligned to source, i.e. transformation equal identity matrix)
 * @param[out] registration cost ( the lower the better the alignment - weight range [0, 0.75] )
 * @param[out] refined_transform refined homogenous transformation matrix aligning the two point clouds based on ICP
 */
template<typename PointT> V4R_EXPORTS void
calcEdgeWeightAndRefineTf  (const typename pcl::PointCloud<PointT>::ConstPtr &cloud_src,
                            const typename pcl::PointCloud<PointT>::ConstPtr &cloud_dst,
                            float &registration_quality,
                            Eigen::Matrix4f &refined_transform)
{
     calcEdgeWeightAndRefineTf(cloud_src, cloud_dst, Eigen::Matrix4f::Identity(), registration_quality, refined_transform);
}

}
#endif
