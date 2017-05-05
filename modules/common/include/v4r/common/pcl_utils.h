/******************************************************************************
 * Copyright (c) 2015 Thomas Faeulhammer
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

#pragma once

#include <boost/dynamic_bitset.hpp>
#include <pcl/common/common.h>
#include <v4r/core/macros.h>


namespace v4r
{
/**
 * @brief sets the sensor origin and sensor orientation fields of the PCL pointcloud header by the given transform
 */
template<typename PointT>
V4R_EXPORTS void
setCloudPose(const Eigen::Matrix4f &trans, pcl::PointCloud<PointT> &cloud);

}

namespace pcl   /// NOTE: THIS NAMESPACE IS AN EXCEPTION
{

/** \brief Extract the indices of a given point cloud as a new point cloud (instead of int types, this function uses a size_t vector)
  * \param[in] cloud_in the input point cloud dataset
  * \param[in] indices the vector of indices representing the points to be copied from \a cloud_in
  * \param[out] cloud_out the resultant output point cloud dataset
  * \note Assumes unique indices.
  * \ingroup common
  */
template <typename PointT> V4R_EXPORTS void
copyPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                const std::vector<size_t> &indices,
                pcl::PointCloud<PointT> &cloud_out);

/** \brief Extract the indices of a given point cloud as a new point cloud (instead of int types, this function uses a size_t vector)
  * \param[in] cloud_in the input point cloud dataset
  * \param[in] indices the vector of indices representing the points to be copied from \a cloud_in
  * \param[out] cloud_out the resultant output point cloud dataset
  * \note Assumes unique indices.
  * \ingroup common
  */
template <typename PointT> V4R_EXPORTS void
copyPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                const std::vector<size_t, Eigen::aligned_allocator<size_t> > &indices,
                pcl::PointCloud<PointT> &cloud_out);

template <typename PointT> V4R_EXPORTS void
copyPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                     const std::vector<bool> &mask,
                     pcl::PointCloud<PointT> &cloud_out);

template <typename PointT> V4R_EXPORTS void
copyPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                     const boost::dynamic_bitset<> &mask,
                     pcl::PointCloud<PointT> &cloud_out);
}
