/******************************************************************************
 * Copyright (c) 2016 Thomas Faeulhammer
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

#ifndef V4R_OCCLUSION_REASONING_H_
#define V4R_OCCLUSION_REASONING_H_

#include <boost/dynamic_bitset.hpp>
#include <pcl/point_cloud.h>
#include <v4r/common/camera.h>


namespace v4r
{
/**
 * @brief reason about occlusion
 * @param organized_cloud point cloud that causes occlusion
 * @param to_be_filtered point cloud to be checked for occlusion
 * @param camera parameters for re-projection
 * @param occlusion threshold in meter
 * @param true if points projected outsided the field of view should be also considered as occluded
 * @return bitmask indicating which points of to_be_filtered are occluded (bits set to true)
 */
template<typename PointTA, typename PointTB>
V4R_EXPORTS
boost::dynamic_bitset<>
occlusion_reasoning (const pcl::PointCloud<PointTA> & organized_cloud,
                     const pcl::PointCloud<PointTB> & to_be_filtered,
                     const Camera::Ptr cam = Camera(),
                     float threshold = 0.01f,
                     bool is_occluded_out_fov = true);
}

#endif
