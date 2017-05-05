/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma
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

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <v4r/core/macros.h>

namespace v4r
{

enum NormalEstimatorType
{
    PCL_DEFAULT = 0x01, // 00000001
    PCL_INTEGRAL_NORMAL = 0x02, // 00000010
    Z_ADAPTIVE  = 0x04, // 00000100
    PRE_PROCESS = 0x08 // 00001000
};


template<typename PointT>
class V4R_EXPORTS NormalEstimator
{
protected:
    typename pcl::PointCloud<PointT>::ConstPtr input_; ///< input cloud
    pcl::PointCloud<pcl::Normal>::Ptr normal_; ///< computed surface normals for input cloud
    std::vector<int> indices_;  ///< indices of the segmented object (extracted keypoints outside of this will be neglected)

public:
    virtual ~NormalEstimator(){ }

    /**
     * @brief setInputCloud
     * @param input input cloud
     */
    void
    setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr & input)
    {
        input_ = input;
    }

    /**
     * @brief setIndices
     * @param indices indices of the segmented object (extracted keypoints outside of this will be neglected)
     */
    void
    setIndices(const std::vector<int> &indices)
    {
        indices_ = indices;
    }

    /**
     * @brief getNormalEstimatorType
     * @return unique type id of normal estimator(as stated in keypoint/types.h)
     */
    virtual int getNormalEstimatorType() const = 0;

    /**
     * @brief compute
     */
    virtual
    pcl::PointCloud<pcl::Normal>::Ptr
    compute () = 0;

    typedef boost::shared_ptr< NormalEstimator<PointT> > Ptr;
    typedef boost::shared_ptr< NormalEstimator<PointT> const> ConstPtr;
};

}
