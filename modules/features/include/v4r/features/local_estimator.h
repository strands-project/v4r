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

#pragma once

#include <v4r/common/normal_estimator.h>
#include <v4r/core/macros.h>
#include <vector>

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS LocalEstimator
{
protected:
    typename pcl::PointCloud<PointT>::ConstPtr cloud_;  ///< input cloud
    pcl::PointCloud<pcl::Normal>::ConstPtr normals_;    ///< input normals
    std::vector<int> keypoint_indices_; ///< extracted keypoint indices

    std::vector<int> indices_;
    std::string descr_name_;
    size_t descr_type_;
    size_t descr_dims_;

public:
    virtual ~LocalEstimator(){}

    size_t
    getFeatureType() const
    {
        return descr_type_;
    }

    virtual bool
    acceptsIndices() const = 0;


    virtual bool
    needNormals () const = 0;

    /**
     * @brief set indices of the object (segmented cluster). Points not within this indices will be ignored.
     * @param indices
     */
    void
    setIndices (const std::vector<int> & indices)
    {
        indices_ = indices;
    }

    /**
     * @brief sets the normals point cloud
     * @param normals
     */
    void
    setNormals(const pcl::PointCloud<pcl::Normal>::ConstPtr & normals)
    {
        normals_ = normals;
    }

    /**
     * @brief sets the input point cloud
     * @param normals
     */
    void
    setInputCloud(const typename pcl::PointCloud<PointT>::ConstPtr &cloud)
    {
        cloud_ = cloud;
    }

    std::vector<int>
    getKeypointIndices() const
    {
        return keypoint_indices_;
    }

    std::string
    getFeatureDescriptorName() const
    {
        return descr_name_;
    }

    size_t
    getFeatureDimensions() const
    {
        return descr_dims_;
    }

    /**
     * @brief getUniqueId in case several local feature estimators of the same type are used (e.g. shot with different support radii), we need to find unique ids
     * @return a unique identity for the local estimator that takes into account its parameters.
     */
    virtual
    std::string
    getUniqueId() const
    {
        return "";
    }

    /**
     * @brief compute features from given input cloud
     * @param signatures
     */
    virtual void
    compute (std::vector<std::vector<float> > & signatures)=0;

    typedef boost::shared_ptr< LocalEstimator<PointT> > Ptr;
    typedef boost::shared_ptr< LocalEstimator<PointT> const> ConstPtr;
};
}
