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

#include <v4r/core/macros.h>
#include <pcl/common/common.h>
#include <v4r/keypoints/types.h>

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS KeypointExtractor
{
protected:
    typename pcl::PointCloud<PointT>::ConstPtr input_; ///< input cloud
    pcl::PointCloud<pcl::Normal>::ConstPtr normals_; ///< surface normals for input cloud
    typename pcl::PointCloud<PointT>::Ptr keypoints_; /// extracted keypoints
    std::vector<int> keypoint_indices_; ///< extracted keypoint indices
    std::vector<int> indices_;  ///< indices of the segmented object (extracted keypoints outside of this will be neglected)

public:
    virtual ~KeypointExtractor() = 0;

    /**
     * @brief setInputCloud
     * @param input input cloud
     */
    void
    setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr & input)
    {
        input_ = input;
    }

    void
    setNormals (const pcl::PointCloud<pcl::Normal>::ConstPtr & normals)
    {
        normals_ = normals;
    }

    virtual bool
    needNormals () const
    {
        return false;
    }

    std::vector<int>
    getKeypointIndices () const
    {
        return keypoint_indices_;
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
     * @brief getKeypointExtractorType
     * @return unique type id of keypoint extractor (as stated in keypoint/types.h)
     */
    virtual int getKeypointExtractorType() const = 0;

    /**
     * @brief getKeypointExtractorName
     * @return type name of keypoint extractor
     */
    virtual std::string getKeypointExtractorName() const = 0;

    /**
     * @brief compute
     * @param keypoints
     */
    virtual void
    compute () = 0;

    /**
     * @brief getKeypoints
     * @return extracted keypoints
     */
    virtual
    typename pcl::PointCloud<PointT>::Ptr
    getKeypoints() const
    {
        return keypoints_;
    }


    typedef boost::shared_ptr< KeypointExtractor<PointT> > Ptr;
    typedef boost::shared_ptr< KeypointExtractor<PointT> const> ConstPtr;
};
}

