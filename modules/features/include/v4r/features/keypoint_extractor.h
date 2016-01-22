/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma
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


#ifndef V4R_KEYPOINT_EXTRACTOR___
#define V4R_KEYPOINT_EXTRACTOR___

#include <v4r/core/macros.h>
#include <pcl/common/common.h>

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS KeypointExtractor
{
protected:
    typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;
    typedef typename pcl::PointCloud<PointT>::Ptr PointOutTPtr;
    typename pcl::PointCloud<PointT>::Ptr input_;
    float radius_;
    std::vector<int> keypoint_indices_;

public:

    void
    setInputCloud (const PointInTPtr & input)
    {
        input_ = input;
    }

    void
    setSupportRadius (float f)
    {
        radius_ = f;
    }

    virtual void
    setNormals (const pcl::PointCloud<pcl::Normal>::Ptr & normals)
    {
        (void)normals;
        std::cerr << "setNormals is not implemented for this object." << std::endl;
    }

    virtual bool
    needNormals ()
    {
        return false;
    }

    void getKeypointsIndices (std::vector<int> &keypoint_indices) const
    {
        keypoint_indices = keypoint_indices_;
    }


    virtual void
    compute (pcl::PointCloud<PointT> & keypoints) = 0;
};
}

#endif
