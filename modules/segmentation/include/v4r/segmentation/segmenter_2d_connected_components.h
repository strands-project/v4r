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

/**
*
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date April, 2016
*      @brief multiplane segmentation (taken from PCL)
*/

#pragma once

#include <v4r/core/macros.h>
#include <v4r/segmentation/segmenter.h>

namespace v4r
{

class V4R_EXPORTS ConnectedComponentsSegmenterParameter : public SegmenterParameter
{
public:
    using SegmenterParameter::min_cluster_size_;
    using SegmenterParameter::max_cluster_size_;
    using SegmenterParameter::distance_threshold_;
    using SegmenterParameter::wsize_;

    ConnectedComponentsSegmenterParameter ()
    {
    }
};


template <typename PointT>
class V4R_EXPORTS ConnectedComponentsSegmenter: public Segmenter<PointT>
{
private:
    using Segmenter<PointT>::normals_;
    using Segmenter<PointT>::clusters_;
    using Segmenter<PointT>::scene_;

    SegmenterParameter param_;

    bool
    check (const pcl::PointXYZI & p1, pcl::PointXYZI & p2) const
    {
        if ( p1.intensity != 0 && ( (p1.getVector3fMap () - p2.getVector3fMap ()).norm () <= param_.cluster_tolerance_) )
        {
                p2.intensity = p1.intensity;
                return false;
        }
        return true; //new label
    }

public:
    ConnectedComponentsSegmenter(const SegmenterParameter &p = SegmenterParameter() ) : param_(p) {  }

    void
    segment();

    bool getRequiresNormals() { return true; }

    typedef boost::shared_ptr< ConnectedComponentsSegmenter<PointT> > Ptr;
    typedef boost::shared_ptr< ConnectedComponentsSegmenter<PointT> const> ConstPtr;
};

}
