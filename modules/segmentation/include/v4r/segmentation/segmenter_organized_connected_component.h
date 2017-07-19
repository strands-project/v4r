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
*      @brief organized multiplane segmentation (taken from PCL)
*/

#pragma once

#include <v4r/core/macros.h>
#include <v4r/segmentation/segmenter.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace v4r
{


class V4R_EXPORTS OrganizedConnectedComponentSegmenterParameter : public SegmenterParameter
{
public:
    using SegmenterParameter::min_cluster_size_;
    using SegmenterParameter::max_cluster_size_;
    using SegmenterParameter::distance_threshold_;

    OrganizedConnectedComponentSegmenterParameter ()
    {
        distance_threshold_ = 0.035f;
    }
};

template <typename PointT>
class V4R_EXPORTS OrganizedConnectedComponentSegmenter : public Segmenter<PointT>
{
    using Segmenter<PointT>::normals_;
    using Segmenter<PointT>::clusters_;
    using Segmenter<PointT>::scene_;

    OrganizedConnectedComponentSegmenterParameter param_;

public:
    OrganizedConnectedComponentSegmenter(const OrganizedConnectedComponentSegmenterParameter &p = OrganizedConnectedComponentSegmenterParameter() )
        : param_(p)
    { }

    bool getRequiresNormals() { return true; }

    void segment();

    typedef boost::shared_ptr< OrganizedConnectedComponentSegmenter<PointT> > Ptr;
    typedef boost::shared_ptr< OrganizedConnectedComponentSegmenter<PointT> const> ConstPtr;
};

}
