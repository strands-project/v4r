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

/**
*
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date April, 2016
*      @brief Euclidean segmentation (taken from PCL)
*/

#pragma once

#include <v4r/core/macros.h>
#include <v4r/common/pcl_utils.h>
#include <v4r/segmentation/segmenter.h>
#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

namespace v4r
{


class V4R_EXPORTS EuclideanSegmenterParameter : public SegmenterParameter
{
public:
    using SegmenterParameter::min_cluster_size_;
    using SegmenterParameter::max_cluster_size_;
    using SegmenterParameter::cluster_tolerance_;

    EuclideanSegmenterParameter ()
    {
        cluster_tolerance_ = 0.035f;
    }
};

template <typename PointT>
class V4R_EXPORTS EuclideanSegmenter : public Segmenter<PointT>
{
    using Segmenter<PointT>::clusters_;
    using Segmenter<PointT>::scene_;

    EuclideanSegmenterParameter param_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EuclideanSegmenter(const EuclideanSegmenterParameter &p = EuclideanSegmenterParameter() ) : param_(p)
    { }

    bool getRequiresNormals() { return false; }

    void
    segment();

    typedef boost::shared_ptr< EuclideanSegmenter<PointT> > Ptr;
    typedef boost::shared_ptr< EuclideanSegmenter<PointT> const> ConstPtr;
};

}
