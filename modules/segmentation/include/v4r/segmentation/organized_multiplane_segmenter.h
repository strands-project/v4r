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

#ifndef V4R_ORGANIZED_MULTIPLANE_SEGMENTER_H__
#define V4R_ORGANIZED_MULTIPLANE_SEGMENTER_H__

#include <v4r/core/macros.h>
#include <v4r/segmentation/segmenter.h>

namespace v4r
{

template <typename PointT>
class V4R_EXPORTS OrganizedMultiplaneSegmenter : public Segmenter<PointT>
{
    using Segmenter<PointT>::indices_;
    using Segmenter<PointT>::normals_;
    using Segmenter<PointT>::clusters_;
    using Segmenter<PointT>::scene_;
    using Segmenter<PointT>::dominant_plane_;

public:

    class Parameter
    {
    public:
        int min_cluster_size_, num_plane_inliers_;
        double sensor_noise_max_,
               angular_threshold_deg_;
        Parameter (int min_cluster_size=500,
                   int num_plane_inliers=1000,
                   double sensor_noise_max = 0.01f,
                   double angular_threshold_deg = 10.f)
            :
              min_cluster_size_ (min_cluster_size),
              num_plane_inliers_ (num_plane_inliers),
              sensor_noise_max_ (sensor_noise_max),
              angular_threshold_deg_ (angular_threshold_deg)
        {}
    }param_;

    OrganizedMultiplaneSegmenter(const Parameter &p = Parameter() ) : param_(p)  { }

    void
    segment();

    typedef boost::shared_ptr< OrganizedMultiplaneSegmenter<PointT> > Ptr;
    typedef boost::shared_ptr< OrganizedMultiplaneSegmenter<PointT> const> ConstPtr;
};

}

#endif
