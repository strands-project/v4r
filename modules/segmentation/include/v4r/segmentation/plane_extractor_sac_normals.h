/******************************************************************************
 * Copyright (c) 2017 Thomas Faeulhammer
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
#include <v4r/segmentation/plane_extractor.h>

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS SACNormalsPlaneExtractor : public PlaneExtractor<PointT>
{
protected:
    using PlaneExtractor<PointT>::cloud_;
    using PlaneExtractor<PointT>::normal_cloud_;
    using PlaneExtractor<PointT>::all_planes_;
    using PlaneExtractor<PointT>::param_;

    float downsample_leaf_;
    /** \brief Number of neighbors for normal estimation */
    size_t k_;
    /** \brief Keep points closer than max_z_bounds */
    float max_z_bounds_;
    /** \brief Threshold for SAC plane segmentation */
    double sac_distance_threshold_;

public:
    SACNormalsPlaneExtractor( const PlaneExtractorParameter &p = PlaneExtractorParameter() ) :
        PlaneExtractor<PointT>(p)
    {
        max_z_bounds_ = 1.5;
        k_ = 50;
        sac_distance_threshold_ = 0.01;
        downsample_leaf_ = 0.005f;
    }

    virtual void compute();
    virtual bool getRequiresNormals() const { return true; }

    typedef boost::shared_ptr< SACNormalsPlaneExtractor<PointT> > Ptr;
    typedef boost::shared_ptr< SACNormalsPlaneExtractor<PointT> const> ConstPtr;
};

}
