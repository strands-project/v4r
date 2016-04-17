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


#ifndef V4R_UNIFORM_SAMPLING_EXTRACTOR__
#define V4R_UNIFORM_SAMPLING_EXTRACTOR__

#include <v4r/keypoints/keypoint_extractor.h>

namespace v4r
{
template<typename PointT>
class V4R_EXPORTS UniformSamplingExtractor : public KeypointExtractor<PointT>
{
private:
    typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;
    using KeypointExtractor<PointT>::input_;
    using KeypointExtractor<PointT>::indices_;
    using KeypointExtractor<PointT>::keypoint_indices_;
    using KeypointExtractor<PointT>::keypoint_extractor_type_;
    using KeypointExtractor<PointT>::keypoint_extractor_name_;

    float sampling_density_; /// @brief sampling distance in meter

public:
    UniformSamplingExtractor(float sampling_density = 0.01f) : sampling_density_ (sampling_density)
    {
        keypoint_extractor_type_ = KeypointType::UniformSampling;
        keypoint_extractor_name_ = "uniform_sampling";
    }

    void
    compute (pcl::PointCloud<PointT> & keypoints);

    typedef boost::shared_ptr< UniformSamplingExtractor<PointT> > Ptr;
    typedef boost::shared_ptr< UniformSamplingExtractor<PointT> const> ConstPtr;
};
}

#endif
