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

#include <v4r/features/keypoint_extractor.h>

namespace v4r
{
template<typename PointT>
class V4R_EXPORTS UniformSamplingExtractor : public KeypointExtractor<PointT>
{
private:
    typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;
    bool filter_planar_;
    using KeypointExtractor<PointT>::input_;
    using KeypointExtractor<PointT>::radius_;
    using KeypointExtractor<PointT>::keypoint_indices_;
    float sampling_density_;
    float max_distance_;
    float threshold_planar_;
    bool z_adaptative_;
    bool force_unorganized_;

    void
    filterPlanar (const PointInTPtr & input, std::vector<int> &kp_idx);

public:

    UniformSamplingExtractor()
    {
        max_distance_ = std::numeric_limits<float>::infinity();
        threshold_planar_ = 1.e-2;
        z_adaptative_ = false;
        force_unorganized_ = false;
    }

    void setForceUnorganized(bool b)
    {
        force_unorganized_ = b;
    }

    void zAdaptative(bool b)
    {
        z_adaptative_ = b;
    }

    void setThresholdPlanar(float t)
    {
        threshold_planar_ = t;
    }

    void setMaxDistance(float d)
    {
        max_distance_ = d;
    }

    void
    setFilterPlanar (bool b)
    {
        filter_planar_ = b;
    }

    void
    setSamplingDensity (float f)
    {
        sampling_density_ = f;
    }

    void
    compute (pcl::PointCloud<PointT> & keypoints);

    void
    compute (std::vector<int> & indices);
};
}

#endif
