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


#ifndef V4R_NARF_KEYPOINT_EXTRACTOR__
#define V4R_NARF_KEYPOINT_EXTRACTOR__

#include <v4r/keypoints/keypoint_extractor.h>

namespace v4r
{
template<typename PointT>
class V4R_EXPORTS NarfKeypointExtractor : public KeypointExtractor<PointT>
{
private:
    typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;
    using KeypointExtractor<PointT>::input_;
    using KeypointExtractor<PointT>::indices_;
    using KeypointExtractor<PointT>::keypoint_indices_;
    using KeypointExtractor<PointT>::keypoint_extractor_type_;
    using KeypointExtractor<PointT>::keypoint_extractor_name_;

public:
    class Parameter
    {
        public:
        size_t img_width_;
        size_t img_height_;
        float cx_;
        float cy_;
        float focal_length_;
        float noise_level_;
        float minimum_range_;

        Parameter(
                size_t img_width = 640,
                size_t img_height = 480,
                float cx = 319.5f,
                float cy = 239.5f,
                float focal_length = 525.5f,
                float noise_level = 0.f,
                float minimum_range = 0.f
                ) :
            img_width_ (img_width),
            img_height_ (img_height),
            cx_ (cx),
            cy_ (cy),
            focal_length_ (focal_length),
            noise_level_ (noise_level),
            minimum_range_ (minimum_range)
        {}
    }param_;

    NarfKeypointExtractor(const Parameter &p = Parameter()) : param_ (p)
    {
        keypoint_extractor_type_ = KeypointType::NARF;
        keypoint_extractor_name_ = "narf";
    }

    void
    compute (pcl::PointCloud<PointT> & keypoints);

    typedef boost::shared_ptr< NarfKeypointExtractor<PointT> > Ptr;
    typedef boost::shared_ptr< NarfKeypointExtractor<PointT> const> ConstPtr;
};
}

#endif
