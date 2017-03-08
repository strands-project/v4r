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

#include <v4r/core/macros.h>
#include <v4r/common/normals.h>
#include <v4r/segmentation/all_headers.h>

#pragma once

namespace v4r
{

namespace apps
{

class V4R_EXPORTS CloudSegmenterParameter
{
public:
    float chop_z_; ///< cut-off distance in meter. Points further away than this threshold will be neglected
    float plane_inlier_threshold_;
    size_t min_plane_inliers_;
    bool skip_segmentation_;    ///< if true, skips segmentation
    bool skip_plane_extraction_;    ///< if true, skips plane extraction
    bool only_remove_planes_;   ///< if true, removes plane from input cloud only. If false, removes plane and everything below it (i.e. further away from the camera)
    bool dominant_plane_only_;  ///< if true, removes only the plane with the largest number of plane inliers

public:
    CloudSegmenterParameter()
        :
          chop_z_ (std::numeric_limits<float>::max()),
          plane_inlier_threshold_(0.02f),
          min_plane_inliers_(2000),
          skip_segmentation_(false),
          skip_plane_extraction_(false),
          only_remove_planes_(false),
          dominant_plane_only_(true)
    {}
};

/**
 * @brief The CloudSegmenter class segments an input cloud by first doing plane removal and then running the segmentation algorithm
 * @author Thomas Faeulhammer
 */

template<typename PointT>
class V4R_EXPORTS CloudSegmenter
{
private:
    typename v4r::PlaneExtractor<PointT>::Ptr plane_extractor_;
    typename v4r::Segmenter<PointT>::Ptr segmenter_;
    typename v4r::NormalEstimator<PointT>::Ptr normal_estimator_;
    std::vector<std::vector<int> > found_clusters_;
    std::vector< Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > planes_;
    std::vector<std::vector<int> > plane_inliers_;

    CloudSegmenterParameter param_;

public:
    CloudSegmenter(const CloudSegmenterParameter &p = CloudSegmenterParameter() ) :
        param_(p)
    { }

    /**
     * @brief initialize initialize Point Cloud Segmenter (sets up plane extraction, segmentation and potential normal estimator)
     * @param arguments
     */
    void initialize(std::vector<std::string> &command_line_arguments);

    /**
     * @brief recognize recognize objects in point cloud
     * @param cloud (organized) point cloud
     * @return
     */
    void
    segment(const typename pcl::PointCloud<PointT>::ConstPtr &cloud);

    /**
     * @brief getClusters
     * @param cluster_indices
     */
    std::vector<std::vector<int> >
    getClusters() const
    {
       return found_clusters_;
    }

    /**
     * @brief getPlanes
     * @return extracted planar surfaces
     */
    std::vector< Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >
    getPlanes( ) const
    {
        return planes_;
    }

    std::vector<std::vector<int> >
    getPlaneInliers() const
    {
        return plane_inliers_;
    }
};

}

}
