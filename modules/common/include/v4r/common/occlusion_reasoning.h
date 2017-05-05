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

#ifndef V4R_OCCLUSION_REASONING_H_
#define V4R_OCCLUSION_REASONING_H_

#include <boost/dynamic_bitset.hpp>
#include <pcl/point_cloud.h>
#include <v4r/common/camera.h>
#include <v4r/core/macros.h>


namespace v4r
{

/**
 * @brief Class to reason about occlusion
 * @author: Thomas Faeulhammer
 * @date August 2016
 */
template<typename PointTA, typename PointTB>
class V4R_EXPORTS OcclusionReasoner
{
private:
    typename pcl::PointCloud<PointTA>::ConstPtr occluder_cloud_; ///< organized_cloud point cloud that potentially causes occlusion
    typename pcl::PointCloud<PointTB>::ConstPtr cloud_to_be_filtered_; ///< to_be_filtered point cloud to be checked for occlusion
    float occlusion_threshold_m_;   ///< occlusion threshold in meter
    Camera::ConstPtr cam_; ///@brief camera parameters for re-projection to image plane by depth buffering (only used if point clouds are not organized)
    boost::dynamic_bitset<> px_is_visible_; ///< indicates if a pixel re-projected by the cloud_to_be_filtered_ is in front of the occlusion cloud (i.e. if the pixel belongs to the object)

public:
    OcclusionReasoner()
        : occlusion_threshold_m_ (0.01f)
    { }

    /**
     * @brief setCamera
     * @param cam
     */
    void
    setCamera ( const Camera::ConstPtr cam )
    {
        cam_ = cam;
    }

    /**
     * @brief setOcclusionThreshold
     * @param occlusion_thresh_m
     */
    void
    setOcclusionThreshold(float occlusion_thresh_m)
    {
        occlusion_threshold_m_ = occlusion_thresh_m;
    }

    /**
     * @brief setOcclusionCloud
     * @param occlusion_cloud cloud that can cause occlusion
     */
    void
    setOcclusionCloud( const typename pcl::PointCloud<PointTA>::ConstPtr occlusion_cloud )
    {
        occluder_cloud_ = occlusion_cloud;
    }

    /**
     * @brief setInputCloud
     * @param cloud_to_be_filtered object cloud that is checked for occlusion
     */
    void
    setInputCloud( const typename pcl::PointCloud<PointTB>::ConstPtr cloud_to_be_filtered )
    {
        cloud_to_be_filtered_ = cloud_to_be_filtered;
    }

    /**
     * @brief getPixelMask
     * @return indicates if a pixel re-projected by the cloud_to_be_filtered_ is in front of the occlusion cloud (i.e. if the pixel belongs to the object) - bitset size is equal to number of pixel
     */
    boost::dynamic_bitset<>
    getPixelMask() const
    {
        return px_is_visible_;
    }

    /**
     * @brief compute occlusion
     * @return binary mask which indicates for each point of the cloud_to_be_filtered_ if it is visible (true) or occluded (false) - bitset size is equal to number of points of cloud
     */
    boost::dynamic_bitset<> computeVisiblePoints();
};
}

#endif
