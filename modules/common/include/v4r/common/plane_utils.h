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

#include <boost/dynamic_bitset.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <v4r/core/macros.h>
#include <v4r/common/camera.h>

#pragma once

namespace v4r
{


/**
 * @brief is_inlier checks if a point (x,y,z) in 3D space if it lies on a plane within a certain threshold \f$|a*x + b*y + c*z + d| \le threshold \f$
 * @param point (x,y,z) in 3D space
 * @param plane defined as Vector (a,b,c,d)
 * @param threshold in meter
 * @return true if is on plane
 */
inline bool is_inlier(const Eigen::Vector3f& point, const Eigen::Vector4f &plane, float threshold)
{
    return fabs(point.dot(plane.segment<3>(0)) + plane(3)) < threshold;
}


/**
 * @brief is_above_plane checks if a point (x,y,z) in 3D space lies above a plane \f$a*x + b*y + c*z + d > threshold \f$
 * @param point (x,y,z) in 3D space
 * @param plane defined as Vector (a,b,c,d)
 * @param threshold in meter
 * @return true if above plane
 */
inline bool is_above_plane(const Eigen::Vector3f& point, const Eigen::Vector4f &plane, float threshold)
{
    return point.dot(plane.segment<3>(0)) + plane(3) > threshold;
}


/**
 * @brief checks for each point (x,y,z) in the cloud if its above a plane \f$a*x + b*y + c*z + d > threshold \f$
 * @param cloud input cloud
 * @param plane defined as Vector (a,b,c,d)
 * @param threshold in meter
 * @return all indices of the cloud fulfilling the equation
 */
template<typename PointT>
V4R_EXPORTS std::vector<int>
get_above_plane_inliers(const pcl::PointCloud<PointT> &cloud, const Eigen::Vector4f &plane, float threshold)
{
    std::vector<int> inliers (cloud.points.size());
    size_t kept=0;
    for (size_t i = 0; i < cloud.points.size(); i++) {
        if ( is_above_plane(cloud.points[i].getVector3fMap(), plane, threshold) )
            inliers[ kept++ ] = i;
    }
    inliers.resize(kept);
    return inliers;
}


/**
 * @brief checks for each point (x,y,z) in the cloud if its on a plane \f$|a*x + b*y + c*z + d| \le threshold \f$
 * @param cloud input cloud
 * @param plane defined as Vector (a,b,c,d)
 * @param threshold in meter
 * @return all indices of the cloud fulfilling the equation
 */
template<typename PointT>
V4R_EXPORTS
std::vector<int>
get_all_plane_inliers(const pcl::PointCloud<PointT> &cloud, const Eigen::Vector4f &plane, float threshold)
{
    std::vector<int> inliers (cloud.points.size());
    size_t kept=0;
    for (size_t i = 0; i < cloud.points.size(); i++) {
        if ( is_inlier(cloud.points[i].getVector3fMap(), plane, threshold) )
            inliers[ kept++ ] = i;
    }
    inliers.resize(kept);
    return inliers;
}

/**
 * @brief getConvexHullCloud computes the convex hull of a plane cloud
 * @param cloud input cloud projected onto the table plane
 * @return convex hull cloud
 */
template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr
getConvexHullCloud(const typename pcl::PointCloud<PointT>::ConstPtr cloud);

/**
 * @brief get_largest_connected_plane_inliers finds the largest cluster of connected points that fulfill the plane equation
 * (e.g. remove points also fulfilling the plane equation but belonging to the background)
 * @param[in] cloud input cloud
 * @param[in] plane defined as Vector (a,b,c,d) with a*x + b*y + c*z + d = 0
 * @param[in] threshold inlier threshold in meter
 * @param[in] cluster_tolerance cluster tolerance
 * @param[in] min_cluster_size minimum number of points neccessary to create a cluster
 * @return indices of the all points belonging to the largest connected component fulfilling the plane equation
 */
template<typename PointT>
V4R_EXPORTS
std::vector<int>
get_largest_connected_plane_inliers(const pcl::PointCloud<PointT> &cloud, const Eigen::Vector4f &plane, float threshold, float cluster_tolerance = 0.01f, int min_cluster_size = 200);

/**
 * @brief get_largest_connected_inliers finds the largest cluster of connected points among the indices
 * (e.g. remove points also fulfilling the plane equation but belonging to the background)
 * @param[in] cloud input cloud
 * @param[in] inlier indices of points that are to be segmented
 * @param[in] cluster_tolerance cluster tolerance
 * @param[in] min_cluster_size minimum number of points neccessary to create a cluster
 * @return indices of the all points belonging to the largest connected component fulfilling the plane equation
 */
template<typename PointT>
V4R_EXPORTS
std::vector<int>
get_largest_connected_inliers(const pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices, float cluster_tolerance = 0.01f, int min_cluster_size = 200);


}
