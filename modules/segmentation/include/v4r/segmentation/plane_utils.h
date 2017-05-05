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
 * @brief dist2plane checks the minimum distance of a point to a plane \f$|a*x + b*y + c*z + d| \le threshold \f$
 * @param point (x,y,z) in 3D space
 * @param plane defined as Vector (a,b,c,d)
 * @return distance to plane (positive if above plane, negative below)
 */
inline float dist2plane(const Eigen::Vector3f& point, const Eigen::Vector4f &plane)
{
    return (point.dot(plane.head(3)) + plane(3)) / plane.head(3).norm();
}

/**
 * @brief getClosestPointOnPlane
 * @param query_pt the point for which the closest point on the plane should be found
 * @param plane plane equation \f$|a*x + b*y + c*z + d|
 * @return closest point on the plane
 */
inline
Eigen::Vector3f
getClosestPointOnPlane(const Eigen::Vector3f &query_pt, const Eigen::Vector4f &plane)
{
    float dist = dist2plane(query_pt, plane);
    Eigen::Vector3f plane_normal = plane.head(3);
    plane_normal.normalize();

    Eigen::Vector3f closest_pt = query_pt - plane_normal * dist;
    return closest_pt;
}


/**
 * @brief DistanceBetweenPlanes (assumes planes are parallel)
 * @param plane1 [a*x + b*y + c*z + d = 0]
 * @param plane2
 * @return distance between two (parallel) planes
 */
inline float DistanceBetweenPlanes(const Eigen::Vector4f &plane1, const Eigen::Vector4f &plane2)
{
    // a b and c must be equal
    float norm1 = plane1.head(3).norm();
    float norm2 = plane2.head(3).norm();
    const Eigen::Vector4f plane1_normalized = plane1/norm1;
    const Eigen::Vector4f plane2_normalized = plane2/norm2;

    return ( fabs(plane1_normalized(3) - plane2_normalized(3)) );
}


/**
 * @brief CosAngleBetweenPlanes
 * @param plane1
 * @param plane2
 * @return cosinus between two planes
 */
inline float CosAngleBetweenPlanes(const Eigen::Vector4f &plane1, const Eigen::Vector4f &plane2)
{
    Eigen::Vector3f normal1 = plane1.head(3);
    Eigen::Vector3f normal2 = plane2.head(3);
    normal1.normalize();
    normal2.normalize();

    return normal1.dot(normal2);
}


/**
 * @brief is_inlier checks if a point (x,y,z) in 3D space if it lies on a plane within a certain threshold \f$|a*x + b*y + c*z + d| \le threshold \f$
 * @param point (x,y,z) in 3D space
 * @param plane defined as Vector (a,b,c,d)
 * @param threshold in meter
 * @return true if is on plane
 */
inline bool is_inlier(const Eigen::Vector3f& point, const Eigen::Vector4f &plane, float threshold)
{
    Eigen::Vector3f normalized_normal = plane.head(3);
    normalized_normal.normalize();

    return fabs( dist2plane(point,plane) ) < threshold;
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
    return dist2plane(point,plane) > threshold;
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


/**
 * @brief visualizePlane visualize plane inliers for a point cloud
 * @param cloud
 * @param plane
 * @param inlier_threshold
 * @param window_title
 */
template<typename PointT>
V4R_EXPORTS
void
visualizePlane(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const Eigen::Vector4f &plane, float inlier_threshold = 0.01f, const std::string &window_title = "plane inliers" );

/**
 * @brief visualizePlanes visualize plane inliers for multiple planes for a point cloud
 * @param cloud
 * @param planes
 * @param inlier_threshold
 * @param window_title
 */
template<typename PointT>
V4R_EXPORTS
void
visualizePlanes(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > &planes, float inlier_threshold = 0.01f, const std::string &window_title = "plane inliers" );

}
