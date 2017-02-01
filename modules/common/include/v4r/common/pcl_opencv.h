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

#include <boost/dynamic_bitset.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <v4r/core/macros.h>
#include <v4r/common/camera.h>

#pragma once

namespace v4r
{

///
/// \brief The PCLOpenCVConverter class converts PCL point clouds into RGB, depth or occupancy images.
/// \author Thomas Faeulhammer
/// \date August 2016
///
template<typename PointT>
class V4R_EXPORTS PCLOpenCVConverter
{
private:
    typedef std::vector<uchar> (PCLOpenCVConverter<PointT>::*pf)(int v, int u) const;

    typename pcl::PointCloud<PointT>::ConstPtr cloud_;   ///@brief cloud to be converted
    std::vector<int> indices_; ///@brief pixel indices to be extracted (if empty, all pixel will be extracted)
    Camera::ConstPtr cam_; ///@brief camera parameters (used for re-projection if point cloud is not organized)
    bool remove_background_;  ///@brief if true, will set pixel not specified by the indices to the background color defined by its member variable background_color_
    cv::Vec3b background_color_; ///@brief background color (only used if indices are not empty and remove_background is set to true)
    float min_depth_m_;   ///@brief minimum depth in meter for normalization
    float max_depth_m_;   ///@brief maximum depth in meter for normalization
    cv::Rect roi_;  ///@brief region of interest with all given indices taken into account
    cv::Mat output_matrix_; /// @brief
    Eigen::MatrixXi index_map_; ///< index map showing which point of the unorganized(!) point cloud maps to which pixel in the image plane (pixel not occupied by any point have value -1)

    cv::Rect computeROIfromIndices();
    void computeOrganizedCloud();

    std::vector<uchar> getRGB(int v, int u)
    {
        const PointT &pt = cloud_->at(u,v);
        std::vector<uchar> rgb = {pt.b, pt.g, pt.r};
        return rgb;
    }

    std::vector<uchar> getZNormalized(int v, int u)
    {
        const PointT &pt = cloud_->at(u,v);
        std::vector<uchar> rgb = {std::min<uchar>(255, std::max<uchar>(0, 255.f*(pt.z-min_depth_m_)/(max_depth_m_-min_depth_m_)) )};
        return rgb;
    }

    std::vector<float> getZ(int v, int u)
    {
        const PointT &pt = cloud_->at(u,v);
        float depth;
        pcl_isfinite(pt.z) ? depth=pt.z : depth=0.f;
        std::vector<float> z = {depth};
        return z;
    }

    std::vector<uchar> getOccupied(int v, int u)
    {
        const PointT &pt = cloud_->at(u,v);
        if (std::isfinite(pt.z) )
            return std::vector<uchar>(1, 255);
        else
            return std::vector<uchar>(1, 0);
    }

public:
    PCLOpenCVConverter(const typename pcl::PointCloud<PointT>::ConstPtr cloud = nullptr) :
        cloud_(cloud), remove_background_ ( true ), background_color_ ( cv::Vec3b(0,0,0) ), min_depth_m_ (0.f), max_depth_m_ (5.f)
    { }

    ///
    /// \brief setInputCloud
    /// \param cloud point cloud to be converted
    ///
    void setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr cloud)
    {
        cloud_ = cloud;
        index_map_.resize(0,0);
    }

    ///
    /// \brief setIndices
    /// \param indices indices to be extracted (if empty, all pixel will be extracted)
    ///
    void setIndices (const std::vector<int> &indices)
    {
        indices_ = indices;
    }

    ///
    /// \brief setCamera
    /// \param cam camera parameters (used for re-projection if point cloud is not organized)
    ///
    void setCamera (const Camera::ConstPtr cam)
    {
        cam_ = cam;
    }

    ///
    /// \brief setRemoveBackground
    /// \param remove_background if true, will set pixel not specified by the indices to the background color defined by its member variable background_color_
    ///
    void setRemoveBackground(bool remove_background = true)
    {
        remove_background_ = remove_background;
    }

    ///
    /// \brief setBackgroundColor
    /// \param background color (only used if indices are not empty and remove_background is set to true)
    ///
    void setBackgroundColor (uchar r, uchar g, uchar b)
    {
        background_color_ = cv::Vec3b(r,g,b);
    }

    ///
    /// \brief setMinMaxDepth (used for extracting normalized depth values)
    /// \param min_depth in meter
    /// \param max_depth in meter
    ///
    void setMinMaxDepth (float min_depth_m, float max_depth_m)
    {
        min_depth_m_ = min_depth_m;
        max_depth_m_ = max_depth_m;
    }

    cv::Mat extractDepth(); /// extracts depth image from pointcloud whereby depth values correspond to distance in meter
    cv::Mat getNormalizedDepth();  /// extracts depth image from pointcloud whereby depth values in meter are normalized uniformly from 0 (<=min_depth_m_) to 255 (>=max_depth_m_)
    cv::Mat getRGBImage();  /// returns the RGB image of the point cloud
    cv::Mat getOccupiedPixel(); /// computes a binary iamge from a point cloud which elements indicate if the corresponding pixel (sorted in row-major order) is hit by a finite point in the point cloud.

    /**
     * @brief getROI
     * @return region of interest (bounding box defined by the given indices). If no indices are given, ROI will be whole image.
     */
    cv::Rect
    getROI() const
    {
        return roi_;
    }


    /**
     * @brief getIndexMap
     * @return returns the map of indices, i.e. which pixel represents which point of the unorganized(!) point cloud.
     */
    Eigen::MatrixXi
    getIndexMap() const
    {
        return index_map_;
    }

    template<typename MatType=uchar> cv::Mat fillMatrix( std::vector<MatType> (PCLOpenCVConverter<PointT>::*pf)(int v, int u) );
};

    /**
      * @brief computes the depth map of a point cloud with fixed size output
      * @param indices of the points belonging to the object (assumes row major indices)
      * @param width of the image/point cloud
      * @param height of the image/point cloud
      */
    V4R_EXPORTS
    cv::Rect
    computeBoundingBox (const std::vector<int> &indices, size_t width, size_t height);
}
