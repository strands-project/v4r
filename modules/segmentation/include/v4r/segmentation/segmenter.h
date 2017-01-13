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
*      @brief base class for segmentation
*/

#ifndef V4R_SEGMENTER_H__
#define V4R_SEGMENTER_H__

#include <v4r/core/macros.h>
#include <v4r/common/plane_model.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS Segmenter
{
private:
    mutable pcl::visualization::PCLVisualizer::Ptr vis_;
    mutable int vp1_, vp2_, vp3_;

protected:
    typename pcl::PointCloud<PointT>::ConstPtr scene_; ///< point cloud to be segmented
    pcl::PointCloud<pcl::Normal>::ConstPtr normals_; ///< normals of the cloud to be segmented
    std::vector<pcl::PointIndices> clusters_; ///< segmented clusters. Each cluster represents a bunch of indices of the input cloud
    std::vector<int> indices_;  ///< region of interest
    Eigen::Vector4f dominant_plane_; ///< extracted dominant table plane (if segmentation algorithm supports it)
    std::vector< typename PlaneModel<PointT>::Ptr > all_planes_; ///< all extracted planes (if segmentation algorithm supports it)
    bool visualize_;

public:
    typedef boost::shared_ptr< Segmenter<PointT> > Ptr;
    typedef boost::shared_ptr< Segmenter<PointT> const> ConstPtr;

    virtual ~Segmenter(){}

    /**
     * @brief sets the cloud which ought to be segmented
     * @param cloud
     */
    void
    setInputCloud ( const typename pcl::PointCloud<PointT>::ConstPtr &cloud )
    {
        scene_ = cloud;
    }

    /**
     * @brief sets the normals of the cloud which ought to be segmented
     * @param normals
     */
    void
    setNormalsCloud ( const pcl::PointCloud<pcl::Normal>::ConstPtr &normals )
    {
        normals_ = normals;
    }

    /**
     * @brief sets the mask of
     * @param cloud
     */
    void
    setIndices ( const std::vector<int> &indices )
    {
        indices_ = indices;
    }

    /**
     * @brief get segmented indices
     * @param indices
     */
    void
    getSegmentIndices ( std::vector<pcl::PointIndices> & indices ) const
    {
        indices = clusters_;
    }

    virtual bool
    getRequiresNormals() = 0;


    /**
     * @brief returns extracted table plane (assumes segmentation algorithm computes it)
     * @return table plane vector
     */
    Eigen::Vector4f
    getTablePlane() const
    {
        return dominant_plane_;
    }

    /**
     * @brief returns all extracted planes (assumes segmentation algorithm computes it)
     * @return extracted planes
     */
    std::vector< typename PlaneModel<PointT>::Ptr >
    getAllPlanes() const
    {
        return all_planes_;
    }

    /**
     * @brief visualize found clusters
     */
    void
    visualize() const;

    /**
     * @brief segment
     */
    virtual void
    segment() = 0;
};

}

#endif
