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
    typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;

    mutable pcl::visualization::PCLVisualizer::Ptr vis_;
    mutable int vp1_, vp2_, vp3_;

protected:
    PointTPtr scene_; /// \brief point cloud to be segmented
    pcl::PointCloud<pcl::Normal>::Ptr normals_; /// @brief normals of the cloud to be segmented
    std::vector<pcl::PointIndices> clusters_; /// @brief segmented clusters. Each cluster represents a bunch of indices of the input cloud
    std::vector<int> indices_;  /// @brief region of interest
    Eigen::Vector4f dominant_plane_; /// @brief extracted dominant table plane (if segmentation algorithm supports it)
    std::vector< typename PlaneModel<PointT>::Ptr > all_planes_; /// @brief all extracted planes (if segmentation algorithm supports it)
    bool visualize_;

public:
    virtual ~Segmenter() = 0;

    /**
     * @brief sets the cloud which ought to be segmented
     * @param cloud
     */
    void
    setInputCloud ( const typename pcl::PointCloud<PointT>::Ptr &cloud )
    {
        scene_ = cloud;
    }

    /**
     * @brief sets the normals of the cloud which ought to be segmented
     * @param normals
     */
    void
    setNormalsCloud ( const pcl::PointCloud<pcl::Normal>::Ptr &normals )
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
    visualize() const
    {
        if(!vis_)
        {
            vis_.reset ( new pcl::visualization::PCLVisualizer("Segmentation Results") );
            vis_->createViewPort(0,0,0.33,1,vp1_);
            vis_->createViewPort(0.33,0,0.66,1,vp2_);
            vis_->createViewPort(0.66,0,1,1,vp3_);
        }
        vis_->removeAllPointClouds();
        vis_->removeAllShapes();
        vis_->addPointCloud(scene_, "cloud", vp1_);


        typename pcl::PointCloud<PointT>::Ptr colored_cloud (new pcl::PointCloud<PointT>());
        for(size_t i=0; i < clusters_.size(); i++)
        {
            pcl::PointCloud<PointT> cluster;
            pcl::copyPointCloud(*scene_, clusters_[i], cluster);

            const uint8_t r = rand()%255;
            const uint8_t g = rand()%255;
            const uint8_t b = rand()%255;
            for(size_t pt_id=0; pt_id<cluster.points.size(); pt_id++)
            {
                cluster.points[pt_id].r = r;
                cluster.points[pt_id].g = g;
                cluster.points[pt_id].b = b;
            }
            *colored_cloud += cluster;
        }
        vis_->addPointCloud(colored_cloud,"segments", vp2_);

        typename pcl::PointCloud<PointT>::Ptr planes (new pcl::PointCloud<PointT>(*scene_));

        Eigen::Matrix3Xf plane_colors(3, all_planes_.size());
        for(size_t i=0; i<all_planes_.size(); i++)
        {
            plane_colors(0, i) = rand()%255;
            plane_colors(1, i) = rand()%255;
            plane_colors(2, i) = rand()%255;
        }

        for(PointT &pt :planes->points)
        {
            if ( !pcl::isFinite( pt ) )
                continue;

            const Eigen::Vector4f xyz_p = pt.getVector4fMap ();
            pt.g = pt.b = pt.r = 0;


            for(size_t i=0; i<all_planes_.size(); i++)
            {
                float val = xyz_p.dot(all_planes_[i]->coefficients_);

                if ( std::abs(val) < 0.02f)
                {
                    pt.r = plane_colors(0,i);
                    pt.g = plane_colors(1,i);
                    pt.b = plane_colors(2,i);
                }
            }

            float val = xyz_p.dot(dominant_plane_);

            if ( std::abs(val) < 0.02f)
                pt.r = 255;
        }
        vis_->addPointCloud(planes,"table plane", vp3_);
        vis_->addText("input", 10, 10, 15, 1, 1, 1, "input", vp1_);
        vis_->addText("segments", 10, 10, 15, 1, 1, 1, "segments", vp2_);
        vis_->addText("dominant plane", 10, 10, 15, 1, 1, 1, "dominant_plane", vp3_);
        vis_->addText("all other planes", 10, 25, 15, 1, 1, 1, "other_planes", vp3_);
        vis_->resetCamera();
        vis_->spin();
    }

    virtual void
    segment() = 0;

    typedef boost::shared_ptr< Segmenter<PointT> > Ptr;
    typedef boost::shared_ptr< Segmenter<PointT> const> ConstPtr;
};

}

#endif
