/******************************************************************************
 * Copyright (c) 2015 Thomas Faeulhammer, Aitor Aldoma
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

#ifndef NOISE_MODELS_H_
#define NOISE_MODELS_H_

#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/common/angles.h>

#include <v4r/core/macros.h>

namespace v4r
{
/**
        * @brief computes Kinect axial and lateral noise parameters for an organized point cloud
        * according to Nguyen et al., 3DIMPVT 2012.
        * It also computed depth discontinuites using PCL's organized edge detection algorithm and the distance of
        * each pixel to these discontinuites.
        * @author Thomas Faeulhammer, Aitor Aldoma
        * @date December 2015
       */
template<class PointT>
class V4R_EXPORTS NguyenNoiseModel
{
public:
    class Parameter
    {
    public:
        bool use_depth_edges_; /// @brief if true, uses PCL's organized edge detection algorithm to compute distance of each pixel to these discontinuites.
        float focal_length_; /// @brief Focal length of the camera
        int edge_radius_;   /// @brief radius in pixel. Only pixels within this radius with respect to an edge point will compute the distance to the edge. Remaining points will have infinite distance to edge
        Parameter(
                bool use_depth_edges = true,
                float focal_length = 525.f,
                int edge_radius = 5)
            :
              use_depth_edges_( use_depth_edges ),
              focal_length_ (focal_length),
              edge_radius_ (edge_radius)
        {}
    }param_;

private:
    typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
    typedef typename pcl::PointCloud<pcl::Normal>::Ptr PointNormalTPtr;
    PointTPtr input_; /// @brief input cloud
    PointNormalTPtr normals_; /// @brief input normal
    std::vector<std::vector<float> > pt_properties_; /// @brief for each pixel save lateral [idx=0] and axial sigma [idx=1] as well as Euclidean distance to depth discontinuity [idx=2]
    pcl::PointIndices discontinuity_edges_; /// @brief indices of the point cloud which represent edges

public:
    NguyenNoiseModel (const Parameter &param=Parameter());

    /**
     * @brief setInputCloud
     * @param[in] input cloud
     */
    void
    setInputCloud (const PointTPtr & input)
    {
        input_ = input;
    }

    /**
     * @brief setInputNormals
     * @param[in] input normals
     */
    void
    setInputNormals (const PointNormalTPtr & normals)
    {
        normals_ = normals;
    }

    /**
     * @brief getDiscontinuityEdges
     * @param[out] point cloud representing the depth discontinuities
     */
    void
    getDiscontinuityEdges(pcl::PointCloud<pcl::PointXYZ>::Ptr & disc) const
    {
        disc.reset(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*input_, discontinuity_edges_, *disc);
    }

    /**
     * @brief computes the point properties for each point (axial, lateral noise as well as distance to depth discontinuities)
     */
    void
    compute();

    /**
     * @brief getPointProperties
     * @return returns for each pixel lateral [idx=0] and axial [idx=1] as well as distance in pixel to closest depth discontinuity [idx=2]
     */
    std::vector<std::vector<float> >
    getPointProperties() const
    {
        return pt_properties_;
    }

    /**
     * @brief compute the expected noise level for one pixel only
     * @param point for which to compute noise level
     * @param surface normal at this point
     * @param sigma_lateral in metres
     * @param sigma_axial in metres
     * @return true if pt and normal are finite, false otherwise
     */
    static
    bool
    computeNoiseLevel(const PointT &pt, const pcl::Normal &n, float &sigma_lateral, float &sigma_axial, float focal_length = 525.f)
    {
        const Eigen::Vector3f & np = n.getNormalVector3fMap();

         if( !pcl::isFinite(pt) || !pcl::isFinite(n) ) {
             sigma_lateral = sigma_axial = std::numeric_limits<float>::max();
             return false;
         }

         //origin to pint
         //Eigen::Vector3f o2p = input_->points[i].getVector3fMap() * -1.f;
         Eigen::Vector3f o2p = Eigen::Vector3f::UnitZ() * -1.f;
         o2p.normalize();
         float angle = pcl::rad2deg(acos(o2p.dot(np)));

         if (angle > 80.f)
             angle = 80.f;

         float sigma_lateral_px = (0.8 + 0.034 * angle / (90.f - angle)) * pt.z / focal_length; // in pixel
         sigma_lateral = sigma_lateral_px * pt.z * 1; // in metres
         sigma_axial = 0.0012 + 0.0019 * ( pt.z - 0.4 ) * ( pt.z - 0.4 ) + 0.0001 * angle * angle / ( sqrt(pt.z) * (90 - angle) * (90 - angle));

         return true;
    }

};
}

#endif /* NOISE_MODELS_H_ */

