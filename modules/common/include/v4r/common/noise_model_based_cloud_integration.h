/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma, Thomas Faeulhammer
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


#ifndef NMBasedCloudIntegration_H
#define NMBasedCloudIntegration_H

#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/octree/impl/octree_iterator.hpp>

#include <v4r/core/macros.h>
#include <v4r/common/miscellaneous.h>

namespace v4r
{

/**
 * @brief reconstructs a point cloud from several input clouds. Each point of the input cloud is associated with a weight
 * which states the measurement confidence ( 0... max noise level, 1... very confident). Each point is accumulated into a
 * big cloud and then reprojected into the various image planes of the input clouds to check for conflicting points.
 * Conflicting points will be removed and the remaining points put into an octree
 *
 */
template<class PointT>
class V4R_EXPORTS NMBasedCloudIntegration
{
public:
    class V4R_EXPORTS Parameter
    {
    public:
        int min_points_per_voxel_;  /// @brief the minimum number of points in a leaf of the octree of the big cloud.
        float final_resolution_;
        float octree_resolution_;   /// @brief resolution of the octree of the big point cloud
        float min_weight_;
        float threshold_ss_;    /// @brief distance in m to check wheter a point from the accumulated cloud is explained by a point from an input cloud
        float max_distance_;    /// @brief each point further away than this distance will be assigned the maximum noise level (and so neglected)
        float focal_length_;   /// @brief focal length of the cameras; used for reprojection of the points into each image plane
        bool average_;  /// @brief if true, takes the average color (for each color componenent) and normal within all the points in the leaf of the octree. Otherwise, it takes the point within the octree with the best noise weight
        bool weighted_average_;
        Parameter(
                int min_points_per_voxel = 0,
                float final_resolution = 0.001f,
                float octree_resolution =  0.005f,
                float min_weight = 0.9f,
                float threshold_ss = 0.003f,
                float max_distance = 5.f,
                float focal_length = 525.f,
                bool average = false,
                bool weighted_average = true) :
            min_points_per_voxel_(min_points_per_voxel),
            final_resolution_(final_resolution),
            octree_resolution_(octree_resolution),
            min_weight_ (min_weight),
            threshold_ss_(threshold_ss),
            max_distance_(max_distance),
            focal_length_ (focal_length),
            average_ (average),
            weighted_average_ (weighted_average)
        {

        }
    }param_;

private:
    typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
    typedef typename pcl::PointCloud<pcl::Normal>::Ptr PointNormalTPtr;
    std::vector<PointTPtr> input_clouds_;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transformations_to_global_;
    std::vector<std::vector<float> > noise_weights_;
    std::vector<std::vector<float> > sigmas_combined_;
    std::vector<std::vector<std::vector<float> > > sigmas_; /// @brief for each cloud, for each pixel represent lateral [idx=0] and axial [idx=1] as well as distance to closest depth discontinuity [idx=2]
    typename boost::shared_ptr<pcl::octree::OctreePointCloudPointVector<PointT> > octree_;
    std::vector<float> big_cloud_weights_;
    std::vector<std::vector<float> > big_cloud_sigmas_;
    std::vector<PointNormalTPtr> input_normals_;
    PointNormalTPtr big_cloud_normals_;
    std::vector<int> big_cloud_origin_cloud_id_;    /// @brief saves the index of the input cloud vector for each point in the big cloud (to know from which camera the point comes from)
    PointNormalTPtr output_normals_;
    std::vector<PointTPtr> input_clouds_used_;
    std::vector<std::vector<size_t> > indices_;

    struct PointInfo{
        Eigen::Vector3f position;
        Eigen::Vector3f normal;
        int index_in_big_cloud;
        float distance_to_depth_discontinuity;
        float probability;
        float r,g,b;

        bool operator<(const PointInfo other) const
        {
            return probability > other.probability;
        }
    };

public:
    NMBasedCloudIntegration (const Parameter &p=Parameter());

    void getInputCloudsUsed(std::vector<PointTPtr> & input_clouds_used) const
    {
        input_clouds_used = input_clouds_used_;
    }

    void getOutputNormals(PointNormalTPtr & output) const
    {
        output = output_normals_;
    }

    void
    setInputClouds (const std::vector<PointTPtr> & input)
    {
        input_clouds_ = input;
    }

    void
    setInputNormals (const std::vector<PointNormalTPtr> & input)
    {
        input_normals_ = input;
    }

    void
    setIndices(const std::vector<std::vector<size_t> > & indices)
    {
        indices_ = indices;
    }

    void
    setIndices(const std::vector<std::vector<int> > & indices)  // deprecated
    {
        indices_.resize(indices.size());
        for(size_t i=0; i<indices.size(); i++)
            indices_[i] = convertVecInt2VecSizet(indices[i]);
    }

    void
    compute (const PointTPtr &output);

    void
    setWeights (const std::vector<std::vector<float> > & weights)
    {
        noise_weights_ = weights;
    }

    void
    setSigmasCombined (const std::vector<std::vector<float> > & sigmas)
    {
        sigmas_combined_ = sigmas;
    }

    void
    setSigmas (const std::vector<std::vector<std::vector<float> > > & sigmas)
    {
        sigmas_ = sigmas;
    }

    void setTransformations(const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & transforms)
    {
        transformations_to_global_ = transforms;
    }
};
}

#endif /* NOISE_MODELS_H_ */
