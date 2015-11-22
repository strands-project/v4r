/*
 * noise_models.h
 *
 *  Created on: Oct 28, 2013
 *      Author: aitor
 */

#ifndef NMBasedCloudIntegration_H
#define NMBasedCloudIntegration_H

#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/octree/impl/octree_iterator.hpp>

#include <v4r/core/macros.h>
#include "v4r/common/miscellaneous.h"

namespace v4r
{
template<class PointT>
class V4R_EXPORTS NMBasedCloudIntegration
{
public:
    class V4R_EXPORTS Parameter
    {
    public:
        int min_points_per_voxel_;
        float final_resolution_;
        float octree_resolution_;
        float min_weight_;
        float threshold_ss_;
        float max_distance_;
        Parameter(
                int min_points_per_voxel=0,
                float final_resolution = 0.001f,
                float octree_resolution =  0.005f,
                float min_weight = 0.9f,
                float threshold_ss = 0.003f,
                float max_distance = 5.f) :
            min_points_per_voxel_(min_points_per_voxel),
            final_resolution_(final_resolution),
            octree_resolution_(octree_resolution),
            min_weight_ (min_weight),
            threshold_ss_(threshold_ss),
            max_distance_(max_distance)
        {

        }
    };

private:
    typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
    typedef typename pcl::PointCloud<pcl::Normal>::Ptr PointNormalTPtr;
    std::vector<PointTPtr> input_clouds_;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transformations_to_global_;
    std::vector<std::vector<float> > noise_weights_;
    typename boost::shared_ptr<pcl::octree::OctreePointCloudPointVector<PointT> > octree_;
    std::vector<float> weights_points_in_octree_;
    std::vector<PointNormalTPtr> input_normals_;
    PointNormalTPtr octree_points_normals_;
    PointNormalTPtr output_normals_;
    std::vector<PointTPtr> input_clouds_used_;
    std::vector<std::vector<size_t> > indices_;

public:
    Parameter param_;
    NMBasedCloudIntegration (const Parameter &p=Parameter());

    void getInputCloudsUsed(std::vector<PointTPtr> & input_clouds_used) const
    {
        input_clouds_used = input_clouds_used_;
    }

    void setThresholdSameSurface(float f)
    {
        param_.threshold_ss_ = f;
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
    setResolution(float r)  // deprecated
    {
        param_.octree_resolution_ = r;
    }

    void setFinalResolution(float r)  // deprecated
    {
        param_.final_resolution_ = r;
    }

    void setMinPointsPerVoxel(int n)  // deprecated
    {
        param_.min_points_per_voxel_ = n;
    }

    void
    setMinWeight(float m_w)  // deprecated
    {
        param_.min_weight_ = m_w;
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

    void setTransformations(const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & transforms)
    {
        transformations_to_global_ = transforms;
    }
};
}

#endif /* NOISE_MODELS_H_ */
