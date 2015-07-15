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

namespace v4r
{
namespace utils
{
template<class PointT>
class NMBasedCloudIntegration
{
private:
    typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
    typedef typename pcl::PointCloud<pcl::Normal>::Ptr PointNormalTPtr;
    std::vector<PointTPtr> input_clouds_;
    std::vector<Eigen::Matrix4f> transformations_to_global_;
    std::vector<std::vector<float> > noise_weights_;
    typename boost::shared_ptr<pcl::octree::OctreePointCloudPointVector<PointT> > octree_;
    std::vector<float> weights_points_in_octree_;
    std::vector<PointNormalTPtr> input_normals_;
    PointNormalTPtr octree_points_normals_;
    PointNormalTPtr output_normals_;
    std::vector<PointTPtr> input_clouds_used_;
    std::vector<std::vector<int> > indices_;

public:
    struct NMparams
    {
        int min_points_per_voxel_;
        float final_resolution_;
        float octree_resolution_;
        float min_weight_;
        float threshold_ss_;
        float max_distance_;
    }nm_params_;

    NMBasedCloudIntegration ();

    void getInputCloudsUsed(std::vector<PointTPtr> & input_clouds_used) const
    {
        input_clouds_used = input_clouds_used_;
    }

    void setThresholdSameSurface(float f)
    {
        nm_params_.threshold_ss_ = f;
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
    setResolution(float r)
    {
        nm_params_.octree_resolution_ = r;
    }

    void setFinalResolution(float r)
    {
        nm_params_.final_resolution_ = r;
    }

    void setMinPointsPerVoxel(int n)
    {
        nm_params_.min_points_per_voxel_ = n;
    }

    void
    setMinWeight(float m_w)
    {
        nm_params_.min_weight_ = m_w;
    }

    void
    setIndices(const std::vector<std::vector<int> > & indices)
    {
        indices_ = indices;
    }

    void
    compute (const PointTPtr &output);

    void
    setWeights (const std::vector<std::vector<float> > & weights)
    {
        noise_weights_ = weights;
    }

    void setTransformations(const std::vector<Eigen::Matrix4f> & transforms)
    {
        transformations_to_global_ = transforms;
    }
};
}
}

#endif /* NOISE_MODELS_H_ */
