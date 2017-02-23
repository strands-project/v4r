/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma
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

#ifndef V4R_NORMAL_ESTIMATOR_H_
#define V4R_NORMAL_ESTIMATOR_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <v4r/core/macros.h>

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS NormalEstimator
{
protected:
    typename pcl::PointCloud<PointT>::ConstPtr input_; ///< input cloud
    std::vector<int> indices_;  ///< indices of the segmented object (extracted keypoints outside of this will be neglected)

public:
    virtual ~NormalEstimator() = 0;

    /**
     * @brief setInputCloud
     * @param input input cloud
     */
    void
    setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr & input)
    {
        input_ = input;
    }

    /**
     * @brief setIndices
     * @param indices indices of the segmented object (extracted keypoints outside of this will be neglected)
     */
    virtual void
    setIndices(const std::vector<int> &indices)
    {
        indices_ = indices;
    }

    /**
     * @brief getNormalEstimatorType
     * @return unique type id of normal estimator(as stated in keypoint/types.h)
     */
    virtual int getNormalEstimatorType() const = 0;

    /**
     * @brief compute
     * @param keypoints
     */
    virtual void
    compute (pcl::PointCloud<PointT> & keypoints) = 0;

    typedef boost::shared_ptr< NormalEstimator<PointT> > Ptr;
    typedef boost::shared_ptr< NormalEstimator<PointT> const> ConstPtr;
};


template<typename PointT, typename PointOutT>
class V4R_EXPORTS PreProcessorAndNormalEstimator
{
private:
    typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;

public:

    bool compute_mesh_resolution_;
    bool do_voxel_grid_;
    bool remove_outliers_;

    //this values are used when CMR=false
    float grid_resolution_;
    float normal_radius_;

    //this are used when CMR=true
    float factor_normals_;
    float factor_voxel_grid_;
    float min_n_radius_;
    bool force_unorganized_;

    bool only_on_indices_;
    pcl::PointIndices indices_;

    PreProcessorAndNormalEstimator () : compute_mesh_resolution_(false), do_voxel_grid_ (false), remove_outliers_ (false),
        grid_resolution_(0.01f), normal_radius_(0.02f),
        factor_normals_(1), factor_voxel_grid_(1), min_n_radius_ (16), force_unorganized_(false), only_on_indices_ (false)
    { }

    void
    setIndices(const std::vector<int> & indices)
    {
        only_on_indices_ = true;
        indices_.indices = indices;
    }

    void
    setForceUnorganized(bool set)
    {
        force_unorganized_ = set;
    }

    void
    setMinNRadius(float r)
    {
        min_n_radius_ = r;
    }

    void
    setFactorsForCMR (float f1, float f2)
    {
        factor_voxel_grid_ = f1;
        factor_normals_ = f2;
    }

    void
    setValuesForCMRFalse (float f1, float f2)
    {
        grid_resolution_ = f1;
        normal_radius_ = f2;
    }

    void
    setDoVoxelGrid (bool b)
    {
        do_voxel_grid_ = b;
    }

    void
    setRemoveOutliers (bool b)
    {
        remove_outliers_ = b;
    }

    void
    setCMR (bool b)
    {
        compute_mesh_resolution_ = b;
    }

    void
    estimate (const typename pcl::PointCloud<PointT>::ConstPtr & in, PointInTPtr & out, pcl::PointCloud<pcl::Normal>::Ptr & normals);
};
}

#endif
