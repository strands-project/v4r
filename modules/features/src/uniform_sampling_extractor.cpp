#include <v4r/features/uniform_sampling_extractor.h>
#include <v4r/features/uniform_sampling.h>
#include <v4r/common/miscellaneous.h>
#include <pcl/common/centroid.h>

namespace v4r
{

template<typename PointT>
void
UniformSamplingExtractor<PointT>::filterPlanar (const PointInTPtr & input, std::vector<int> &kp_idx)
{
    typename pcl::search::Search<PointT>::Ptr tree; //create a search object

    if (input->isOrganized () && !force_unorganized_)
        tree.reset (new pcl::search::OrganizedNeighbor<PointT> ());
    else
        tree.reset (new pcl::search::KdTree<PointT> (false));

    tree->setInputCloud (input);

    size_t kept = 0;
    for (size_t i = 0; i < kp_idx.size (); i++)
    {
        std::vector<int> neighborhood_indices;
        std::vector<float> neighborhood_dist;

        if (tree->radiusSearch (kp_idx[i], radius_, neighborhood_indices, neighborhood_dist))
        {
            EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
            Eigen::Vector4f xyz_centroid;
            EIGEN_ALIGN16 Eigen::Vector3f eigenValues;
            EIGEN_ALIGN16 Eigen::Matrix3f eigenVectors;

            //compute planarity of the region
            computeMeanAndCovarianceMatrix (*input, neighborhood_indices, covariance_matrix, xyz_centroid);
            pcl::eigen33 (covariance_matrix, eigenVectors, eigenValues);

            float eigsum = eigenValues.sum ();
            if (!pcl_isfinite(eigsum))
                PCL_ERROR("Eigen sum is not finite\n");

            float t_planar = threshold_planar_;

            if(z_adaptative_)
                t_planar *= (1 + (std::max(input->points[kp_idx[i]].z,1.f) - 1.f));

            //if ((fabs (eigenValues[0] - eigenValues[1]) < 1.5e-4) || (eigsum != 0 && fabs (eigenValues[0] / eigsum) > 1.e-2))
            if ((fabs (eigenValues[0] - eigenValues[1]) < 1.5e-4) || (eigsum != 0 && fabs (eigenValues[0] / eigsum) > t_planar))
            {
                //region is not planar, add to filtered keypoint
                kp_idx[kept] = kp_idx[i];
                kept++;
            }
        }
    }
    kp_idx.resize (kept);
}

template<typename PointT>
void
UniformSamplingExtractor<PointT>::compute (pcl::PointCloud<PointT> & keypoints)
{
    compute(keypoint_indices_);
    pcl::copyPointCloud (*input_, keypoint_indices_, keypoints);
}

template<typename PointT>
void
UniformSamplingExtractor<PointT>::compute (std::vector<int> & indices)
{
    pcl::UniformSampling<PointT> keypoint_extractor;
    keypoint_extractor.setRadiusSearch (sampling_density_);
    keypoint_extractor.setInputCloud (input_);

    pcl::PointCloud<int> keypoints_idxes;
    keypoint_extractor.compute (keypoints_idxes);

    std::vector<bool> obj_mask;
    if(indices_.empty())
        obj_mask.resize(input_->width * input_->height, true);
    else
        obj_mask = createMaskFromIndices(indices_, input_->width * input_->height);

    keypoint_indices_.resize (keypoints_idxes.points.size ());
    size_t kept = 0;
    for(size_t i=0; i < keypoints_idxes.points.size(); i++)
    {
        int idx = keypoints_idxes.points[i];
        if( obj_mask[idx] && (!pcl_isfinite(max_distance_) || input_->points[idx].z < max_distance_) )
        {
            keypoint_indices_[kept] = idx;
            kept++;
        }
    }
    keypoint_indices_.resize(kept);

    if (filter_planar_)
        filterPlanar (input_, keypoint_indices_);

    indices = keypoint_indices_;
    indices_.clear();
}

template class V4R_EXPORTS UniformSamplingExtractor<pcl::PointXYZ>;
template class V4R_EXPORTS UniformSamplingExtractor<pcl::PointXYZRGB>;
}
