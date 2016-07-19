#include <v4r/keypoints/uniform_sampling_extractor.h>
#include <pcl/keypoints/uniform_sampling.h>

namespace v4r
{

template<typename PointT>
void
UniformSamplingExtractor<PointT>::compute (pcl::PointCloud<PointT> & keypoints)
{
    pcl::UniformSampling<PointT> us;
    us.setRadiusSearch (sampling_density_);
    us.setInputCloud (input_);
    if(!indices_.empty())
    {
        boost::shared_ptr <std::vector<int> > IndicesPtr (new std::vector<int>);
        *IndicesPtr = indices_;
        us.setIndices(IndicesPtr);
    }
    pcl::PointCloud<int> keypoints_idxes;
    us.compute (keypoints_idxes);

    keypoint_indices_.resize (keypoints_idxes.points.size ());
    for(size_t i=0; i < keypoints_idxes.points.size(); i++)
        keypoint_indices_[i] = keypoints_idxes.points[i];

    pcl::copyPointCloud (*input_, keypoint_indices_, keypoints);

    indices_.clear();
}

template class V4R_EXPORTS UniformSamplingExtractor<pcl::PointXYZ>;
template class V4R_EXPORTS UniformSamplingExtractor<pcl::PointXYZRGB>;
}
