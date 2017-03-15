#include <v4r/keypoints/iss_keypoint_extractor.h>
#include <pcl/common/angles.h>
#include <pcl/keypoints/iss_3d.h>

namespace v4r
{

template<typename PointT>
void
IssKeypointExtractor<PointT>::compute ()
{
    typename pcl::search::OrganizedNeighbor<PointT>::Ptr search_method (new pcl::search::OrganizedNeighbor<PointT> ());
    pcl::ISSKeypoint3D<PointT, PointT> iss_detector;
    iss_detector.setSearchMethod (search_method);
    iss_detector.setSalientRadius (param_.salient_radius_);
    iss_detector.setNonMaxRadius (param_.non_max_radius_);

    if (param_.with_border_estimation_)
    {
        iss_detector.setNormalRadius (param_.normal_radius_);
        iss_detector.setBorderRadius (param_.border_radius_);
    }

    iss_detector.setThreshold21 (param_.gamma_21_);
    iss_detector.setThreshold32 (param_.gamma_32_);
    iss_detector.setMinNeighbors (param_.min_neighbors_);
    iss_detector.setNumberOfThreads (param_.threads_);
    iss_detector.setAngleThreshold ( pcl::deg2rad(param_.angle_thresh_deg_) );
    iss_detector.setInputCloud (input_);
    iss_detector.setNormals(normals_);
    keypoints_.reset(new pcl::PointCloud<PointT>);
    iss_detector.compute (*keypoints_);

#if PCL_VERSION >= 100702
    pcl::PointIndicesConstPtr keypoints_idxes = iss_detector.getKeypointsIndices();
    keypoint_indices_.resize (keypoints_idxes->indices.size ());
    for(size_t i=0; i < keypoint_indices_.size(); i++)
        keypoint_indices_[i] = keypoints_idxes->indices[i];
#else
    std::cerr << "Getkeypointsindices is not available for this PCL version!" << std::endl;
#endif

    indices_.clear();
}

template class V4R_EXPORTS IssKeypointExtractor<pcl::PointXYZ>;
template class V4R_EXPORTS IssKeypointExtractor<pcl::PointXYZRGB>;

}
