#include <v4r/keypoints/harris3d_keypoint_extractor.h>
#include <pcl/keypoints/harris_3d.h>

namespace v4r
{

template<typename PointT>
void
Harris3DKeypointExtractor<PointT>::compute ()
{
#if PCL_VERSION >= 100702
    typename pcl::search::OrganizedNeighbor<PointT>::Ptr search_method (new pcl::search::OrganizedNeighbor<PointT> ());
    pcl::HarrisKeypoint3D <PointT, pcl::PointXYZI> detector;
    detector.setNonMaxSupression (true);
    detector.setRadiusSearch(param_.search_radius_);
    detector.setThreshold (param_.threshold_);
    detector.setRefine(param_.refine_);
    detector.setSearchMethod(search_method);
    detector.setNormals(normals_);
    detector.setInputCloud (input_);
    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoint_idx (new pcl::PointCloud<pcl::PointXYZI>);
    detector.compute (*keypoint_idx);
    pcl::PointIndicesConstPtr keypoints_indices = detector.getKeypointsIndices ();

    keypoint_indices_.resize( keypoints_indices->indices.size() );
    for(size_t i=0; i < keypoints_indices->indices.size(); i++)
        keypoint_indices_[i] = keypoints_indices->indices[i];

#else
    std::cerr << "HARRIS 3D is not available with keypointindices for this PCL version!" << std::endl;
#endif
}

template class V4R_EXPORTS Harris3DKeypointExtractor<pcl::PointXYZ>;
template class V4R_EXPORTS Harris3DKeypointExtractor<pcl::PointXYZRGB>;
}
