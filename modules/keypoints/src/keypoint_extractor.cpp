#include <v4r/keypoints/keypoint_extractor.h>

namespace v4r
{
template<typename PointT> KeypointExtractor<PointT>::~KeypointExtractor(){}

template class V4R_EXPORTS KeypointExtractor<pcl::PointXYZ>;
template class V4R_EXPORTS KeypointExtractor<pcl::PointXYZRGB>;
}


