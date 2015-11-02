#include <pcl/point_types.h>
#include <pcl/impl/instantiate.hpp>
#include <v4r/recognition/source.h>

template class V4R_EXPORTS v4r::Source<pcl::PointXYZ>;
template class V4R_EXPORTS v4r::Source<pcl::PointXYZRGB>;
template class V4R_EXPORTS v4r::Model<pcl::PointXYZ>;
template class V4R_EXPORTS v4r::Model<pcl::PointXYZRGB>;
