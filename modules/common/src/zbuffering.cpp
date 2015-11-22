#include <v4r/common/zbuffering.h>
#include <v4r/common/impl/zbuffering.hpp>

template class V4R_EXPORTS v4r::ZBuffering<pcl::PointXYZ,pcl::PointXYZ>;
template class V4R_EXPORTS v4r::ZBuffering<pcl::PointXYZ,pcl::PointXYZRGB>;
template class V4R_EXPORTS v4r::ZBuffering<pcl::PointXYZ,pcl::PointXYZRGBA>;
template class V4R_EXPORTS v4r::ZBuffering<pcl::PointXYZRGB,pcl::PointXYZ>;
template class V4R_EXPORTS v4r::ZBuffering<pcl::PointXYZRGB,pcl::PointXYZRGB>;
template class V4R_EXPORTS v4r::ZBuffering<pcl::PointXYZRGB,pcl::PointXYZRGBA>;
template class V4R_EXPORTS v4r::ZBuffering<pcl::PointXYZRGBA,pcl::PointXYZRGBA>;
template class V4R_EXPORTS v4r::ZBuffering<pcl::PointXYZRGBA,pcl::PointXYZRGB>;
template class V4R_EXPORTS v4r::ZBuffering<pcl::PointXYZRGBA,pcl::PointXYZ>;
