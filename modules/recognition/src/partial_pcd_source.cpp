#include <v4r/recognition/impl/partial_pcd_source.hpp>

template class V4R_EXPORTS v4r::PartialPCDSource<struct pcl::PointXYZRGBNormal, struct pcl::PointXYZRGB, struct pcl::PointXYZRGB>;
//template class V4R_EXPORTS v4r::PartialPCDSource<struct pcl::PointXYZRGBNormal, struct pcl::PointXYZ, struct pcl::PointXYZ>;
template class V4R_EXPORTS v4r::PartialPCDSource<struct pcl::PointXYZRGBNormal, struct pcl::PointXYZRGBA, struct pcl::PointXYZRGBA>;
//template class V4R_EXPORTS v4r::PartialPCDSource<struct pcl::PointNormal, struct pcl::PointXYZ, struct pcl::PointXYZ>;

