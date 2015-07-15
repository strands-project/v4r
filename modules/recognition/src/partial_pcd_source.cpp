#include <v4r/recognition/impl/partial_pcd_source.hpp>

template class v4r::rec_3d_framework::PartialPCDSource<struct pcl::PointXYZRGBNormal, struct pcl::PointXYZRGB, struct pcl::PointXYZRGB>;
//template class faat_pcl::rec_3d_framework::PartialPCDSource<struct pcl::PointXYZRGBNormal, struct pcl::PointXYZ, struct pcl::PointXYZ>;
template class v4r::rec_3d_framework::PartialPCDSource<struct pcl::PointXYZRGBNormal, struct pcl::PointXYZRGBA, struct pcl::PointXYZRGBA>;
//template class faat_pcl::rec_3d_framework::PartialPCDSource<struct pcl::PointNormal, struct pcl::PointXYZ, struct pcl::PointXYZ>;

