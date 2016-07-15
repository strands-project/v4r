#include <pcl/point_types.h>
#include <pcl/impl/instantiate.hpp>
#include <v4r/recognition/source.h>

namespace v4r
{

template <typename PointT> Source<PointT>::~Source(){}

template class V4R_EXPORTS Source<pcl::PointXYZ>;
template class V4R_EXPORTS Source<pcl::PointXYZRGB>;

}
