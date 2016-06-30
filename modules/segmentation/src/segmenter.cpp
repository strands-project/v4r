#include <v4r/segmentation/segmenter.h>

namespace v4r
{

template<typename PointT> Segmenter<PointT>::~Segmenter(){}

template class V4R_EXPORTS Segmenter<pcl::PointXYZRGB>;
}
