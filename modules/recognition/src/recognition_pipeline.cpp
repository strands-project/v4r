#include <pcl/impl/instantiate.hpp>
#include <v4r/recognition/recognition_pipeline.h>

namespace v4r
{

template<typename PointT>
std::vector<std::pair<std::string,float> > RecognitionPipeline<PointT>::elapsed_time_;

#define PCL_INSTANTIATE_RecognitionPipeline(T) template class V4R_EXPORTS RecognitionPipeline<T>;
PCL_INSTANTIATE(RecognitionPipeline, (pcl::PointXYZRGB))

}
