#include <pcl/impl/instantiate.hpp>
#include <v4r/recognition/recognition_pipeline.h>

namespace v4r
{

#define PCL_INSTANTIATE_RecognitionPipeline(T) template class V4R_EXPORTS RecognitionPipeline<T>;
PCL_INSTANTIATE(RecognitionPipeline, (pcl::PointXYZRGB))

}
