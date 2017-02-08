#include <pcl/impl/instantiate.hpp>
#include <v4r/recognition/object_hypothesis.h>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>


namespace v4r
{

#define PCL_INSTANTIATE_ObjectHypothesis(T) template class V4R_EXPORTS ObjectHypothesis<T>;
PCL_INSTANTIATE(ObjectHypothesis, (pcl::PointXYZRGB) )

#define PCL_INSTANTIATE_ObjectHypothesesGroup(T) template class V4R_EXPORTS ObjectHypothesesGroup<T>;
PCL_INSTANTIATE(ObjectHypothesesGroup, (pcl::PointXYZRGB) )

}


