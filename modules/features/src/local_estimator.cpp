

#include "v4r/features/local_estimator.h"

template class v4r::LocalEstimator<struct pcl::PointXYZ, struct pcl::Histogram<352> >;
template class v4r::UniformSamplingExtractor<struct pcl::PointXYZ>;
