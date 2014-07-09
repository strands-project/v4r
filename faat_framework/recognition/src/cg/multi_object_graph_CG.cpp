#include "pcl/point_types.h"
#include "faat_pcl/recognition/cg/multi_object_graph_CG.h"
#include "faat_pcl/recognition/impl/cg/multi_object_graph_CG.hpp"

template class faat_pcl::MultiObjectGraphGeometricConsistencyGrouping<pcl::PointXYZ,pcl::PointXYZ>;
template class faat_pcl::MultiObjectGraphGeometricConsistencyGrouping<pcl::PointXYZRGB,pcl::PointXYZRGB>;

/*template class faat_pcl::GraphGeometricConsistencyGrouping<pcl::PointXYZI,pcl::PointXYZI>;
template class faat_pcl::GraphGeometricConsistencyGrouping<pcl::PointXYZRGB,pcl::PointXYZRGB>;
template class faat_pcl::GraphGeometricConsistencyGrouping<pcl::PointXYZRGBA,pcl::PointXYZRGBA>;
template class faat_pcl::GraphGeometricConsistencyGrouping<pcl::PointNormal,pcl::PointNormal>;
template class faat_pcl::GraphGeometricConsistencyGrouping<pcl::PointXYZRGBNormal,pcl::PointXYZRGBNormal>;*/
