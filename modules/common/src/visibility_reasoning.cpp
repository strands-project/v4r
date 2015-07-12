/*
 * visibility_reasoning.cpp
 *
 *  Created on: Mar 19, 2013
 *      Author: aitor
 */

#include "pcl/point_types.h"
//#include <faat_pcl/registration/visibility_reasoning.h>
#include <v4r/common/impl/visibility_reasoning.hpp>

template class v4r::common::VisibilityReasoning<pcl::PointXYZ>;
template class v4r::common::VisibilityReasoning<pcl::PointXYZRGB>;
template class v4r::common::VisibilityReasoning<pcl::PointNormal>;
template class v4r::common::VisibilityReasoning<pcl::PointXYZRGBNormal>;

