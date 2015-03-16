/*
 * visibility_reasoning.cpp
 *
 *  Created on: Mar 19, 2013
 *      Author: aitor
 */

#include "pcl/point_types.h"
#include "VisibilityReasoning.hpp"

template class v4r::registration::VisibilityReasoning<pcl::PointXYZ>;
template class v4r::registration::VisibilityReasoning<pcl::PointXYZRGB>;
template class v4r::registration::VisibilityReasoning<pcl::PointNormal>;
template class v4r::registration::VisibilityReasoning<pcl::PointXYZRGBNormal>;
