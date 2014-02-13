/*
 * visibility_reasoning.cpp
 *
 *  Created on: Mar 19, 2013
 *      Author: aitor
 */

#include "pcl/point_types.h"
#include <faat_pcl/object_modelling/visibility_reasoning.h>
#include <faat_pcl/object_modelling/impl/visibility_reasoning.hpp>

template class faat_pcl::object_modelling::VisibilityReasoning<pcl::PointXYZ>;
template class faat_pcl::object_modelling::VisibilityReasoning<pcl::PointXYZRGB>;
template class faat_pcl::object_modelling::VisibilityReasoning<pcl::PointXYZRGBNormal>;
template class faat_pcl::object_modelling::VisibilityReasoning<pcl::PointNormal>;
