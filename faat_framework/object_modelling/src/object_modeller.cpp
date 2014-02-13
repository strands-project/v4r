/*
 * object_modeller.cpp
 *
 *  Created on: Mar 15, 2013
 *      Author: aitor
 */

#include "pcl/point_types.h"
#include <faat_pcl/object_modelling/object_modeller.h>
#include <faat_pcl/object_modelling/impl/object_modeller.hpp>

template class faat_pcl::object_modelling::ObjectModeller<flann::L1, pcl::PointXYZ, pcl::PointNormal>;
template class faat_pcl::object_modelling::ObjectModeller<flann::L1, pcl::PointXYZRGB, pcl::PointXYZRGBNormal>;
