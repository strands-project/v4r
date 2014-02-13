/**
 *  Copyright (C) 2012  
 *    Andreas Richtsfeld, Johann Prankl, Thomas Mörwald
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1170 Vienn, Austria
 *    ari(at)acin.tuwien.ac.at
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/
 */

/**
 * @file ModelAbstractor.h
 * @author Andreas Richtsfeld
 * @date August 2012
 * @version 0.1
 * @brief Abstract point clouds to parametric surface models
 */


#ifndef MODEL_ABSTRACTOR_H
#define MODEL_ABSTRACTOR_H

#include <stdio.h>

#include "v4r/SurfaceUtils/KinectData.h"
// #include "v4r/PCLAddOns/PCLUtils.h"
#include "v4r/SurfaceUtils/SurfaceModel.hpp"
#include "v4r/SurfaceClustering/ZAdaptiveNormals.hh"
#include "v4r/SurfaceClustering/ClusterNormalsToPlanes.hh"
#include "v4r/SurfaceModeling/SurfaceModeling.hh"

#include <pcl/io/pcd_io.h>

namespace segment
{
  
/**
 * @class ModelAbstractor
 */
class ModelAbstractor
{
private:

public:

private:
  
public:
  ModelAbstractor();
  ~ModelAbstractor();
  
  /** Process a point cloud and return labeled cloud **/
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr
    processPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud);

};

}

#endif
