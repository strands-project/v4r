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
 * @file ModelAbstractor.cpp
 * @author Andreas Richtsfeld
 * @date August 2012
 * @version 0.1
 * @brief Abstract point clouds to parametric surface models
 */


#include "ModelAbstractor.h"

namespace segment
{

/* --------------- ModelAbstractor --------------- */

ModelAbstractor::ModelAbstractor()
{}

ModelAbstractor::~ModelAbstractor()
{}

pcl::PointCloud<pcl::PointXYZRGBL>::Ptr
ModelAbstractor::processPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud)
{
  surface::View view;
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr result(new pcl::PointCloud<pcl::PointXYZRGBL>);
  pcl::copyPointCloud(*pcl_cloud, *result);
  
  // calcuate normals
  view.normals.reset(new pcl::PointCloud<pcl::Normal>);
  surface::ZAdaptiveNormals<pcl::PointXYZRGB>::Parameter za_param;
  za_param.adaptive = true;
  surface::ZAdaptiveNormals<pcl::PointXYZRGB> nor(za_param);
  nor.setInputCloud(pcl_cloud);
  nor.compute();
  nor.getNormals(view.normals);
  
  // adaptive clustering
  surface::ClusterNormalsToPlanes::Parameter param;
  param.adaptive = true;
  surface::ClusterNormalsToPlanes clusterNormals(param);
  clusterNormals.setInputCloud(pcl_cloud);
  clusterNormals.setView(&view);
  clusterNormals.setPixelCheck(true, 5);
  clusterNormals.compute();
  
  // model abstraction
  pcl::on_nurbs::SequentialFitter::Parameter nurbsParams;
  nurbsParams.order = 3;
  nurbsParams.refinement = 0;
  nurbsParams.iterationsQuad = 0;
  nurbsParams.iterationsBoundary = 0;
  nurbsParams.iterationsAdjust = 0;
  nurbsParams.iterationsInterior = 3;
  nurbsParams.forceBoundary = 100.0;
  nurbsParams.forceBoundaryInside = 300.0;
  nurbsParams.forceInterior = 1.0;
  nurbsParams.stiffnessBoundary = 0.1;
  nurbsParams.stiffnessInterior = 0.1;
  nurbsParams.resolution = 16; 
  surface::SurfaceModeling::Parameter sfmParams;
  sfmParams.nurbsParams = nurbsParams;
  sfmParams.sigmaError = 0.003;
  sfmParams.kappa1 = 0.008;
  sfmParams.kappa2 = 1.0;
  sfmParams.planePointsFixation = 8000;
  sfmParams.z_max = 0.01;
  surface::SurfaceModeling surfModeling(sfmParams);
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  surfModeling.setIntrinsic(525., 525., 320., 240.);
  surfModeling.setExtrinsic(pose);
  surfModeling.setInputCloud(pcl_cloud);
  surfModeling.setView(&view);
  surfModeling.compute();
  
  for(unsigned i=0; i<view.surfaces.size(); i++) {
    for(unsigned j=0; j<view.surfaces[i]->indices.size(); j++) {
      result->points[view.surfaces[i]->indices[j]].label = view.surfaces[i]->label;
    }
  }
  return result;
}

} // end segment
