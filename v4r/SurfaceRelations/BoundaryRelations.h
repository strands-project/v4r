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
 * @file BoundaryRelations.h
 * @author Richtsfeld
 * @date October 2012
 * @version 0.1
 * @brief Calculate boundary relations between patches.
 */

#ifndef SURFACE_BOUNDARY_RELATIONS_H
#define SURFACE_BOUNDARY_RELATIONS_H

#include <stdio.h>

#include <opencv2/opencv.hpp>

#include <pcl/filters/project_inliers.h>
#include <pcl/io/pcd_io.h>

//#include "v4r/SurfaceUtils/SurfaceModel.hpp"


namespace surface
{

class BoundaryRelations
{
public:
  
protected:

private:

  bool computed;
  bool have_cloud;
  bool have_normals;
  bool projectPts;
  bool have_indices;
  
  pcl::PointIndices::Ptr indices;
  
  double max3DDistancePerMeter;                                         ///< Maximum distance for 3D neighbourhood per meter
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;                     ///< Input cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_model;               ///< Input cloud projected to model
  pcl::PointCloud<pcl::Normal>::Ptr normals;                  ///< Normals (set from outside or from surfaces)

  //void projectPts2Model();                                              ///< Project plane points to model

  int width, height;

  void createPatchImage(pcl::PointIndices::Ptr _indices, uchar idx, cv::Mat &patches);
    
public:

  typedef boost::shared_ptr<BoundaryRelations> Ptr;
  
  BoundaryRelations();
  ~BoundaryRelations();
  
  /** Set input point cloud **/
  void setInputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud);
  //sets normals
  void setNormals(pcl::PointCloud<pcl::Normal>::Ptr _normals);

  // sets indices
  void setIndices(pcl::PointIndices::Ptr _indices);
  void setIndices(cv::Rect rect);

  pcl::PointIndices::Ptr getIndices() {return indices;};
  
  /** Compare patches **/
  bool compare(BoundaryRelations::Ptr br);
};

/*************************** INLINE METHODES **************************/

} //--END--

#endif

