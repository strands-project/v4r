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
 * @file CornerDetector.h
 * @author Richtsfeld
 * @date October 2012
 * @version 0.1
 * @brief Detect the corners in the segmented image.
 */

#ifndef SURFACE_CORNER_DETECTOR_H
#define SURFACE_CORNER_DETECTOR_H

#include <vector>
#include <stdio.h>

#include "v4r/PCLAddOns/PCLCommonHeaders.h"
#include "v4r/PCLAddOns/PCLUtils.h"
#include "v4r/SurfaceUtils/SurfaceModel.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>


namespace surface
{

class CornerDetector
{
public:
  
protected:

private:

  bool have_cloud;
  bool have_view;
  
  std::vector<surface::Edgel> edgels;                                   ///< detected edgels
  std::vector<surface::Corner> corners;                                 ///< detected corners
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud;                     ///< Input cloud
  surface::View *view;                                                  ///< Surface models
  
  void RecursiveClustering(int x, int y, 
                           int id_0, int id_1, 
                           bool horizontal, Edge &_ec);

  inline int GetIdx(short x, short y);
  inline short X(int idx);
  inline short Y(int idx);
  
public:
  CornerDetector();
  ~CornerDetector();
  
  /** Set input point cloud **/
  void setInputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_pcl_cloud);
  
  /** Set input surface patches **/
  void setView(surface::View *_view);

  /** Compute corners between surfaces **/
  void compute();

};


/*************************** INLINE METHODES **************************/
/** Return index for coordinates x,y **/
inline int CornerDetector::GetIdx(short x, short y)
{
  return y*pcl_cloud->width + x;
}

/** Return x coordinate for index **/
inline short CornerDetector::X(int idx)
{
  return idx%pcl_cloud->width;
}

/** Return y coordinate for index **/
inline short CornerDetector::Y(int idx)
{
  return idx/pcl_cloud->width;
}


} //--END--

#endif

