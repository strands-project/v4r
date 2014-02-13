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
 * @file SegmentEvaluation.h
 * @author Richtsfeld
 * @date Dezember 2012
 * @version 0.1
 * @brief Evaluation of the segmentation results.
 */

#ifndef SURFACE_SEGMENT_EVALUATION_H
#define SURFACE_SEGMENT_EVALUATION_H

#include <cstdio>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "SurfaceModel.hpp"

namespace surface
{

class SegmentEvaluation
{
public:
  
protected:

private:
  
  bool gpAnno;                                                          ///< Annotation for ground plane is available
  bool consider_nan_values;                                             ///< Consider also ground truth if no depth data is available
  
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcl_cloud_labeled;            ///< Input cloud
  surface::View *view;                                                  ///< Surface models in view
  
public:
  SegmentEvaluation();
  ~SegmentEvaluation();

  /** Consider non-depth groud truth values **/
  void setConsiderNans(bool _c) {consider_nan_values = _c;}

  /** Set input point cloud **/
  void setInputCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &_pcl_cloud) {pcl_cloud_labeled = _pcl_cloud;}
  
  /** Set input surface patches **/
  void setView(surface::View *_view) {view = _view;}

  /** Set ground plane annotation **/
  void setGroundPlaneAnnotation(bool _gpAnno) {gpAnno = _gpAnno;}
  
  /** Compute concave corners of surfaces **/
  void compute();
  
};

/*************************** INLINE METHODES **************************/

} //--END--

#endif

