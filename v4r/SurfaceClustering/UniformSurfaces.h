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
 * @file UniformSurfaces.h
 * @author Andreas Richtsfeld
 * @date December 2012
 * @version 0.1
 * @brief Get uniform patches from depth image by detection of discontinuities.
 */

#ifndef SURFACE_UNIFORM_SURFACES_H
#define SURFACE_UNIFORM_SURFACES_H

#include <iostream>
#include <vector>
#include <queue>
#include <set>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#ifdef DEBUG
  #include "v4r/TomGinePCL/tgTomGineThreadPCL.h"
#endif

#include "v4r/PCLAddOns/PCLUtils.h"
#include "v4r/SurfaceUtils/SurfaceModel.hpp"
#include "v4r/RGBDSegment/RGBDSegment.h"
#include "v4r/SurfaceClustering/ZAdaptiveNormals.hh"
#include "v4r/SurfaceRelations/Surf.h"
   
#include <pcl/point_types.h>
#include <pcl/features/organized_edge_detection.h>

namespace surface
{


/**
 * Uniform surfaces
 */
class UniformSurfaces
{
public:

private:
  TomGine::tgEngine *m_engine;
  TomGine::tgImageProcessor* m_ip;
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud;             /// original pcl cloud
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcl_cloud_labeled;    /// clustered point cloud
  pcl::PointCloud<pcl::Normal>::Ptr pcl_normals;                /// point cloud normals
  cv::Mat_<cv::Vec4f> cv_normals;                               /// normals in opencv-style
  
  surface::View *view;                                          /// view to scene

  bool initialized;
  void Initialize();
  
  void NormaliseDepthAndCurvature(const cv::Mat_<cv::Vec4f> &cloud, 
                                  const cv::Mat_<cv::Vec4f> &normals,
                                  cv::Mat_<float> &depth, cv::Mat_<float> &curvature);
  
  void EdgeDetectionRGBDC(TomGine::tgEngine* engine,
                          TomGine::tgImageProcessor* m_ip,
                          const cv::Mat_<cv::Vec3b> &image, 
                          const cv::Mat_<float> &depth,
                          cv::Mat_<float> &curvature, 
                          const cv::Mat_<uchar> &mask, 
                          cv::Mat_<float> &color_edges,
                          cv::Mat_<float> &depth_edges, 
                          cv::Mat_<float> &curvature_edges, 
                          cv::Mat_<float> &mask_edges, 
                          cv::Mat_<float> &edges);
  
  void RecursiveClustering(const cv::Mat_<float> &_edges);
  void ReassignMaskPoints(cv::Mat_<float> &_edges);
  
  void CopyToLabeledCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_in, 
                          pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &_out);

  void CreateView();
  
  inline int GetIdx(short x, short y);
  inline short X(int idx);
  inline short Y(int idx);


public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

#ifdef DEBUG
  cv::Mat dbg;
  cv::Ptr<TomGine::tgTomGineThread> dbgWin;
#endif
  
//   UniformSurfaces(Parameter p=Parameter());
  UniformSurfaces();
  ~UniformSurfaces();

  /** Set parameters for plane estimation **/
//   void setParameter(Parameter p);

  /** Set input cloud **/
  void setInputCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_pcl_cloud);

  /** Set input normals **/
  void setInputNormals(const pcl::PointCloud<pcl::Normal>::Ptr &_pcl_normals);

  /** Compute planes by surface normal grouping **/
  void compute();
  
  /** Get view with segmented surfaces **/
  void getView(surface::View *_view) {view = _view;}
};



/*********************** INLINE METHODES **************************/


inline int UniformSurfaces::GetIdx(short x, short y)
{
  return y*pcl_cloud->width+x; 
}

inline short UniformSurfaces::X(int idx)
{
  return idx%pcl_cloud->width;
}

inline short UniformSurfaces::Y(int idx)
{
  return idx/pcl_cloud->width;
}

}

#endif

