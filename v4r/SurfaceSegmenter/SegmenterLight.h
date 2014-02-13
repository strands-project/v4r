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
 * @file SegmenterLight.h
 * @author Andreas Richtsfeld
 * @date Januray 2013
 * @version 0.1
 * @brief Segment images efficiently
 */

#ifndef V4R_SEGMENT_SEGMENTERLIGHT_H
#define V4R_SEGMENT_SEGMENTERLIGHT_H

#include <pcl/point_types.h>

#include "v4r/SurfaceUtils/SurfaceModel.hpp"
#include "v4r/SurfaceUtils/ContourDetector.h"
#include "v4r/SurfaceRelations/StructuralRelationsLight.h"
#include "v4r/SurfaceClustering/ZAdaptiveNormals.hh"
#include "v4r/SurfaceClustering/ClusterNormalsToPlanes.hh"
#include "v4r/SurfaceModeling/SurfaceModeling.hh"
#include "v4r/svm/SVMPredictorSingle.h"
#include "v4r/GraphCut/GraphCut.h"

namespace segment
{

  /**
   * @class SegmenterLight
   */
  class SegmenterLight
  {
  private:

    bool useStructuralLevel;    ///< Use structural level svm
    bool fast;			///< Set fast processing without modelling
    std::string model_path;     ///< path to the svm model and scaling files
    int detail;                 ///< Degree of detail for pre-segmenter

  public:

  private:

  public:
    SegmenterLight (std::string _model_path = "model/");
    ~SegmenterLight ();

    void
    computeNormals (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in, pcl::PointCloud<pcl::Normal>::Ptr &normals_out);

    void
    computePlanes (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in, pcl::PointCloud<pcl::Normal>::Ptr &normals_in,
                   std::vector<surface::SurfaceModel::Ptr> &surfaces_out);

    void
    computeSurfaces (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                     std::vector<surface::SurfaceModel::Ptr> &surfaces_in_out);

    void
    computeObjects (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                    std::vector<surface::SurfaceModel::Ptr> &surfaces_in_out,
                    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &cloud_out);

    /** Process a point cloud and return labeled cloud **/
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr
    processPointCloud (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud);

    /** Process a point cloud and return vector of segment indices **/
    std::vector<pcl::PointIndices>
    processPointCloudV (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud);

    /** Change detail of pre-segmentation **/
    void
    setDetail(int _detail = 0) {detail = _detail;}

    /** Change detail of pre-segmentation **/
    void
    setFast(bool _fast) {fast = _fast;}
  };

}

#endif
