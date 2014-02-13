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
 * @file Segmenter.h
 * @author Andreas Richtsfeld
 * @date July 2012
 * @version 0.1
 * @brief Segment images
 */

#ifndef V4R_SEGMENT_SEGMENTER_H
#define V4R_SEGMENT_SEGMENTER_H

#include <pcl/point_types.h>

#include "v4r/SurfaceUtils/SurfaceModel.hpp"
#include "v4r/SurfaceUtils/ContourDetector.h"
#include "v4r/SurfaceRelations/StructuralRelations.h"
#include "v4r/SurfaceRelations/AssemblyRelations.h"

namespace segment
{

  /**
   * @class Segmenter
   */
  class Segmenter
  {
  private:
    bool useStructuralLevel;    ///< Use structural level svm
    bool useAssemblyLevel;      ///< Use assembly
    std::string model_path;     ///< Path to the svm model and scaling files
    int detail;                 ///< Degree of detail for pre-segmenter
    surface::View view;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

  private:
    void computeNormals ();
    void clusterNormals ();
    void computeSurfaces ();
    void computeContours();
    void computeStructuralRelations ();
    void computeAssemblyRelations ();
    void classifyRelations ();
    void computeGraphCut ();

  public:
    Segmenter (std::string _model = "model/");
    ~Segmenter ();

    void setInputCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud);
    void compute();

    /** Use assembly level for processing */
    void setAssemblyLevel(bool _on) {useAssemblyLevel = _on;}

    /** Change detail of pre-segmentation */
    void  setDetail(int _detail = 0) {detail = _detail;}

    /** Get the labeled point cloud */
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr getLabels ();

    /** Get a vector of segment indices */
    std::vector<pcl::PointIndices> getSegments ();

    /** Get surfaces */
    std::vector<surface::SurfaceModel::Ptr> getSurfaces ();
  };

}

#endif
