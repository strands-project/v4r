/**
 * $Id$
 * 
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
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
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl, Aitor Aldoma
 *
 */

#ifndef KP_CONVERT_NORMALS_HPP
#define KP_CONVERT_NORMALS_HPP

#include <float.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/common/PointTypes.h>


namespace v4r 
{


inline void convertNormals(const v4r::DataMatrix2D<Eigen::Vector3f> &kp_normals, pcl::PointCloud<pcl::Normal> &pcl_normals)
{
  pcl_normals.points.resize(kp_normals.data.size());
  pcl_normals.width = kp_normals.cols;
  pcl_normals.height = kp_normals.rows;
  pcl_normals.is_dense = false;

  for (unsigned i=0; i<pcl_normals.points.size(); i++)
  {
    pcl_normals.points[i].getNormalVector3fMap() = kp_normals.data[i];
  }
}

inline void convertNormals(const pcl::PointCloud<pcl::Normal> &pcl_normals, v4r::DataMatrix2D<Eigen::Vector3f> &kp_normals)
{
  kp_normals.data.resize(pcl_normals.points.size());
  kp_normals.cols = pcl_normals.width;
  kp_normals.rows = pcl_normals.height;

  for (unsigned i=0; i<pcl_normals.points.size(); i++)
  {
    kp_normals.data[i] = pcl_normals.points[i].getNormalVector3fMap();
  }


}


} //--END--
#endif

