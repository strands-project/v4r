/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_CONVERT_CLOUD_KP_HPP
#define KP_CONVERT_CLOUD_KP_HPP

#include <float.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "v4r/KeypointTools/DataMatrix2D.hpp"
#include "v4r/KeypointTools/PointTypes.hpp"


namespace kp 
{


void convertCloudPCLtoKP(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, kp::DataMatrix2D<kp::PointXYZRGB> &kp_cloud)
{
  kp_cloud.resize(cloud.height, cloud.width);

  for (unsigned i=0; i<cloud.points.size(); i++)
  {
    const pcl::PointXYZRGB &pt = cloud.points[i];
    kp::PointXYZRGB &kp = kp_cloud.data[i];

    kp.pt = pt.getVector4fMap();
    kp.rgb = pt.rgb;
  }
}

void convertCloudKPtoPCL(const kp::DataMatrix2D<kp::PointXYZRGB> &kp_cloud, pcl::PointCloud<pcl::PointXYZRGB> &pcl_cloud)
{
  pcl_cloud.points.resize(kp_cloud.data.size());
  pcl_cloud.width = kp_cloud.cols;
  pcl_cloud.height = kp_cloud.rows;
  pcl_cloud.is_dense = false;

  for (unsigned i=0; i<pcl_cloud.points.size(); i++)
  {
    const kp::PointXYZRGB &kp = kp_cloud.data[i];
    pcl::PointXYZRGB &pt = pcl_cloud.points[i];

    pt.getVector4fMap() = kp.pt;
    pt.rgb = kp.rgb;
  }
}


} //--END--

#endif

