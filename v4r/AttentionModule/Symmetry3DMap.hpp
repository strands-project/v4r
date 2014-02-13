#ifndef SYMMETRY3D_HPP
#define SYMMETRY3D_HPP

#include <opencv2/opencv.hpp>

#include <iostream>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/common/intersections.h>

#include "v4r/AttentionModule/AttentionModule.hpp"
#include "v4r/EPUtils/EPUtils.hpp"

namespace AttentionModule
{

/**
 * parameters for 3Dsymmetry map saliency map
 * */

struct Symmetry3DMapParameters
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
  pcl::PointCloud<pcl::Normal>::Ptr   normals;
  pcl::PointIndices::Ptr              indices;
  int                                 R1;
  int                                 R2;
  int                                 S;
  int                                 normalization_type;
  int                                 filter_size;
  int                                 width;
  int                                 height;
  cv::Mat                             map;
  std::vector<float>                  cameraParametrs;
  PyramidParameters                   pyramidParameters;
  Symmetry3DMapParameters();
};
  
//void Calculate3DSymmetryPCA(AttentionModule::Symmetry3DMapParameters &parameters);
//void Calculate3DSymmetryPCAPyramid(AttentionModule::Symmetry3DMapParameters &parameters);

} //AttentionModule

#endif //SYMMETRY3D