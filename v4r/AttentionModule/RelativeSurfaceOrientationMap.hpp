#ifndef RELATIVE_SURFACE_ORIENATION_HPP
#define RELATIVE_SURFACE_ORIENATION_HPP

#include "headers.hpp"
#include "pyramids.hpp"

namespace AttentionModule
{

enum OrientationTypes
{
  AM_VERTICAL      = 0,
  AM_HORIZONTAL,
};
  
/**
 * parameters for relative surface orientation saliency map
 * */

struct RelativeSurfaceOrientationMapParameters
{
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud;
  pcl::PointIndices::Ptr                   indices;
  pcl::PointCloud<pcl::Normal>::Ptr        normals;
  pcl::Normal                              normal;
  int                                      normalization_type;
  int                                      filter_size;
  int                                      width;
  int                                      height;
  cv::Mat                                  map;
  cv::Mat                                  mask;
  int                                      orientationType;
  float                                    normal_module_threshold;
  std::vector<float>                       cameraParametrs;
  PyramidParameters                        pyramidParameters;
  RelativeSurfaceOrientationMapParameters();
};

/**
 * calculates single relative surface orientation map
 * */
int CalculateRelativeSurfaceOrientationMap(RelativeSurfaceOrientationMapParameters &parameters);
int CalculateRelativeSurfaceOrientationMapPyramid(RelativeSurfaceOrientationMapParameters &parameters);
float getOrientationCoefficient(int orientationType);

} // namespace AttentionModule

#endif //RELATIVE_SURFACE_ORIENATION_HPP