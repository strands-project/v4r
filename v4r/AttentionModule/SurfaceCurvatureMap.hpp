#ifndef SURFACE_CURVATURE_HPP
#define SURFACE_CURVATURE_HPP

#include "headers.hpp"
#include "pyramids.hpp"

namespace AttentionModule
{

/**
 * parameters for surface curvature saliency map
 * */

enum CurvatureTypes
{
  AM_FLAT      = 0,
  AM_CONVEX,
};

struct SurfaceCurvatureMapParameters
{
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud;
  pcl::PointIndices::Ptr                   indices;
  pcl::PointCloud<pcl::Normal>::Ptr        normals;
  int                                      normalization_type;
  int                                      filter_size;
  int                                      width;
  int                                      height;
  int                                      curvatureType;
  cv::Mat                                  mask;
  cv::Mat                                  map;
  std::vector<float>                       cameraParametrs;
  PyramidParameters                        pyramidParameters;
  SurfaceCurvatureMapParameters();
};

/**
 * calculates single surface curvature map
 * */
int CalculateSurfaceCurvatureMap(SurfaceCurvatureMapParameters &parameters);
int CalculateSurfaceCurvatureMapPyramid(SurfaceCurvatureMapParameters &parameters);
float getCurvatureCoefficient(int curvatureType);

} // namespace AttentionModule

#endif //SURFACE_CURVATURE_HPP