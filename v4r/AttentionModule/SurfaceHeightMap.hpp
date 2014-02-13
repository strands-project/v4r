#ifndef SURFACEHEIGHT_HPP
#define SURFACEHEIGHT_HPP

#include "headers.hpp"
#include "pyramids.hpp"

namespace AttentionModule
{

/**
 * parameters for surface height saliency map
 * */

enum SurfaceTypes
{
  AM_TALL      = 0,
  AM_SHORT,
};

class SurfaceHeightSaliencyMap
{
public:
  SurfaceHeightSaliencyMap();
  void setCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_);
  void setIndices(pcl::PointIndices::Ptr indices_);
  void setModelCoefficients(pcl::ModelCoefficients::Ptr coefficients_);
  void setDistanceFromTop(float distance_from_top_);
  void setNormalizationType(int normalization_type_);
  void setFilterSize(int filter_size_);
  void setMaxDistance(int max_distance_);
  void setWidth(int width_);
  void setHeight(int height_);
  void setHeightType(int heightType_);
  void setMask(cv::Mat &mask_);
  void setCameraParameters(std::vector<float> &cameraParametrs_);
  
/**
* calculates single surface height map
* */
  int calculateSurfaceHeightMap(cv::Mat &map_);
  int calculateSurfaceDistanceMap(cv::Mat &map_);
  int calculateSurfaceHeightMapPyramid(cv::Mat &map_);
  bool updateMask(cv::Mat &new_mask_);


private:
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud;
  pcl::PointIndices::Ptr                   indices;
  pcl::ModelCoefficients::Ptr              coefficients;
  float                                    distance_from_top;
  int                                      normalization_type;
  int                                      filter_size;
  float                                    max_distance;
  int                                      width;
  int                                      height;
  int                                      heightType;
  cv::Mat                                  mask;
  cv::Mat                                  map;
  std::vector<float>                       cameraParametrs;
  bool                                     getUpdate;
  bool                                     maskInUse;
  PyramidParameters                        pyramidParameters;
  
  float getHeightCoefficient(int heightType);
};

} // namespace AttentionModule

#endif //SURFACEHEIGHT_HPP