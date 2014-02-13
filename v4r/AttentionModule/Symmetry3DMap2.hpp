#ifndef SYMMETRY3D_MAP_HPP
#define SYMMETRY3D_MAP_HPP

#include "headers.hpp"

namespace AttentionModule
{

struct MiddlePoint {
    int num;
    float distance;
    pcl::Normal normal;
    pcl::PointXYZ point;
    MiddlePoint();
  };
  
  MiddlePoint::MiddlePoint()
  {
    num = 0;
    distance = 0;
    normal.normal[0] = 0;
    normal.normal[1] = 0;
    normal.normal[2] = 0;
    point.x = 0;
    point.y = 0;
    point.z = 0;
  }
  
class Symmetry3DMap
{
private:
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  pcl::PointCloud<pcl::Normal>::Ptr   normals;
  pcl::PointIndices::Ptr              indices;
  int                                 R;
  int                                 normalizationType;
  int                                 filterSize;
  int                                 width;
  int                                 height;
  bool                                pyramidMode;
  cv::Mat                             map;
  std::vector<float>                  cameraParametrs;
  //AttentionModule::PyramidParameters  pyramidParameters;
  
  void createLookUpMap(cv::Mat &lookupMap);
  void checkParameters();
  void computeSingle();
  //void computePyramid();
  
public:
  Symmetry3DMap();
  
  void setCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_);
  void setNormals(pcl::PointCloud<pcl::Normal>::Ptr normals_);
  void setIndices(pcl::PointIndices::Ptr indices_);
  void setR(int R_);
  //void setNormalizationType(int normalizationType_);
  //void setFilterSize(int filterSize_);
  void setWidth(int width_);
  void setHeight(int height_);
  void setPyramidMode(bool pyramidMode_);
  //void setCameraParameters(std::vector<float> cameraParametrs_);
  //void setPyramidParameters(AttentionModule::PyramidParameters pyramidParameters_);
  
  //pcl::PointCloud<pcl::PointXYZ>::Ptr getCloud();
  //pcl::PointCloud<pcl::Normal>::Ptr getNormals();
  //pcl::PointIndices::Ptr getIndices();
  void getMap(cv::Mat &map_)
  {
    map.copyTo(map_);
  };
  void compute();
  
};

} //namespace AttentionModule

#endif //SYMMETRY3D_MAP_HPP
