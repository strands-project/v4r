#include "SurfaceCurvatureMap.hpp"

namespace AttentionModule
{

SurfaceCurvatureMapParameters::SurfaceCurvatureMapParameters()
{
  //cloud = NULL;
  //indices = NULL
  //normals = NULL;
  
  normalization_type = EPUtils::NT_NONE_REAL;
  filter_size = 5;
  mask = cv::Mat_<float>::zeros(0,0);
  map = cv::Mat_<uchar>::zeros(0,0);
  width = 0;
  height = 0;
  curvatureType = AM_CONVEX;
  cameraParametrs.clear();
}

int CalculateSurfaceCurvatureMap(SurfaceCurvatureMapParameters &parameters)
{
  if(parameters.cloud == NULL)
  {
    return(AM_POINTCLOUD);
  }
  
  if(parameters.indices == NULL)
  {
    return(AM_POINTCLOUD);
  }

  if(parameters.normals == NULL)
  {
    return(AM_NORMALCLOUD);
  }

  if(parameters.indices->indices.size() != parameters.normals->size())
  {
    return(AM_DIFFERENTSIZES);
  }

  if(( (parameters.width == 0) || (parameters.height == 0) ) && ( (parameters.map.rows == 0) || (parameters.map.cols == 0) ))
  {
    return(AM_IMAGE);
  }

  if((parameters.width == 0) || (parameters.height == 0))
  {
    parameters.height = parameters.map.rows;
    parameters.width  = parameters.map.cols;
  }

  if((parameters.mask.cols != parameters.width) || (parameters.mask.rows != parameters.height))
  {
    parameters.mask = cv::Mat_<uchar>::ones(parameters.height,parameters.width);
  }
  
  parameters.map = cv::Mat_<float>::zeros(parameters.height,parameters.width);

  float curvatureCoefficient = getCurvatureCoefficient(parameters.curvatureType);
  
  for(unsigned int pi = 0; pi < parameters.indices->indices.size(); ++pi)
  {
    int idx = parameters.indices->indices.at(pi);
    int r = idx / parameters.width;
    int c = idx % parameters.width;
    
    if(parameters.mask.at<uchar>(r,c))
    {
      float nx = parameters.normals->points.at(pi).normal[0];
      float ny = parameters.normals->points.at(pi).normal[1];
      float nz = parameters.normals->points.at(pi).normal[2];

      if(std::isnan(nx) || std::isnan(ny) || std::isnan(nz))
      {
        continue;
      }
      
      float value = parameters.normals->points.at(pi).curvature;
      float t1 = 1.0-curvatureCoefficient;
      float t2 = curvatureCoefficient;
      value = t1*value + t2*(1.0-value);
      
      parameters.map.at<float>(r,c) = value;
    }
  }

  int filter_size = parameters.filter_size;
  cv::blur(parameters.map,parameters.map,cv::Size(filter_size,filter_size));

  EPUtils::normalize(parameters.map,parameters.normalization_type);

  return(AM_OK);
}

int CalculateSurfaceCurvatureMapPyramid(SurfaceCurvatureMapParameters &parameters)
{
  if(parameters.cloud == NULL)
  {
    return(AM_POINTCLOUD);
  }
  
  if(parameters.indices == NULL)
  {
    return(AM_POINTCLOUD);
  }
  
  // set values of the input image
  if(( (parameters.width == 0) || (parameters.height == 0) ) && ( (parameters.map.rows == 0) || (parameters.map.cols == 0) ))
  {
    return(AM_IMAGE);
  }

  if((parameters.width == 0) || (parameters.height == 0))
  {
    parameters.height = parameters.map.rows;
    parameters.width  = parameters.map.cols;
  }
  
  if(parameters.cameraParametrs.size() != 4)
  {
    return(AM_CAMERAPARAMETRS);
  }
  
  // create depth
  cv::Mat depth;
  EPUtils::PointCloud2Depth(depth,parameters.cloud,parameters.width,parameters.height,parameters.indices);
  
  // calculate puramid with saliency maps
  int max_level = parameters.pyramidParameters.max_level + 1;
  parameters.pyramidParameters.pyramidImages.clear();
  cv::buildPyramid(depth,parameters.pyramidParameters.pyramidImages,max_level);
  parameters.pyramidParameters.pyramidFeatures.clear();
  parameters.pyramidParameters.pyramidFeatures.resize(parameters.pyramidParameters.pyramidImages.size());
  
  for(int i = parameters.pyramidParameters.start_level; i <= parameters.pyramidParameters.max_level; ++i)
  {
    int scalingFactor = pow(2.0f,i);
    std::vector<float> cameraParametrs;
    cameraParametrs.resize(4);
    cameraParametrs.at(0) = parameters.cameraParametrs.at(0)/scalingFactor;
    cameraParametrs.at(1) = parameters.cameraParametrs.at(1)/scalingFactor;
    cameraParametrs.at(2) = parameters.cameraParametrs.at(2)/scalingFactor;
    cameraParametrs.at(3) = parameters.cameraParametrs.at(3)/scalingFactor;
    
    // start creating parameters
    SurfaceCurvatureMapParameters parameters_current;
    parameters_current.width = parameters.pyramidParameters.pyramidImages.at(i).cols;
    parameters_current.height = parameters.pyramidParameters.pyramidImages.at(i).rows;
    
    // create scaled point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointIndices::Ptr indices(new pcl::PointIndices());
    EPUtils::Depth2PointCloud(cloud,indices,parameters.pyramidParameters.pyramidImages.at(i),cameraParametrs);
    
    parameters_current.cloud = cloud;
    parameters_current.indices = indices;
    
    //calculate point cloud normals
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    if(!pclAddOns::ComputePointNormals<pcl::PointXYZ>(parameters_current.cloud,parameters_current.indices,normals))
    {
      return(AM_NORMALCLOUD);
    }
    
    parameters_current.normals = normals;
    
    CalculateSurfaceCurvatureMap(parameters_current);
    parameters_current.map.copyTo(parameters.pyramidParameters.pyramidFeatures.at(i));
  }

  // combine saliency maps
  combinePyramid(parameters.pyramidParameters);
  parameters.pyramidParameters.map.copyTo(parameters.map);
  
  return(0);
}

float getCurvatureCoefficient(int curvatureType)
{
  switch(curvatureType)
  {
    case AM_FLAT:
      return(1.0);
    case AM_CONVEX:
      return(0.0);
  }
  return(0.0);
}

} //namespace AttentionModule