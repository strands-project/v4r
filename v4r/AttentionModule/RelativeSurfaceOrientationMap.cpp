#include "RelativeSurfaceOrientationMap.hpp"

namespace AttentionModule
{

RelativeSurfaceOrientationMapParameters::RelativeSurfaceOrientationMapParameters()
{
  //cloud = NULL;
  //indices = NULL;
  //normals = NULL;
  
  normalization_type = EPUtils::NT_NONE_REAL;
  filter_size = 5;
  map = cv::Mat_<float>::zeros(0,0);
  mask = cv::Mat_<uchar>::zeros(0,0);
  orientationType = AM_HORIZONTAL;
  width = 0;
  height = 0;
  normal_module_threshold = 0.000001;
  cameraParametrs.clear();
}

int CalculateRelativeSurfaceOrientationMap(RelativeSurfaceOrientationMapParameters &parameters)
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

  if(parameters.indices->indices.size() != parameters.normals->points.size())
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

  // Retrieve normal values
  float a = parameters.normal.normal[0];
  float b = parameters.normal.normal[1];
  float c = parameters.normal.normal[2];
  
  if(a*a + b*b + c*c < parameters.normal_module_threshold)
  {
    return(AM_NORMALCOEFFICIENTS);
  }

  float orientationCoefficient = getOrientationCoefficient(parameters.orientationType);

  parameters.map = cv::Mat_<float>::zeros(parameters.height,parameters.width);

  for(unsigned int pi = 0; pi < parameters.indices->indices.size(); ++pi)
  {
    int idx = parameters.indices->indices.at(pi);
    
    int y = idx / parameters.width;
    int x = idx % parameters.width;
    
    if(parameters.mask.at<uchar>(y,x))
    {
      float nx = parameters.normals->points.at(pi).normal[0];
      float ny = parameters.normals->points.at(pi).normal[1];
      float nz = parameters.normals->points.at(pi).normal[2];
      
      if(std::isnan(nx) || std::isnan(ny) || std::isnan(nz))
      {
        continue;
      }

      float value = nx*a + ny*b + nz*c;
      float n_mod = sqrt(nx*nx + ny*ny + nz*nz);
      float t_mod = sqrt(a*a + b*b + c*c);
      value = value / (n_mod*t_mod);
      value = value>0 ? value:-value;
      
      float t1 = 1.0-orientationCoefficient;
      float t2 = orientationCoefficient;
      value = t1*value + t2*(1.0-value);

      parameters.map.at<float>(y,x) = value;
    }
  }

  int filter_size = parameters.filter_size;
  cv::blur(parameters.map,parameters.map,cv::Size(filter_size,filter_size));

  EPUtils::normalize(parameters.map,parameters.normalization_type);

  return(AM_OK);
}

int CalculateRelativeSurfaceOrientationMapPyramid(RelativeSurfaceOrientationMapParameters &parameters)
{
  if(parameters.cloud == NULL)
  {
    return(AM_POINTCLOUD);
  }
  
  if(parameters.indices == NULL)
  {
    return(AM_POINTCLOUD);
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
    RelativeSurfaceOrientationMapParameters parameters_current;
    parameters_current.width = parameters.pyramidParameters.pyramidImages.at(i).cols;
    parameters_current.height = parameters.pyramidParameters.pyramidImages.at(i).rows;
    parameters_current.normal = parameters.normal;
    
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
    
    CalculateRelativeSurfaceOrientationMap(parameters_current);
    parameters_current.map.copyTo(parameters.pyramidParameters.pyramidFeatures.at(i));
  }

  // combine saliency maps
  combinePyramid(parameters.pyramidParameters);
  parameters.pyramidParameters.map.copyTo(parameters.map);
  
  return(0);
}

float getOrientationCoefficient(int orientationType)
{
  switch(orientationType)
  {
    case AM_HORIZONTAL:
      return(0.0);
    case AM_VERTICAL:
      return(1.0);
  }
  return(0.0);
}

} //namespace AttentionModule