#include <opencv2/opencv.hpp>

#include "v4r/AttentionModule/AttentionModule.hpp"
#include "v4r/EPUtils/EPUtils.hpp"

int main(int argc, char** argv)
{
  
  // read image
  std::string image_name(argv[1]);
  cv::Mat image = cv::imread(image_name,-1);
    
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  std::string cloud_name(argv[2]);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cloud_name,*cloud) == -1)
  {
    std::cerr << "[ERROR] Couldn't read point cloud." << std::endl;
    return -1;
  }
  
  // start creating parameters
  AttentionModule::SurfaceCurvatureMapParameters parameters;
  parameters.width = image.cols;
  parameters.height = image.rows;
    
  // create filtered point cloud
  pcl::PointIndices::Ptr indices(new pcl::PointIndices());
  if(!pclAddOns::FilterPointCloud<pcl::PointXYZ>(cloud,indices))
    return(pclAddOns::FILTER);
  
  // segment plane
  pcl::PointIndices::Ptr plane_indices(new pcl::PointIndices());
  pcl::PointIndices::Ptr objects_indices(new pcl::PointIndices());
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  if(!pclAddOns::SegmentPlane<pcl::PointXYZ>(cloud,indices,plane_indices,objects_indices,coefficients))
  {
    return(pclAddOns::SEGMENT);
  }
  
  //calculate point cloud normals
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
  if(!pclAddOns::ComputePointNormals<pcl::PointXYZ>(cloud,objects_indices,normals))
  {
    return(pclAddOns::NORMALS);
  }
  
  parameters.cloud = cloud;
  parameters.indices = objects_indices;
  parameters.normals = normals;
    
  AttentionModule::CalculateSurfaceCurvatureMap(parameters);
  
  cv::imshow("map",parameters.map);
  cv::waitKey();
    
  return(0);
}