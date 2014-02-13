#include <opencv2/opencv.hpp>

#include "v4r/AttentionModule/AttentionModule.hpp"
#include "v4r/EPUtils/EPUtils.hpp"

// This program shows the use of Surface Height Saliency Map on one image

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
  AttentionModule::SurfaceHeightSaliencyMap parameters;
  parameters.setWidth(image.cols);
  parameters.setHeight(image.rows);
    
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
  
  parameters.setCloud(cloud);
  parameters.setIndices(objects_indices);
  parameters.setModelCoefficients(coefficients);
    
  cv::Mat map;
  parameters.calculateSurfaceDistanceMap(map);
  
  cv::imshow("map",map);
  cv::waitKey();
  
  map.convertTo(map,CV_8U,255);
  std::string map_path = boost::filesystem::basename(image_name) + ".pgm";
  cv::imwrite(map_path,map);
    
  return(0);
}