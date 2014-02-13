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
  
  //calculate point cloud normals
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
  if(!pclAddOns::ComputePointNormals<pcl::PointXYZ>(cloud,objects_indices,normals))
  {
    return(pclAddOns::NORMALS);
  }
  
  parameters.setCloud(cloud);
  parameters.setIndices(objects_indices);
  parameters.setModelCoefficients(coefficients);
  
  std::vector<cv::Mat> maps;
  maps.resize(3);
  
  parameters.calculateSurfaceDistanceMap(maps.at(0));
  
  AttentionModule::RelativeSurfaceOrientationMapParameters parametersRSO;
  parametersRSO.width = image.cols;
  parametersRSO.height = image.rows;
  parametersRSO.cloud = cloud;
  parametersRSO.indices = objects_indices;
  parametersRSO.normals = normals;
  parametersRSO.normal.normal[0] = coefficients->values[0];
  parametersRSO.normal.normal[1] = coefficients->values[1];
  parametersRSO.normal.normal[2] = coefficients->values[2];
  AttentionModule::CalculateRelativeSurfaceOrientationMap(parametersRSO);
  parametersRSO.map.copyTo(maps.at(1));
  
  AttentionModule::IKNMapParameters parametersIKN;
  image.copyTo(parametersIKN.image);
  parametersIKN.width = image.cols;
  parametersIKN.height = image.rows;
  AttentionModule::CalculateIKNMap(parametersIKN);
  parametersIKN.map.copyTo(maps.at(2));
  
  cv::Mat map;
  AttentionModule::CombineMaps(maps,map);
  
  cv::Mat temp;
  int kernel_size = 8;//pow(2,5-1);
  cv::Mat element = cv::Mat_<uchar>::ones(kernel_size,kernel_size);
  cv::erode(map,temp,element);
  
  cv::imshow("temp",temp);
  cv::waitKey();
  
  map.convertTo(map,CV_8U,255);
  std::string map_path = boost::filesystem::basename(image_name) + ".pgm";
  cv::imwrite(map_path,map);
    
  return(0);
}