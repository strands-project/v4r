#include "v4r/AttentionModule/AttentionModule.hpp"
#include "v4r/EPUtils/EPUtils.hpp"

int main(int argc, char** argv)
{
  if(argc != 3)
  {
    std::cerr << "Usage: image cloud" << std::endl;
    return(0);
  }
  
  std::string image_name(argv[1]);
  std::string cloud_name(argv[2]);
  // read image
  cv::Mat image = cv::imread(image_name,-1);
    
  AttentionModule::Symmetry3DMap symmetry3DMap;
    
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>() );
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cloud_name,*(cloud)) == -1)
  {
    std::cerr << "[ERROR] Couldn't read point cloud." << std::endl;
    return -1;
  }
  
  // start creating parameters
  symmetry3DMap.setCloud(cloud);
  symmetry3DMap.setWidth(image.cols);
  symmetry3DMap.setHeight(image.rows);
    
  //filter just obtained point cloud
  pcl::PointIndices::Ptr indices (new pcl::PointIndices() );
  if(!pclAddOns::FilterPointCloud<pcl::PointXYZ>(cloud,indices))
    return(pclAddOns::FILTER);
  symmetry3DMap.setIndices(indices);
    
  //calculate point cloud normals
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>() );
  if(!pclAddOns::ComputePointNormals<pcl::PointXYZ>(cloud,indices,normals,50))
    return(pclAddOns::NORMALS);
  symmetry3DMap.setNormals(normals);

  symmetry3DMap.setPyramidMode(false);
  symmetry3DMap.compute();  
  cv::Mat symmetry_map_one_level;
  symmetry3DMap.getMap(symmetry_map_one_level);
  
  symmetry3DMap.setPyramidMode(true);
  symmetry3DMap.compute();  
  cv::Mat symmetry_map_pyramid;
  symmetry3DMap.getMap(symmetry_map_pyramid);
  
  cv::imshow("symmetry_map_one_level",symmetry_map_one_level);
  cv::imshow("symmetry_map_pyramid",symmetry_map_pyramid);
  cv::waitKey();
    
  return(0);
}