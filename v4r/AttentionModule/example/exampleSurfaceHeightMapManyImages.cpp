#include <opencv2/opencv.hpp>

#include "v4r/AttentionModule/AttentionModule.hpp"
#include "v4r/EPUtils/EPUtils.hpp"

// This program shows the use of Surface Height Saliency Map on one image

int main(int argc, char** argv)
{
  std::string directory_clouds("/home/kate/work/objectness/alexe/TOSD/test/clouds/");
  std::string directory_images("/home/kate/work/objectness/alexe/TOSD/test/rgb/");
  
  std::string directory_output("/home/kate/work/objectness/alexe/TOSD/test/SH/");

  std::vector<std::string> cloud_names;
  EPUtils::readFiles(directory_clouds,cloud_names);
  std::vector<std::string> image_names;
  EPUtils::readFiles(directory_images,image_names);
  
  for(unsigned int i = 0; i < cloud_names.size(); ++i)
  {
    std::cerr << "[INFO] Processing cloud: " << cloud_names.at(i) << "." << std::endl;
    std::cerr << "[INFO] Processing image: " << image_names.at(i) << "." << std::endl;
    
    cv::Mat image = cv::imread(image_names.at(i),-1);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (cloud_names.at(i),*cloud) == -1)
    {
      std::cerr << "[ERROR] Couldn't read point cloud." << std::endl;
      return -1;
    }
  
    // start creating parameters
    AttentionModule::SurfaceHeightMapParameters parameters;
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
  
    parameters.cloud = cloud;
    parameters.indices = objects_indices;
    parameters.coefficients = coefficients;
    
    CalculateSurfaceHeightMap(parameters);
  
    std::string map_path = directory_output;
    map_path += boost::filesystem::basename(image_names.at(i)) + ".pgm";
    cv::Mat temp;
    parameters.map.convertTo(temp,CV_8U,255);
    cv::imwrite(map_path,temp);
  }
    
  return(0);
}