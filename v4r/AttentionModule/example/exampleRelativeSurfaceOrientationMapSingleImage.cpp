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
  AttentionModule::RelativeSurfaceOrientationMapParameters parameters;
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
  
  parameters.normal.normal[0] = coefficients->values[0];
  parameters.normal.normal[1] = coefficients->values[1];
  parameters.normal.normal[2] = coefficients->values[2];
  
/*  // Create object to 
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(objects_indices);
  // Extract plane points
  extract.setNegative(false);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_objects(new pcl::PointCloud<pcl::PointXYZ>());
  extract.filter(*cloud_objects);
  
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cloud_objects, 255, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud_objects, color, "cloud_objects");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "cloud_objects");
  viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (cloud_objects, normals, 10, 0.02, "normals");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "normals");

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
*/  
  AttentionModule::CalculateRelativeSurfaceOrientationMap(parameters);
  
  cv::imshow("map",parameters.map);
  cv::waitKey();
  
  cv::Mat map;
  parameters.map.convertTo(map,CV_8U,255);
  std::string map_path = boost::filesystem::basename(image_name) + ".pgm";
  cv::imwrite(map_path,map);
    
  return(0);
}