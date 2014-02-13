#ifndef EPCONVERTIONS_H
#define EPCONVERTIONS_H

#include "headers.hpp"

namespace EPUtils
{

#ifndef NOT_USE_PCL
  
/**
 * converts depth image to XYZ point cloud
 * */
void Depth2PointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointIndices::Ptr indices, const cv::Mat depth, 
                      const std::vector<float> cameraParametrs, cv::Mat mask = cv::Mat(), float th = 0.0);

/**
 * converts depth and color images to XYZ point cloud
 * */
void ColorAndDepth2PointCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, const cv::Mat depth, const cv::Mat color,
                              const std::vector<float> cameraParametrs, float th = 0.0);

void PointCloud2Depth(cv::Mat &depth, pcl::PointCloud<pcl::PointXYZRGBL>::ConstPtr cloud,
                      int width, int height, pcl::PointIndices::Ptr indices = pcl::PointIndices::Ptr(new pcl::PointIndices()), float th = 0.0);
void PointCloud2Depth(cv::Mat &depth, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                      int width, int height, pcl::PointIndices::Ptr indices = pcl::PointIndices::Ptr(new pcl::PointIndices()), float th = 0.0);
/**
 * converts XYZ point cloud to disparity
 * */
void PointCloud2Disparity(cv::Mat &disparity, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, 
                          int width, int height, pcl::PointIndices::Ptr indices = pcl::PointIndices::Ptr(new pcl::PointIndices()), 
                          float f = 525, float b = 0.075, float th = 0.4);
/**
 * converts XYZ point cloud to mask
 * */
void Indices2Mask(cv::Mat &mask, int width, int height, pcl::PointIndices::Ptr indices= pcl::PointIndices::Ptr(new pcl::PointIndices()));

void PointCloudXYZ2PointCloudXYZRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb, 
                                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, cv::Mat image);

void PointCloudXYZRGB2PointCloudXYZ(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb, 
                                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

/**
 * converts XYZRGB point cloud to image
 * */
void PointCloudXYZRGB2RGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, cv::Mat_<cv::Vec3b> &image);
void PointCloudXYZRGBL2RGB(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, cv::Mat &image);

/**
 * creates bin masks from clusters
 * */
void binMasksFromClusters(std::vector<cv::Mat> &binMasks, std::vector<pcl::PointIndices::ConstPtr> clusters);

#endif

/**
 * converts disparity to depth
 * */
void Disparity2Depth(cv::Mat &depth, const cv::Mat disparity, float f = 525, float b = 0.075);

/**
 * transfers double to char map for future visualization
 * (assumes that map already normalized to (0,1))
 * */
void FloatMap2UcharMap(cv::Mat &map_u, const cv::Mat map_f);

} //namespace EPUtils

#endif // EPCONVERTIONS_H
