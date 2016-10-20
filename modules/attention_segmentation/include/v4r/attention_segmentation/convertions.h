/**
 *  Copyright (C) 2012  
 *    Ekaterina Potapova
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1040 Vienna, Austria
 *    potapova(at)acin.tuwien.ac.at
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/
 */


#ifndef EPCONVERTIONS_H
#define EPCONVERTIONS_H

#include <v4r/core/macros.h>
#include "v4r/attention_segmentation/eputils_headers.h"

namespace v4r
{

#ifndef NOT_USE_PCL
  
/**
 * converts depth image to XYZ point cloud
 * */
V4R_EXPORTS void Depth2PointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointIndices::Ptr indices, const cv::Mat depth, 
                      const std::vector<float> cameraParametrs, cv::Mat mask = cv::Mat(), float th = 0.0);

/**
 * converts depth and color images to XYZ point cloud
 * */
V4R_EXPORTS void ColorAndDepth2PointCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, const cv::Mat depth, const cv::Mat color,
                              const std::vector<float> cameraParametrs, float th = 0.0);

//ep:begin: revision at 17-07-2014
V4R_EXPORTS void pointCloud_2_depth(cv::Mat &depth, pcl::PointCloud<pcl::PointXYZRGBL>::ConstPtr cloud,
                        unsigned int width, unsigned int height, pcl::PointIndices::Ptr indices = pcl::PointIndices::Ptr(new pcl::PointIndices()), float th = 0.0);
V4R_EXPORTS void pointCloud_2_depth(cv::Mat &depth, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                        unsigned int width, unsigned int height, pcl::PointIndices::Ptr indices = pcl::PointIndices::Ptr(new pcl::PointIndices()), float th = 0.0);
V4R_EXPORTS void pointCloud_2_rgb(cv::Mat &RGB, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_xyzrgb,
                      unsigned int width, unsigned int height, pcl::PointIndices::Ptr indices = pcl::PointIndices::Ptr(new pcl::PointIndices()));
V4R_EXPORTS void pointCloud_2_channels(cv::Mat &xchannel, cv::Mat &ychannel, cv::Mat &zchannel, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                         unsigned int width, unsigned int height, pcl::PointIndices::Ptr indices = pcl::PointIndices::Ptr(new pcl::PointIndices()), float th = 0.0);
V4R_EXPORTS void normals_2_channels(cv::Mat &xnormals, cv::Mat &ynormals, cv::Mat &znormals, pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                         unsigned int width, unsigned int height, pcl::PointIndices::Ptr indices = pcl::PointIndices::Ptr(new pcl::PointIndices()));

/**
 * converts XYZ point cloud to disparity
 * */
V4R_EXPORTS void pointCloud_2_disparity(cv::Mat &disparity, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, 
                            int width, int height, pcl::PointIndices::Ptr indices = pcl::PointIndices::Ptr(new pcl::PointIndices()), 
                            float f = 525, float b = 0.075, float th = 0.4);
/**
 * converts XYZ point cloud to mask
 * */
V4R_EXPORTS void indices_2_image(cv::Mat &mask, unsigned int width, unsigned int height, pcl::PointIndices::Ptr indices= pcl::PointIndices::Ptr(new pcl::PointIndices()));
//ep:end: revision at 17-07-2014

//ep:begin: revision at 17-07-2014
V4R_EXPORTS void pointCloudXYZimageRGB_2_cloudXYZRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb, 
                                         pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz, cv::Mat &RGB, unsigned int width, unsigned int height);

V4R_EXPORTS void depthRGB_2_cloudXYZRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb, const cv::Mat &depth, const cv::Mat &RGB,
                            const std::vector<float> &cameraParametrs, unsigned int width, unsigned int height, float th = 0.0);

V4R_EXPORTS void pointCloudXYZRGB_2_cloudXYZimageRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb, 
                                         pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz, cv::Mat &RGB, unsigned int width, unsigned int height);

/**
 * converts XYZRGB point cloud to image
 * */
V4R_EXPORTS void pointCloudXYZimageRGBimageL_2_cloudXYZRGBL(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_xyzrgbl, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz,
						cv::Mat &RGB, cv::Mat &L, unsigned int width, unsigned int height);
V4R_EXPORTS void pointCloudXYZRGBL_2_cloudXYZimageRGBlableL(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_xyzrgbl, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz, 
						cv::Mat &RGB, cv::Mat &L, unsigned int width, unsigned int height);
V4R_EXPORTS void pointCloudXYZRGBimageL_2_cloudXYZRGBL(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_xyzrgbl, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb,
					   cv::Mat &L, unsigned int width, unsigned int height);
V4R_EXPORTS void pointCloudXYZRGBL_2_cloudXYZRGBlableL(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_xyzrgbl, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb, 
					   cv::Mat &L, unsigned int width, unsigned int height);
//ep:end: revision at 17-07-2014

/**
 * creates bin masks from clusters
 * */
V4R_EXPORTS void binMasksFromClusters(std::vector<cv::Mat> &binMasks, std::vector<pcl::PointIndices::ConstPtr> clusters);

#endif

/**
 * converts disparity to depth
 * */
V4R_EXPORTS void Disparity2Depth(cv::Mat &depth, const cv::Mat disparity, float f = 525, float b = 0.075);

/**
 * transfers double to char map for future visualization
 * (assumes that map already normalized to (0,1))
 * */
V4R_EXPORTS void FloatMap2UcharMap(cv::Mat &map_u, const cv::Mat map_f);

} //namespace v4r

#endif // EPCONVERTIONS_H
