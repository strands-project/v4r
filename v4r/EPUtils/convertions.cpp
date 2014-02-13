#include "convertions.hpp"

namespace EPUtils
{

#ifndef NOT_USE_PCL
  
//
void Depth2PointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointIndices::Ptr indices, const cv::Mat depth, 
                      const std::vector<float> cameraParametrs, cv::Mat mask, float th)
{
  if( mask.empty() )
  {
    mask = cv::Mat_<uchar>::ones(depth.size());
  }
  
  // check types
  assert(mask.type() == CV_8UC1);
  assert(depth.type() == CV_32FC1);
  assert(mask.size() == depth.size());
  
  float cx, cy, fx, fy;
  fx = cameraParametrs.at(0);
  fy = cameraParametrs.at(1);
  cx = cameraParametrs.at(2);
  cy = cameraParametrs.at(3);

  cloud->points.clear();
  cloud->is_dense = false;
  cloud->width  = depth.cols;
  cloud->height = depth.rows;
  cloud->points.reserve(cloud->width * cloud->height);
  indices->indices.clear();
  
  for(int i = 0; i < depth.rows; ++i)
  {
    for(int j = 0; j < depth.cols; ++j)
    {
      if(mask.at<uchar>(i,j) == 0)
	continue;
      
      float z_ir = depth.at<float>(i,j);
      if(z_ir < th)
      {
        pcl::PointXYZ point(std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN());
        cloud->points.push_back(point);
        continue; 
      }

      float x_ir = ((j - cx) / fx) * z_ir;
      float y_ir = ((i - cy) / fy) * z_ir;

      pcl::PointXYZ point(x_ir,y_ir,z_ir);
      cloud->points.push_back(point);
      int idx = i*(depth.cols)+j;
      indices->indices.push_back(idx);
    }
  }
  
  cloud->width  = cloud->points.size();
  cloud->height = 1;
}

//
void ColorAndDepth2PointCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, const cv::Mat depth, const cv::Mat color,
                              const std::vector<float> cameraParametrs, float th)
{
  // check types
  assert(color.type() == CV_8UC3);
  assert(depth.type() == CV_32FC1);
  assert(color.size() == depth.size());
  
  float cx, cy, fx, fy;
  fx = cameraParametrs.at(0);
  fy = cameraParametrs.at(1);
  cx = cameraParametrs.at(2);
  cy = cameraParametrs.at(3);

  cloud->points.clear();
  cloud->is_dense = false;
  cloud->width = depth.cols;
  cloud->height = depth.rows;
  cloud->points.reserve (cloud->width * cloud->height);
  
  for(int i = 0; i < depth.rows; ++i)
  {
    for(int j = 0; j < depth.cols; ++j)
    {
      float z_ir = depth.at<float>(i,j);
      
      uchar rr = color.at<uchar>(i,3*j+2);//(r,3*c+2)
      uchar gg = color.at<uchar>(i,3*j+1);
      uchar bb = color.at<uchar>(i,3*j+0);
      
      pcl::PointXYZRGBL point;
      
      if(z_ir < th)
      {
	point.x = std::numeric_limits<float>::quiet_NaN();
        point.y = std::numeric_limits<float>::quiet_NaN();
        point.z = std::numeric_limits<float>::quiet_NaN();
      }
      else
      {
        float x_ir = ((j - cx) / fx) * z_ir;
        float y_ir = ((i - cy) / fy) * z_ir;
	
	point.x = x_ir;
        point.y = y_ir;
        point.z = z_ir;
      }

      point.r = rr;
      point.g = gg;
      point.b = bb;
      
      point.label = 0;
      
      cloud->points.push_back(point);
    }
  }
}

//
void PointCloud2Depth(cv::Mat &depth, pcl::PointCloud<pcl::PointXYZRGBL>::ConstPtr cloud,
                      int width, int height, pcl::PointIndices::Ptr indices, float th)
{
  if( (indices->indices.size()) == 0 )
  {
    indices->indices.resize(cloud->size());
    for(unsigned int i = 0; i < cloud->size(); ++i)
    {
      indices->indices.at(i) = i;
    }
  }
  
  // check types
  assert(indices->indices.size() <= cloud->size());
  assert(width*height == cloud->size());
    
  depth = cv::Mat_<float>::zeros(height,width);

  for(unsigned int i = 0; i < indices->indices.size(); ++i)
  {
    int idx = indices->indices.at(i);
    
    if(std::isnan(cloud->points.at(idx).x) ||
       std::isnan(cloud->points.at(idx).y) ||
       std::isnan(cloud->points.at(idx).z))
    {
      continue;
    }
    
    float z_ir = cloud->points.at(idx).z;

    if(z_ir < th)
    {
      continue;
    }

    int r = idx / width;
    int c = idx % width;
    depth.at<float>(r,c) = z_ir;
  }
}

void PointCloud2Depth(cv::Mat &depth, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                      int width, int height, pcl::PointIndices::Ptr indices, float th)
{
  if( (indices->indices.size()) == 0 )
  {
    indices->indices.resize(cloud->size());
    for(unsigned int i = 0; i < cloud->size(); ++i)
    {
      indices->indices.at(i) = i;
    }
  }
  
  // check types
  assert(indices->indices.size() <= cloud->size());
  assert(width*height == cloud->size());
  
  depth = cv::Mat_<float>::zeros(height,width);
  
  for(unsigned int i = 0; i < indices->indices.size(); ++i)
  {
    int idx = indices->indices.at(i);
    
    if(std::isnan(cloud->points.at(idx).x) ||
      std::isnan(cloud->points.at(idx).y) ||
      std::isnan(cloud->points.at(idx).z))
    {
      continue;
    }
    
    float z_ir = cloud->points.at(idx).z;
    
    if(z_ir < th)
    {
      continue;
    }
    
    int r = idx / width;
    int c = idx % width;
    depth.at<float>(r,c) = z_ir;
  }
}

//
void PointCloud2Disparity(cv::Mat &disparity, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, 
                          int width, int height, pcl::PointIndices::Ptr indices, float f, float b, float th)
{
  if( (indices->indices.size()) == 0 )
  {
    indices->indices.resize(cloud->size());
    for(unsigned int i = 0; i < cloud->size(); ++i)
    {
      indices->indices.at(i) = i;
    }
  }
  
  // check types
  assert(indices->indices.size() <= cloud->size());
  assert(width*height == cloud->size());
  
  cv::Scalar defaultDisparityValue(255);
  disparity = cv::Mat(height,width,CV_32FC1,defaultDisparityValue);
  
  for(unsigned int i = 0; i < indices->indices.size(); ++i)
  {
    int idx = indices->indices.at(i);
    if(std::isnan(cloud->points.at(idx).x) ||
       std::isnan(cloud->points.at(idx).y) ||
       std::isnan(cloud->points.at(idx).z))
    {
      continue;
    }
    
    float z_ir = cloud->points.at(idx).z;
    
    if(z_ir < th)
    {
      continue;
    }

    float disp = b*f/z_ir;
    
    if(disp < 0)
    {
      continue;
    }

    int r = idx / width;
    int c = idx % width;
    
    disparity.at<float>(r,c) = disp;
  }
}


//
void Indices2Mask(cv::Mat &mask, int width, int height, pcl::PointIndices::Ptr indices)
{
  if( (indices->indices.size()) == 0 )
  {
    indices->indices.resize(width*height);
    for(int i = 0; i < width*height; ++i)
    {
      indices->indices.at(i) = i;
    }
  }
  
  // check types
  assert(indices->indices.size() <= width*height);
  
  mask = cv::Mat_<uchar>::zeros(height,width);

  for(unsigned int i = 0; i < indices->indices.size(); ++i)
  {
    int idx = indices->indices.at(i);
    int r = idx / width;
    int c = idx % width;
    mask.at<uchar>(r,c) = 255;
  }
}

//
void PointCloudXYZ2PointCloudXYZRGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb, 
                                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, cv::Mat image)
{
  // check types
  assert(cloud->points.size() == image.rows*image.cols);
  
  cloud_xyzrgb->points.resize(cloud->points.size());
  
  int width = image.cols;
  int height = image.rows;
  
  cloud_xyzrgb->width = width;
  cloud_xyzrgb->height = height;
  
  for(int r = 0; r < height; ++r)
  {
    for(int c = 0; c < width; ++c)
    {
      int index = r*width + c;
      
      cloud_xyzrgb->points.at(index).x = cloud->points.at(index).x;
      cloud_xyzrgb->points.at(index).y = cloud->points.at(index).y;
      cloud_xyzrgb->points.at(index).z = cloud->points.at(index).z;
      
      uint8_t r_col = image.at<uchar>(r,3*c+2);
      uint8_t g_col = image.at<uchar>(r,3*c+1);
      uint8_t b_col = image.at<uchar>(r,3*c+0);
      uint32_t rgb = ((uint32_t)r_col << 16 | (uint32_t)g_col << 8 | (uint32_t)b_col);
      cloud_xyzrgb->points.at(index).rgb = *reinterpret_cast<float*>(&rgb);
    }
  }
}

//
void PointCloudXYZRGB2PointCloudXYZ(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, 
                                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyzrgb)
{
  cloud->points.resize(cloud_xyzrgb->points.size());
  
  int width = cloud_xyzrgb->width;
  int height = cloud_xyzrgb->height;
  
  cloud_xyzrgb->width = width;
  cloud_xyzrgb->height = height;
  
  for(int r = 0; r < height; ++r)
  {
    for(int c = 0; c < width; ++c)
    {
      int index = r*width + c;
      
      cloud->points.at(index).x = cloud_xyzrgb->points.at(index).x;
      cloud->points.at(index).y = cloud_xyzrgb->points.at(index).y;
      cloud->points.at(index).z = cloud_xyzrgb->points.at(index).z;
    }
  }
}

void PointCloudXYZRGB2RGB(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, cv::Mat_<cv::Vec3b> &image)
{
  int height = cloud->height;
  int width = cloud->width;

  image = cv::Mat_<cv::Vec3b>(height, width);
  
  for (unsigned row = 0; row < height; row++)
  {
    for (unsigned col = 0; col < width; col++)
    {
      cv::Vec3b &cvp = image.at<cv::Vec3b>(row,col);
      int position = row * width + col;
      const pcl::PointXYZRGB pt = cloud->points.at(position);
      
      cvp[2] = pt.r;
      cvp[1] = pt.g;
      cvp[0] = pt.b;
    }
  }
}

void PointCloudXYZRGBL2RGB(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud, cv::Mat &image)
{
  int height = cloud->height;
  int width = cloud->width;

  image = cv::Mat::zeros(height,width,CV_8UC3);
  
  for (unsigned row = 0; row < height; row++)
  {
    for (unsigned col = 0; col < width; col++)
    {
      int position = row * width + col;
      const pcl::PointXYZRGBL pt = cloud->points.at(position);
      
      image.at<uchar>(row,3*col+2) = pt.r;
      image.at<uchar>(row,3*col+1) = pt.g;
      image.at<uchar>(row,3*col+0) = pt.b;
    }
  }
}

//
void binMasksFromClusters(std::vector<cv::Mat> &binMasks, std::vector<pcl::PointIndices::ConstPtr> clusters)
{
  binMasks.clear();
  binMasks.resize(clusters.size());
  for(unsigned int k = 0; k < clusters.size(); ++k)
  {
    binMasks.at(k) = cv::Mat_<uchar>::zeros(480,640);
    for(unsigned int i = 0; i < clusters.at(k)->indices.size(); ++i)
    {
      int idx = clusters.at(k)->indices.at(i);
      int r = idx / 640;
      int c = idx % 640;
      binMasks.at(k).at<uchar>(r,c) = 255;
    }
  }
}

#endif

//
void Disparity2Depth(cv::Mat &depth, const cv::Mat disparity, float f, float b)
{
  depth = cv::Mat::zeros(disparity.size(),CV_32FC1);
  
  for(int i = 0; i < disparity.rows; ++i)
  {
    for(int j = 0; j < disparity.cols; ++j)
    {
      if(disparity.at<float>(i,j) >= 255)
      {
        continue;
      }

      float disp = disparity.at<float>(i,j);
      float z = b*f/disp;
      depth.at<float>(i,j) = z;
    }
  }
}

void FloatMap2UcharMap(cv::Mat &map_u, const cv::Mat map_f)
{
  cv::convertScaleAbs(map_f,map_u,255,0);
}

} //namespace EPUtils