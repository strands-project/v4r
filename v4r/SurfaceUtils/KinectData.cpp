/**
 *  Copyright (C) 2012  
 *    Andreas Richtsfeld
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1170 Vienn, Austria
 *    ari(at)acin.tuwien.ac.at
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

/**
 * @file KinectData.cpp
 * @author Andreas Richtsfeld
 * @date March 2012
 * @version 0.1
 * @brief Get kinect data live or from files (pcd, png, sfv)
 */


#include "KinectData.h"

typedef union
{
  struct
  {
    unsigned char b;  // Blue channel
    unsigned char g;  // Green channel
    unsigned char r;  // Red channel
    unsigned char a;  // Alpha channel
  };
  float float_value;
  long long_value;
} RGBValue;


cv::Vec4f KinectData::DepthToWorld(const int &x, const int &y, const float &depthValue)
{
  float centerX;
  float centerY;
  centerX = 320 - 0.5f;
  centerY = 240 - 0.5f;
  float constant = 1.9047619104/1000000.;    // from Kinect.cpp (valid for 640x480)
//   float constant = 570.3;
 
  cv::Vec4f result;

  // convert depth
  result[0] = (x - centerX) * depthValue * constant;
  result[1] = (y - centerY) * depthValue * constant;
  result[2] = depthValue * 0.001;

  // convert color
  RGBValue col;
  uchar *ptr = rgb_image.data;
  col.r = ptr[(y * rgb_image.cols + x) * 3 + 2]; // change red and blue channel
  col.g = ptr[(y * rgb_image.cols + x) * 3 + 1];
  col.b = ptr[(y * rgb_image.cols + x) * 3];
  result[3] = col.float_value;

  return result;
}

void KinectData::loadDepthImage(std::string filename, 
                                pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &cloud)
{
  if (cloud.get() == 0)
    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGBL>);
  
  unsigned size = 1024;
  char depth_next[1024] = "";
  char color_next[1024] = "";
  int cnt = 0;
  
  char current_color_file[128];
  char current_pcd_file[128];
  sprintf(current_color_file, data_ipl_filename.c_str(), data_current);
  sprintf(current_pcd_file, data_pcd_filename.c_str(), data_current);
  
  if(use_database_path)
    cnt += snprintf(color_next + cnt, size - cnt, "%simage_color/%s", db_path.c_str(), current_color_file);
  else
    sprintf(color_next, data_ipl_filename.c_str(), data_current);

  cnt = 0;
  if(use_database_path)
    cnt += snprintf(depth_next + cnt, size - cnt, "%sdisparity/%s", db_path.c_str(), current_pcd_file);
  else
    sprintf(depth_next, data_pcd_filename.c_str(), data_current);  

  // load color and depth image
  printf("[KinectData::loadDepthImage] Load color image file: %s\n", color_next);
  rgb_image = cv::imread(color_next);

  printf("[KinectData::loadDepthImage] Load depth file: %s\n", depth_next);
  cv::Mat depth;
  cv::Mat depth_new;
  depth = cv::imread(depth_next, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
  depth.convertTo(depth_new, CV_32F);
  
  // create pcl-cloud from color and depht
  cloud->width = depth.cols;
  cloud->height = depth.rows;
  cloud->points.resize(depth.cols*depth.rows);
  
  for (unsigned col = 0; col < cloud->width; col++) {
    for (unsigned row = 0; row < cloud->height; row++) {
      float d = depth_new.at<float>(row, col);
      cv::Vec4f p = DepthToWorld(col, row, d);
      cloud->points[row*cloud->width+col].x = p[0];
      cloud->points[row*cloud->width+col].y = p[1];
      cloud->points[row*cloud->width+col].z = p[2];
      cloud->points[row*cloud->width+col].rgb = p[3];
      cloud->points[row*cloud->width+col].label = 0;
    }
  }
  
  // ######################## Setup TomGine ########################
//   TomGine::tgTomGineThread dbgWin(640, 480, "TomGine Render Engine");
//   cv::Mat R = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
//   cv::Mat t = (cv::Mat_<double>(3, 1) << 0, 0, 0);
//   cv::Vec3d rotCenter(0, 0, 1.0);
// 
//   cv::Mat intrinsic;
//   surface::View view;
//   intrinsic = cv::Mat::zeros(3, 3, CV_64F);
//   view.intrinsic = Eigen::Matrix3d::Zero();
//   intrinsic.at<double> (0, 0) = intrinsic.at<double> (1, 1) = view.intrinsic(0, 0) = view.intrinsic(1, 1) = 525;
//   intrinsic.at<double> (0, 2) = view.intrinsic(0, 2) = 320;
//   intrinsic.at<double> (1, 2) = view.intrinsic(1, 2) = 240;
//   intrinsic.at<double> (2, 2) = view.intrinsic(2, 2) = 1.;
// 
//   dbgWin.SetClearColor(0.5, 0.5, 0.5);
//   dbgWin.SetCoordinateFrame();
//   dbgWin.SetCamera(intrinsic);
//   dbgWin.SetCamera(R, t);
//   dbgWin.SetRotationCenter(rotCenter);
//   dbgWin.Update();
//   
//   cv::Mat_<cv::Vec4f> cvCloud;
//   pclA::ConvertPCLCloud2CvMat(cloud, cvCloud);
//   dbgWin.AddPointCloud(cvCloud);
//   
//   cvWaitKey();
  // ######################## Setup TomGine ########################
}


void KinectData::ConvertCvMat2PCLCloud(const cv::Mat_<cv::Vec4f> &cv_cloud, 
                                       pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &pcl_cloud)
{
  int pos = 0;
  if (pcl_cloud.get() == 0)
    pcl_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGBL>);

  pcl_cloud->width = cv_cloud.cols;
  pcl_cloud->height = cv_cloud.rows;
  pcl_cloud->points.resize(cv_cloud.cols * cv_cloud.rows);

  for (int row = 0; row < cv_cloud.rows; row++) {
    for (int col = 0; col < cv_cloud.cols; col++) {
      pos = row * cv_cloud.cols + col;
      pcl_cloud->points[pos].x = cv_cloud(row, col)[0];
      pcl_cloud->points[pos].y = cv_cloud(row, col)[1];
      pcl_cloud->points[pos].z = cv_cloud(row, col)[2];
      pcl_cloud->points[pos].rgb = cv_cloud(row, col)[3];
      pcl_cloud->points[pos].label = 0;
    }
  }
}

void KinectData::ConvertPCLCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &in,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr &out)
{
  out.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

  out->width = in->width;
  out->height = in->height;
  out->points.resize(in->width*in->height);
  for (unsigned row = 0; row < in->height; row++) {
    for (unsigned col = 0; col < in->width; col++) {
      int idx = row * in->width + col;
      pcl::PointXYZRGBL &pt = in->points[idx];
      pcl::PointXYZRGB &npt = out->points[idx];
      npt.x = pt.x;
      npt.y = pt.y;
      npt.z = pt.z;
      npt.rgb = pt.rgb;
    }
  }
}

/* ------------------------ Kinect Data ---------------------------- */

KinectData::KinectData()
{
  deb = true;
  read_live = false;
  initialized = false;
  use_database_path = false;
  
  modelLoader = new surface::LoadFileSequence();
}

KinectData::~KinectData()
{
  delete modelLoader;
  if(read_live) {
    delete kinect;
  }
}


void KinectData::setDatabasePath(std::string path)
{
    use_database_path = true;
    db_path = path;
}

void KinectData::setReadDataFromFile(std::string pcd_filename, std::string color_filename,
                                     unsigned start, unsigned end, bool depth)
{
  initialized = true;
  data_start = start;
  data_end = end;
  data_current = start;
  load_depth = depth;
  data_ipl_filename = color_filename;
  data_pcd_filename = pcd_filename;
}

void KinectData::setReadDataLive()
{
  read_live = true;
  kinect = new Kinect::Kinect();
  kinect->StartCapture(0);
  initialized = true;
}

void KinectData::setReadDataLive(std::string _kinect_config)
{
  read_live = true;
  const char *config = _kinect_config.c_str();
  kinect = new Kinect::Kinect(config);
  kinect->StartCapture(0);
  initialized = true;
}

void KinectData::setReadModelsFromFile(std::string sfv_filename, unsigned start, unsigned end)
{
  initialized = true;
  int cnt = 0;
  unsigned size = 1024;
  char sfv[1024] = "";
  if(use_database_path) {
    cnt += snprintf(sfv + cnt, size - cnt, "%sresults/%s", db_path.c_str(), sfv_filename.c_str());
    modelLoader->InitFileSequence(sfv, start, end);
  }
  else
     modelLoader->InitFileSequence(sfv_filename, start, end);
}

void KinectData::getImageData(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &cloud)
{
  
  if(!initialized) {
    printf("[KinectData::getImageData] Error: Loading not initailised. Abort\n");
    return;
  }
  
  if(cloud.get() == 0)
    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGBL>);

  char current_pcd_file[128];
  unsigned size = 1024;
  char pcd_next[1024] = "";
  int cnt=0;
  
  if(!read_live) {
    if(data_current > data_end) {
      printf("[KinectData::getImageData] End of images reached. Exit\n");
      exit(0);
    }
    sprintf(current_pcd_file, data_pcd_filename.c_str(), data_current);
    if(use_database_path) {
      cnt += snprintf(pcd_next + cnt, size - cnt, "%s%s", db_path.c_str(), current_pcd_file);
      printf("[KinectData::getImageData] Load pcd file: %s\n", pcd_next);
    }
    else {
      sprintf(pcd_next, data_pcd_filename.c_str(), data_current);  
      printf("[KinectData::getImageData] Load pcd file: %s\n", pcd_next);
    }
    
    if(!load_depth)
      pcl::io::loadPCDFile(pcd_next, *cloud);
    else
      loadDepthImage(pcd_next, cloud);

    if(cloud->points.size() == 0) {
      printf("[KinectData::getImageData] No image data found!\n");
      std::exit(0);
    }
    
    c = cloud;
    data_current++;
  }
  else
  {
    cv::Mat_<cv::Vec4f> mat_cloud;
    kinect->NextFrame();
    kinect->Get3dWorldPointCloud(mat_cloud, 1);
    ConvertCvMat2PCLCloud(mat_cloud, cloud);
    c = cloud;
  }
}

void KinectData::getImageData(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr l_cloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
  getImageData(l_cloud);
  ConvertPCLCloud(l_cloud, cloud);
}

void KinectData::loadModelData(surface::View &view)
{
  modelLoader->LoadNextView(view);
}

