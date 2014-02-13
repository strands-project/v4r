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
 * @file KinectData.h
 * @author Andreas Richtsfeld
 * @date March 2012
 * @version 0.1
 * @brief Get kinect data live or from files (pcd, png, sfv)
 */


#ifndef KINECT_DATA_H
#define KINECT_DATA_H

#include <ostream>
#include <cstdio>

#include <pcl/io/pcd_io.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "v4r/KinectInterface/Kinect.h"

#include "SurfaceModel.hpp"
#include "FileSystem.hh"


/**
 * @class KinectData
 */
class KinectData
{
private:
  bool deb;                                       ///< print debug messages
  bool read_live;                                 ///< load data live
  bool load_depth;                                ///< load data from depth image
  bool initialized;                               ///< kinect loader is initialized
  bool use_database_path;                         ///< use path to database
  
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr c;      ///< current pcl cloud
  cv::Mat rgb_image;                              ///< rgb-image from 
  
  unsigned data_start;                            ///< start image number
  unsigned data_end;                              ///< end image number
  unsigned data_current;                          ///< current image number
  std::string db_path;                            ///< database patch
  std::string data_pcd_filename;                  ///< pcd filename
  std::string data_ipl_filename;                  ///< ipl filename
  
  surface::LoadFileSequence *modelLoader;         ///< Load surface models
  Kinect::Kinect *kinect;                         ///< Load live kinect data

private:
  cv::Vec4f DepthToWorld(const int &x, const int &y, const float &depthValue);
  void loadDepthImage(std::string filename, 
                      pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &cloud);
  void ConvertCvMat2PCLCloud(const cv::Mat_<cv::Vec4f> &cv_cloud, 
                             pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &pcl_cloud);
  void ConvertPCLCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &in,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr &out);

public:
  KinectData();
  ~KinectData();
  
  /** Set database path **/
  void setDatabasePath(std::string path);
  
  /** Set read live **/
  void setReadDataLive();
  void setReadDataLive(std::string _kinect_config);
  
  /** Read pcd files from disk (or depth files, if depth=true) **/
  void setReadDataFromFile(std::string pcd_filename, std::string color_filename, 
                           unsigned start, unsigned end, bool depth = false);
  
  /** Read sfv models from disk **/
  void setReadModelsFromFile(std::string sfv_filename, unsigned start, unsigned end);

  /** Get image data **/
  void getImageData(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &cloud);
  void getImageData(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

  /** Get image number **/
  int getImageNumber() {return data_current-1;}
  
  /** Load model data from disk **/
  void loadModelData(surface::View &view);

};



#endif
