/**
 *  Copyright (C) 2012  
 *    Ekaterina Potapova, Andreas Richtsfeld, Johann Prankl, Thomas Mörwald, Michael Zillich
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1170 Vienna, Austria
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
 * @file createTrainingSet.cpp
 * @author Andreas Richtsfeld, Ekaterina Potapova
 * @date January 2014
 * @version 0.1
 * @brief Creates training set.
 */


#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <fstream>

#include <pcl/io/pcd_io.h>

#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem.hpp>

#include "v4r/attention_segmentation/PCLUtils.h"
#include "v4r/attention_segmentation/segmentation.h"

void readData(std::string filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud, pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &pcl_cloud_l)
{

  if(!(pclAddOns::readPointCloud<pcl::PointXYZRGBL>(filename.c_str(),pcl_cloud_l)))
  {
    //exit(0);
    if(!(pclAddOns::readPointCloud<pcl::PointXYZRGB>(filename.c_str(),pcl_cloud)))
    {
      exit(0);
    }
    v4r::ClipDepthImage(pcl_cloud);
    return;
  }
  
  v4r::ConvertPCLCloud(pcl_cloud_l,pcl_cloud);
  v4r::ClipDepthImage(pcl_cloud);
}

void printUsage(char *av)
{
  printf("Usage: %s cloud.pcd training_data.txt \n"
    " Options:\n"
    "   [-h] ... show this help.\n"
    "   cloud.pcd         ... specify rgbd-image filename\n"
    "   training_data.txt ... filename to put training samples\n", av);
  std::cout << " Example: " << av << " cloud.pcd training_data.txt" << std::endl;
}

int main(int argc, char *argv[])
{
  if(argc != 3)
  {
    printUsage(argv[0]);
    exit(0);
  }
  
  std::string rgbd_filename = argv[1];
  std::string train_ST_file_name = argv[2];
  
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcl_cloud_l( new pcl::PointCloud<pcl::PointXYZRGBL>() );        ///< labeled pcl-cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud( new pcl::PointCloud<pcl::PointXYZRGB>() );           ///< original pcl point cloud
  
  readData(rgbd_filename,pcl_cloud,pcl_cloud_l);
  
  v4r::Segmenter segmenter; 
  segmenter.setPointCloud(pcl_cloud_l);
  segmenter.setPointCloud(pcl_cloud);
  segmenter.setTrainSTFilename(train_ST_file_name);
  
  segmenter.createTrainFile();
  
  return(0);
}


