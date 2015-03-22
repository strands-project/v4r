/*
 * local_recognition_mian_dataset.cpp
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>

int
main (int argc, char ** argv)
{
  std::string pcd_file = "";
  float z_dist = 3.f;
  std::string save_to_;
  pcl::console::parse_argument (argc, argv, "-save_to", save_to_);

  pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
  pcl::console::parse_argument (argc, argv, "-z_dist", z_dist);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile(pcd_file, *cloud);

  pcl::PassThrough<pcl::PointXYZRGB> pass_;
  pass_.setFilterLimits (0.f, z_dist);
  pass_.setFilterFieldName ("z");
  pass_.setInputCloud (cloud);
  pass_.setKeepOrganized (true);
  pass_.filter (*cloud);

  cv::Mat depth(cloud->height, cloud->width, CV_8UC1);
  depth.setTo(cv::Scalar(0));

  cv::Mat image(cloud->height, cloud->width, CV_8UC3);

  for(int r=0; r < (int)cloud->height; r++)
  {
      for(int c=0; c < (int)cloud->width; c++)
      {
          unsigned char rs = cloud->at(c, r).r;
          unsigned char gs = cloud->at(c, r).g;
          unsigned char bs = cloud->at(c, r).b;
          image.at<cv::Vec3b>(r,c) = cv::Vec3b(bs,gs,rs);
      }
  }

  for(int r=0; r < depth.rows; r++)
  {
      for(int c=0; c < depth.cols; c++)
      {
          Eigen::Vector3f p = cloud->at(c,r).getVector3fMap();
          if(!pcl_isfinite(p[2]))
              continue;

          depth.at<unsigned char>(r,c) = (p[2] / z_dist) * 255;
      }
  }

  cv::imshow("depth", depth);
  cv::waitKey(0);

  {
      std::stringstream save_to;
      save_to << save_to_ << "_color.png";
      cv::imwrite(save_to.str(), image);
  }

  {
      std::stringstream save_to;
      save_to << save_to_ << "_depth.png";
      cv::imwrite(save_to.str(), depth);
  }

  //cv::imwrite(save_to_, depth);

}
