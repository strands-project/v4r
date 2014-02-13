/*
 * test_linemod.cpp
 *
 *  Created on: Oct 29, 2013
 *      Author: aitor
 */

#include <pcl/recognition/linemod/line_rgbd.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <faat_pcl/recognition/hv/hv_go_3D.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <faat_pcl/3d_rec_framework/defines/faat_3d_rec_framework_defines.h>
#include <algorithm>    // std::random_shuffle
#include <faat_pcl/utils/pcl_opencv.h>

namespace bf = boost::filesystem;

//./bin/GO3D -input_dir /home/aitor/aldoma_employee_svn/code/thomas/code/T_16_GO3D/

void getMaskFromObjectIndices(pcl::PointCloud<IndexPoint> & obj_indices_cloud,
                                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
                                  pcl::MaskMap & mask, pcl::RegionXY & region)
{
  pcl::PointIndices indices;
  indices.indices.resize(obj_indices_cloud.points.size());
  for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
    indices.indices[kk] = obj_indices_cloud.points[kk].idx;

  pcl::PointCloud<int> mask_cloud;
  mask_cloud.width = cloud->width;
  mask_cloud.height = cloud->height;
  mask_cloud.points.resize(cloud->width * cloud->height);
  for(size_t k=0; k < mask_cloud.points.size(); k++)
    mask_cloud.points[k] = 0;

  for(size_t k=0; k < indices.indices.size(); k++)
    mask_cloud.points[indices.indices[k]] = 1;

  mask.resize(cloud->width, cloud->height);
  size_t min_x (cloud->width), min_y (cloud->height), max_x (0), max_y (0);

  for (size_t j = 0; j < cloud->height; ++j)
  {
    for (size_t i = 0; i < cloud->width; ++i)
    {
      mask (i,j) = mask_cloud.points[j*cloud->width+i];
      if (mask_cloud.points[j*cloud->width+i])
      {
        min_x = std::min (min_x, i);
        max_x = std::max (max_x, i);
        min_y = std::min (min_y, j);
        max_y = std::max (max_y, j);
      }
    }
  }

  //pcl::RegionXY region;
  region.x = static_cast<int> (min_x);
  region.y = static_cast<int> (min_y);
  region.width = static_cast<int> (max_x - min_x + 1);
  region.height = static_cast<int> (max_y - min_y + 1);
}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr RGBTORGBA(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & rgb_cloud)
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rgba_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::copyPointCloud(*rgb_cloud, *rgba_cloud);
  return rgba_cloud;
}

struct linemod_training_template
{
  std::string object_id_;
  int view_id_;
  Eigen::Matrix4f matrix_;
};

int
main (int argc, char ** argv)
{
  std::string input_dir_;
  std::string detect_on_;
  int n_detections_to_show_ = 10;

  pcl::console::parse_argument (argc, argv, "-training_templates", input_dir_);
  pcl::console::parse_argument (argc, argv, "-detect_on_", detect_on_);
  pcl::console::parse_argument (argc, argv, "-n_detections", n_detections_to_show_);

  bf::path input = input_dir_;
  std::vector<std::string> training_files;
  std::vector<std::string> object_indices_files;
  std::vector<std::string> pose_files;

  std::string pattern_scenes = ".*cloud_.*.pcd";
  std::string pattern_oi = ".*object_indices.*.pcd";
  std::string pattern_poses = ".*pose_.*.txt";
  std::string so_far = "";
  faat_pcl::utils::getFilesInDirectoryRecursive(input, so_far, training_files, pattern_scenes);
  so_far = "";
  faat_pcl::utils::getFilesInDirectoryRecursive(input, so_far, object_indices_files, pattern_oi);
  so_far = "";
  faat_pcl::utils::getFilesInDirectoryRecursive(input, so_far, pose_files, pattern_poses);

  std::cout << "Number of clouds:" << training_files.size() << std::endl;
  std::cout << "Number of object indices:" << object_indices_files.size() << std::endl;
  std::cout << "Number of poses:" << pose_files.size() << std::endl;

  std::sort(pose_files.begin(), pose_files.end());
  std::sort(training_files.begin(), training_files.end());
  std::sort(object_indices_files.begin(), object_indices_files.end());

  std::vector<linemod_training_template> linemod_id_to_template;
  linemod_id_to_template.resize(training_files.size());

  pcl::LineRGBD<pcl::PointXYZRGBA> line_rgbd;
  size_t template_id = 0;
  //std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> views;
  //views.resize(training_files.size());

  for(size_t i=0; i < training_files.size(); i++, template_id++)
  {
    std::cout << training_files[i] << std::endl;
    std::cout << object_indices_files[i] << std::endl;
    std::cout << pose_files[i] << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>);
    Eigen::Matrix4f trans;
    pcl::PointCloud<IndexPoint> obj_indices_cloud;

    {
      std::stringstream load;
      load << input_dir_ << "/" << training_files[i];
      pcl::io::loadPCDFile(load.str(), *scene);
    }

    {
      std::stringstream load;
      load << input_dir_ << "/" << pose_files[i];
      faat_pcl::utils::readMatrixFromFile(load.str(), trans);
    }

    {
      std::stringstream load;
      load << input_dir_ << "/" << object_indices_files[i];
      pcl::io::loadPCDFile(load.str(), obj_indices_cloud);
    }

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_rgba = RGBTORGBA(scene);
    //views[i] = scene_rgba;

    line_rgbd.setInputCloud(scene_rgba);
    line_rgbd.setInputColors(scene_rgba);

    //creates and fills mask
    pcl::MaskMap mask;
    pcl::RegionXY region;
    getMaskFromObjectIndices(obj_indices_cloud, scene, mask, region);

    size_t object_id = template_id;
    line_rgbd.createAndAddTemplate(*scene_rgba, object_id, mask, mask, region);

    std::vector<std::string> strs;
    boost::split (strs, training_files[i], boost::is_any_of ("/"));
    std::string obj_id = "unknown";
    if(strs.size() > 1)
    {
      obj_id = strs[0];
    }

    linemod_id_to_template[template_id].object_id_ = obj_id;
    linemod_id_to_template[template_id].view_id_ = (int)i;
    linemod_id_to_template[template_id].matrix_ = trans;
  }

  std::vector<std::string> training_files_orig = training_files;

  if(detect_on_.compare("") != 0)
  {
    bf::path input = detect_on_;
    training_files.clear();

    std::string pattern_scenes = ".*cloud_.*.pcd";
    std::string so_far = "";
    faat_pcl::utils::getFilesInDirectoryRecursive(input, so_far, training_files, pattern_scenes);
  }
  else
  {
    std::random_shuffle ( training_files.begin(), training_files.end() );
  }

  for(size_t i=0; i < training_files.size(); i++)
  {
    std::cout << "Going to detect on:" << training_files[i] << std::endl;

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGBA>);

    {
      std::stringstream load;
      if(detect_on_.compare("") != 0)
      {
        load << detect_on_ << "/" << training_files[i];
      }
      else
      {
        load << input_dir_ << "/" << training_files[i];
      }
      pcl::io::loadPCDFile(load.str(), *scene);
    }

    line_rgbd.setInputCloud(scene);
    line_rgbd.setInputColors(scene);

    std::vector<pcl::LineRGBD<pcl::PointXYZRGBA>::Detection > detections;
    line_rgbd.detectSemiScaleInvariant(detections);

    std::cout << "Number of detections:" << detections.size() << std::endl;
    std::vector< std::pair<size_t, float> > template_id_response;
    for(size_t d=0; d < detections.size(); d++)
    {
      //std::cout << detections[d].object_id << " " << detections[d].response << " " << linemod_id_to_template[detections[d].object_id].view_id_ << std::endl;
      template_id_response.push_back(std::make_pair(d, detections[d].response));
    }

    std::sort(template_id_response.begin(), template_id_response.end(),
              boost::bind(&std::pair<size_t, float>::second, _1) >
              boost::bind(&std::pair<size_t, float>::second, _2));

    cv::Mat_ < cv::Vec3b > colorImage;
    PCLOpenCV::ConvertPCLCloud2Image<pcl::PointXYZRGBA> (scene, colorImage);
    cv::Mat collage(colorImage.rows, colorImage.cols*2, CV_8UC3);

    for(size_t d=0; d < std::min(n_detections_to_show_, (int)detections.size()); d++)
    {
      std::cout << detections[template_id_response[d].first].object_id << " " << template_id_response[d].second << " " << linemod_id_to_template[detections[template_id_response[d].first].object_id].view_id_ << std::endl;

      cv::Point tl = cv::Point(detections[template_id_response[d].first].region.x, detections[template_id_response[d].first].region.y);
      cv::Point br = cv::Point(detections[template_id_response[d].first].region.x + detections[template_id_response[d].first].region.width,
                               detections[template_id_response[d].first].region.y + detections[template_id_response[d].first].region.height);

      cv::Scalar color = cv::Scalar( 255, 0, 0);
      cv::rectangle( colorImage, tl, br, color, 2, 8, 0 );

      collage(cv::Range(0, collage.rows), cv::Range(0, colorImage.cols)) = colorImage +cv::Scalar(0);
      cv::Mat_ < cv::Vec3b > colorImage_match;

      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_match(new pcl::PointCloud<pcl::PointXYZRGBA>);

      {
        std::stringstream load;
        load << input_dir_ << "/" << training_files_orig[linemod_id_to_template[detections[template_id_response[d].first].object_id].view_id_];
        pcl::io::loadPCDFile(load.str(), *scene_match);
      }
      PCLOpenCV::ConvertPCLCloud2Image<pcl::PointXYZRGBA> (scene_match, colorImage_match);

      collage(cv::Range(0, collage.rows), cv::Range(collage.cols/2, collage.cols)) = colorImage_match +cv::Scalar(0);

      cv::namedWindow("test");
      cv::imshow("test", collage);
      cv::waitKey(0);
    }
  }
}
