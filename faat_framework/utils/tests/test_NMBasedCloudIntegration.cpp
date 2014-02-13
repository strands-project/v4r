/*
 * GO3D.cpp
 *
 *  Created on: Oct 24, 2013
 *      Author: aitor
 */

#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <pcl/visualization/pcl_visualizer.h>
#include <faat_pcl/utils/noise_models.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/transforms.h>
#include <faat_pcl/utils/noise_model_based_cloud_integration.h>

namespace bf = boost::filesystem;

//./bin/GO3D -input_dir /home/aitor/aldoma_employee_svn/code/thomas/code/T_16_GO3D/

int
main (int argc, char ** argv)
{
  bool organized_normals = true;
  float w_t = 0.5f;
  bool depth_edges = true;
  float max_angle = 70.f;
  float lateral_sigma = 0.002f;
  float resolution = 0.005f;
  int min_points_per_voxel = 0;
  float final_resolution = resolution;
  bool reverse = false;
  std::string transformations_pattern = ".*transformation_.*.txt";
  std::string pattern_scenes = ".*cloud_.*.pcd";

  std::string input_dir_, input_dir_transforms;
  pcl::console::parse_argument (argc, argv, "-input_dir", input_dir_);
  pcl::console::parse_argument (argc, argv, "-input_dir_transforms", input_dir_transforms);
  pcl::console::parse_argument (argc, argv, "-organized_normals", organized_normals);
  pcl::console::parse_argument (argc, argv, "-w_t", w_t);
  pcl::console::parse_argument (argc, argv, "-depth_edges", depth_edges);
  pcl::console::parse_argument (argc, argv, "-max_angle", max_angle);
  pcl::console::parse_argument (argc, argv, "-lateral_sigma", lateral_sigma);
  pcl::console::parse_argument (argc, argv, "-resolution", resolution);
  pcl::console::parse_argument (argc, argv, "-min_points_per_voxel", min_points_per_voxel);
  pcl::console::parse_argument (argc, argv, "-final_resolution", final_resolution);
  pcl::console::parse_argument (argc, argv, "-reverse", reverse);
  pcl::console::parse_argument (argc, argv, "-pattern_scenes", pattern_scenes);
  pcl::console::parse_argument (argc, argv, "-transformations_pattern", transformations_pattern);

  bf::path input = input_dir_;
  std::vector<std::string> scene_files;
  std::vector<std::string> transformation_files;

  faat_pcl::utils::getFilesInDirectory(input, scene_files, pattern_scenes);

  bf::path input_trans = input_dir_transforms;
  faat_pcl::utils::getFilesInDirectory(input_trans, transformation_files, transformations_pattern);

  std::cout << "Number of clouds:" << scene_files.size() << std::endl;
  std::cout << "Number of transformations:" << transformation_files.size() << std::endl;

  std::sort(scene_files.begin(), scene_files.end());
  std::sort(transformation_files.begin(), transformation_files.end());

  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> occlusion_clouds;
  std::vector < Eigen::Matrix4f > transforms_to_global;
  std::vector<std::vector<float> > weights_;
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normal_clouds;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_from_transforms_(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_from_transforms_no_filter_(new pcl::PointCloud<pcl::PointXYZRGB>);

  for(size_t i=0; i < scene_files.size(); i++)
  {
    std::cout << scene_files[i] << " " << transformation_files[i] << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>);

    {
      std::stringstream load;
      load << input_dir_ << "/" << scene_files[i];
      pcl::io::loadPCDFile(load.str(), *scene);
      occlusion_clouds.push_back(scene);
    }

    {
      Eigen::Matrix4f trans;
      std::stringstream load;
      load << input_dir_transforms << "/" << transformation_files[i];
      faat_pcl::utils::readMatrixFromFile(load.str(), trans);
      transforms_to_global.push_back(trans);
    }

    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
    if(organized_normals)
    {
      std::cout << "Organized normals" << std::endl;
      pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
      ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
      ne.setMaxDepthChangeFactor (0.02f);
      ne.setNormalSmoothingSize (20.0f);
      ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::BORDER_POLICY_MIRROR);
      ne.setInputCloud (occlusion_clouds[i]);
      ne.compute (*normal_cloud);
    }
    else
    {
      std::cout << "Not organized normals" << std::endl;
      pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
      ne.setInputCloud (occlusion_clouds[i]);
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
      ne.setSearchMethod (tree);
      ne.setRadiusSearch (0.02);
      ne.compute (*normal_cloud);
    }

    normal_clouds.push_back(normal_cloud);

    faat_pcl::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
    nm.setInputCloud(scene);
    nm.setInputNormals(normal_cloud);
    nm.setLateralSigma(lateral_sigma);
    nm.setMaxAngle(max_angle);
    nm.setUseDepthEdges(depth_edges);
    nm.compute();
    std::vector<float> weights;
    nm.getWeights(weights);

    weights_.push_back(weights);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered;
    nm.getFilteredCloudRemovingPoints(filtered, w_t);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_trans(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*filtered, *scene_trans, transforms_to_global[i]);
    *big_cloud_from_transforms_ += *scene_trans;

    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_trans(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::transformPointCloud(*occlusion_clouds[i], *scene_trans, transforms_to_global[i]);
        *big_cloud_from_transforms_no_filter_ += *scene_trans;
    }

  }

  if(reverse)
  {
      std::reverse(occlusion_clouds.begin(), occlusion_clouds.end());
      std::reverse(weights_.begin(), weights_.end());
      std::reverse(transforms_to_global.begin(), transforms_to_global.end());
      std::reverse(normal_clouds.begin(), normal_clouds.end());
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  faat_pcl::utils::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration;
  nmIntegration.setInputClouds(occlusion_clouds);
  nmIntegration.setResolution(resolution);
  nmIntegration.setWeights(weights_);
  nmIntegration.setTransformations(transforms_to_global);
  nmIntegration.setMinWeight(w_t);
  nmIntegration.setInputNormals(normal_clouds);
  nmIntegration.setMinPointsPerVoxel(min_points_per_voxel);
  nmIntegration.setFinalResolution(final_resolution);
  nmIntegration.compute(octree_cloud);

  pcl::visualization::PCLVisualizer vis ("registered cloud");
  int v1, v2, v3, v4;
  vis.createViewPort (0, 0, 0.5, 0.5, v1);
  vis.createViewPort (0.5, 0, 1, 0.5, v2);
  vis.createViewPort (0, 0.5, 0.5, 1, v3);
  vis.createViewPort (0.5, 0.5, 1, 1, v4);

  //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (big_cloud_from_transforms_);
  //vis.addPointCloud (big_cloud_from_transforms_, handler, "big", v2);

  {
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (octree_cloud);
    vis.addPointCloud (octree_cloud, handler, "octree_cloud", v1);
  }

  {
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (big_cloud_from_transforms_no_filter_);
    vis.addPointCloud (big_cloud_from_transforms_no_filter_, handler, "big_no_filter", v3);
  }

  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> used_clouds;
  pcl::PointCloud<pcl::Normal>::Ptr big_normals(new pcl::PointCloud<pcl::Normal>);
  nmIntegration.getOutputNormals(big_normals);
  nmIntegration.getInputCloudsUsed(used_clouds);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_from_masked (new pcl::PointCloud<pcl::PointXYZRGB>);

  for(size_t i=0; i < used_clouds.size(); i++)
  {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

      pcl::transformPointCloud(*used_clouds[i], *cloud, transforms_to_global[i]);
      *big_cloud_from_masked += *cloud;
  }

  /*{
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (big_cloud_from_masked);
    vis.addPointCloud (big_cloud_from_masked, handler, "big_from_clouds", v4);
  }*/

  vis.spin ();

}
