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
#include <faat_pcl/recognition/hv/hv_go_3D.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <faat_pcl/utils/noise_models.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>

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

  std::string input_dir_;
  pcl::console::parse_argument (argc, argv, "-input_dir", input_dir_);
  pcl::console::parse_argument (argc, argv, "-organized_normals", organized_normals);
  pcl::console::parse_argument (argc, argv, "-w_t", w_t);
  pcl::console::parse_argument (argc, argv, "-depth_edges", depth_edges);
  pcl::console::parse_argument (argc, argv, "-max_angle", max_angle);
  pcl::console::parse_argument (argc, argv, "-lateral_sigma", lateral_sigma);

  bf::path input = input_dir_;
  std::vector<std::string> scene_files;
  std::vector<std::string> model_files;
  std::vector<std::string> transformation_files;
  std::string pattern_scenes = ".*cloud_.*.pcd";
  std::string pattern_models = ".*model_.*.pcd";
  std::string transformations_pattern = ".*transformation_.*.txt";

  faat_pcl::utils::getFilesInDirectory(input, scene_files, pattern_scenes);
  faat_pcl::utils::getFilesInDirectory(input, model_files, pattern_models);
  faat_pcl::utils::getFilesInDirectory(input, transformation_files, transformations_pattern);

  std::cout << "Number of clouds:" << scene_files.size() << std::endl;
  std::cout << "Number of models:" << model_files.size() << std::endl;
  std::cout << "Number of transformations:" << transformation_files.size() << std::endl;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  std::stringstream load;
  load << input_dir_ << "/big_cloud.pcd";
  pcl::io::loadPCDFile(load.str(), *big_cloud);

  std::sort(scene_files.begin(), scene_files.end());
  std::sort(transformation_files.begin(), transformation_files.end());

  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> aligned_models;
  std::vector < std::string > ids;
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> occlusion_clouds;
  std::vector < Eigen::Matrix4f > transforms_to_global;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_from_transforms_(new pcl::PointCloud<pcl::PointXYZRGB>);

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
      load << input_dir_ << "/" << transformation_files[i];
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

    faat_pcl::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
    nm.setInputCloud(scene);
    nm.setInputNormals(normal_cloud);
    nm.setLateralSigma(lateral_sigma);
    nm.setMaxAngle(max_angle);
    nm.setUseDepthEdges(depth_edges);
    nm.compute();
    std::vector<float> weights;
    nm.getWeights(weights);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered;
    nm.getFilteredCloudRemovingPoints(filtered, w_t);

    occlusion_clouds[i] = filtered;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_trans(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*filtered, *scene_trans, transforms_to_global[i]);
    *big_cloud_from_transforms_ += *scene_trans;
  }

  for(size_t i=0; i < model_files.size(); i++)
  {
    std::cout << model_files[i] << std::endl;

    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>);
      std::stringstream load;
      load << input_dir_ << "/" << model_files[i];
      pcl::io::loadPCDFile(load.str(), *scene);
      aligned_models.push_back(scene);
    }
  }

  float leaf = 0.005f;
  typedef pcl::PointXYZRGB PointT;
  faat_pcl::GO3D<PointT, PointT> go;
  go.setResolution (leaf);
  go.setAbsolutePoses (transforms_to_global);
  go.setOcclusionsClouds (occlusion_clouds);
  go.setZBufferSelfOcclusionResolution (250);
  go.setInlierThreshold (0.0075f);
  go.setRadiusClutter (0.035f);
  go.setDetectClutter (false); //Attention, detect clutter turned off!
  go.setRegularizer (3.f);
  go.setClutterRegularizer (3.f);
  go.setHypPenalty (0.f);
  go.setIgnoreColor (false);
  go.setColorSigma (0.25f);
  go.setOptimizerType (0);
  go.setSceneCloud (big_cloud);
  go.addModels (aligned_models, true);
  go.verify ();
  std::vector<bool> mask;
  go.getMask (mask);

  pcl::visualization::PCLVisualizer vis ("registered cloud");
  int v1, v2, v3, v4;
  vis.createViewPort (0, 0, 0.5, 0.5, v1);
  vis.createViewPort (0.5, 0, 1, 0.5, v2);
  vis.createViewPort (0, 0.5, 0.5, 1, v3);
  vis.createViewPort (0.5, 0.5, 1, 1, v4);

  /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_ds = go.getSceneCloud();
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (big_cloud_ds);
  vis.addPointCloud (big_cloud_ds, handler, "big", v3);*/

  /*pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (big_cloud_vx_after_mv);
  vis.addPointCloud (big_cloud_vx_after_mv, handler, "big", v1);*/

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (big_cloud_from_transforms_);
  vis.addPointCloud (big_cloud_from_transforms_, handler, "big", v3);

  for(size_t i=0; i < aligned_models.size(); i++)
  {
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified (aligned_models[i]);
    std::stringstream name;
    name << "Hypothesis_model_" << i;
    vis.addPointCloud<pcl::PointXYZRGB> (aligned_models[i], handler_rgb_verified, name.str (), v2);
  }

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr smooth_cloud_ =  go.getSmoothClustersRGBCloud();
  if(smooth_cloud_)
  {
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> random_handler (smooth_cloud_);
    vis.addPointCloud<pcl::PointXYZRGBA> (smooth_cloud_, random_handler, "smooth_cloud", v4);
  }
  else
  {
    std::vector<pcl::PointCloud<PointT>::ConstPtr> visible_models;
    visible_models = go.getVisibleModels();
    for (size_t i = 0; i < mask.size (); i++)
   {
     if (mask[i])
     {

       pcl::PointCloud<pcl::PointXYZRGB>::Ptr inliers_outlier_cloud;
       go.getInlierOutliersCloud((int)i, inliers_outlier_cloud);

       pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified (inliers_outlier_cloud);
       std::stringstream name;
       name << "verified_visible_" << i;
       vis.addPointCloud<pcl::PointXYZRGB> (inliers_outlier_cloud, handler_rgb_verified, name.str (), v4);

       /*pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified (visible_models[i]);
       std::stringstream name;
       name << "verified_visible_" << i;
       vis.addPointCloud<pcl::PointXYZRGB> (visible_models[i], handler_rgb_verified, name.str (), v4);*/
     }
   }
  }

  int verified = 0;
  for (size_t i = 0; i < mask.size (); i++)
  {
    if (mask[i])
    {

      /*{
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified (aligned_models[i]);
        std::stringstream name;
        name << "verified" << i;
        vis.addPointCloud<pcl::PointXYZRGB> (aligned_models[i], handler_rgb_verified, name.str (), v3);
      }*/

      verified++;

      {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified (aligned_models[i]);
        std::stringstream name;
        name << "verified___" << i;
        vis.addPointCloud<pcl::PointXYZRGB> (aligned_models[i], handler_rgb_verified, name.str (), v1);
      }
    }
  }

  std::cout << "Verified:" << verified << std::endl;

  vis.spin ();

}
