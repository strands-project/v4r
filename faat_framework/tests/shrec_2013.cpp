/*
 * shrec_2013.cpp
 *
 *  Created on: Feb 10, 2013
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <faat_pcl/3d_rec_framework/pc_source/mesh_source.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/ourcvfh_estimator.h>
#include <faat_pcl/3d_rec_framework/pipeline/global_nn_recognizer_cvfh.h>
#include "faat_pcl/3d_rec_framework/utils/metrics.h"
#include <faat_pcl/3d_rec_framework/utils/vtk_model_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/octree/octree.h>

void
getModelsInDirectory (bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext)
{
  bf::directory_iterator end_itr;
  for (bf::directory_iterator itr (dir); itr != end_itr; ++itr)
  {
    //check if its a directory, then get models in it
    if (bf::is_directory (*itr))
    {
#if BOOST_FILESYSTEM_VERSION == 3
      std::string so_far = rel_path_so_far + (itr->path ().filename ()).string () + "/";
#else
      std::string so_far = rel_path_so_far + (itr->path ()).filename () + "/";
#endif

      bf::path curr_path = itr->path ();
      getModelsInDirectory (curr_path, so_far, relative_paths, ext);
    }
    else
    {
      //check that it is a ply file and then add, otherwise ignore..
      std::vector<std::string> strs;
#if BOOST_FILESYSTEM_VERSION == 3
      std::string file = (itr->path ().filename ()).string ();
#else
      std::string file = (itr->path ()).filename ();
#endif

      boost::split (strs, file, boost::is_any_of ("."));
      std::string extension = strs[strs.size () - 1];

      if (extension.compare (ext) == 0)
      {
#if BOOST_FILESYSTEM_VERSION == 3
        std::string path = rel_path_so_far + (itr->path ().filename ()).string ();
#else
        std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif

        relative_paths.push_back (path);
      }
    }
  }
}

int
main (int argc, char ** argv)
{
  std::string models_dir = "";
  std::string training_dir = "";
  float radius_normals_go_ = 0.015f;
  float model_scale = 1.f;
  bool force_retrain = false;
  bool normalize_bins_ = true;
  std::string query_dir = "";

  pcl::console::parse_argument (argc, argv, "-training_dir", training_dir);
  pcl::console::parse_argument (argc, argv, "-models_dir", models_dir);
  pcl::console::parse_argument (argc, argv, "-query_dir", query_dir);
  pcl::console::parse_argument (argc, argv, "-force_retrain", force_retrain);

  //configure mesh source
  /*boost::shared_ptr<faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > mesh_source (new faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ>);
   mesh_source->setPath (models_dir);
   mesh_source->setResolution (250);
   mesh_source->setTesselationLevel (0);
   mesh_source->setViewAngle (57.f);
   mesh_source->setRadiusSphere (1.5f);
   mesh_source->setModelScale (model_scale);
   mesh_source->setRadiusNormals(radius_normals_go_);
   mesh_source->generate (training_dir);

   boost::shared_ptr<faat_pcl::rec_3d_framework::Source<pcl::PointXYZ> > cast_source;
   cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > (mesh_source);

   //configure normal estimator
   boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal> > normal_estimator;
   normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal>);
   normal_estimator->setCMR (true);
   normal_estimator->setDoVoxelGrid (true);
   normal_estimator->setRemoveOutliers (false);
   //normal_estimator->setValuesForCMRFalse (0.003f, 0.018f);
   normal_estimator->setFactorsForCMR(3,7);

   boost::shared_ptr<faat_pcl::rec_3d_framework::OURCVFHEstimator<pcl::PointXYZ, pcl::VFHSignature308> > vfh_estimator;
   vfh_estimator.reset (new faat_pcl::rec_3d_framework::OURCVFHEstimator<pcl::PointXYZ, pcl::VFHSignature308>);
   vfh_estimator->setNormalEstimator (normal_estimator);
   vfh_estimator->setNormalizeBins (normalize_bins_);
   vfh_estimator->setRefineClustersParam (2.5f);
   vfh_estimator->setAdaptativeMLS (false);

   vfh_estimator->setAxisRatio (1.f);
   vfh_estimator->setMinAxisValue (1.f);

   vfh_estimator->setCVFHParams (0.15f, 0.015f, 2.5f);

   std::string desc_name = "our_cvfh";
   if (normalize_bins_)
   {
   desc_name = "our_cvfh_normalized";
   }

   std::cout << "Descriptor name:" << desc_name << std::endl;
   boost::shared_ptr<faat_pcl::rec_3d_framework::OURCVFHEstimator<pcl::PointXYZ, pcl::VFHSignature308> > cast_estimator;
   cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::OURCVFHEstimator<pcl::PointXYZ, pcl::VFHSignature308> > (vfh_estimator);

   faat_pcl::rec_3d_framework::GlobalNNCVFHRecognizer<faat_pcl::Metrics::HistIntersectionUnionDistance, pcl::PointXYZ, pcl::VFHSignature308> ourcvfh_global_;
   ourcvfh_global_.setDataSource (cast_source);
   ourcvfh_global_.setTrainingDir (training_dir);
   ourcvfh_global_.setDescriptorName (desc_name);
   ourcvfh_global_.setFeatureEstimator (cast_estimator);
   ourcvfh_global_.setNN (10);
   ourcvfh_global_.setICPIterations (0); //ATTENTION: Danger
   ourcvfh_global_.setNoise (0.0f);
   ourcvfh_global_.setUseCache (false);
   ourcvfh_global_.setAcceptHypThreshold (0.5f);
   ourcvfh_global_.setMaxDescDistance (1.f);
   ourcvfh_global_.initialize (force_retrain);

   {
   //segmentation parameters for recognition
   vfh_estimator->setCVFHParams (0.15f, 0.015f, 2.5f);

   vfh_estimator->setAxisRatio (0.5f);
   vfh_estimator->setMinAxisValue (0.5f);
   }*/

  //*****************************************
  //TRAINING USING MODELS
  //*****************************************

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> model_clouds;
  {
    std::vector<std::string> files;
    std::string start = "";
    std::string ext = std::string ("ply");
    bf::path dir = models_dir;
    getModelsInDirectory (dir, start, files, ext);
    for (size_t i = 0; i < files.size (); i++)
    {
      std::cout << files[i] << std::endl;
      std::stringstream filestr;
      filestr << models_dir << "/" << files[i];
      std::string file = filestr.str ();
      pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud (new pcl::PointCloud<pcl::PointXYZ>);
      faat_pcl::rec_3d_framework::uniform_sampling (file, 100000, *model_cloud, model_scale);
      model_clouds.push_back(model_cloud);
    }
  }

  //*****************************************
  //RECOGNITION
  //*****************************************

  //Read files from query_dir
  bf::path input = query_dir;
  std::vector<std::string> files_to_recognize;

  if (bf::is_directory (input))
  {
    std::vector<std::string> files;
    std::string start = "";
    std::string ext = std::string ("ply");
    bf::path dir = input;
    getModelsInDirectory (dir, start, files, ext);
    std::cout << "Number of scenes in directory is:" << files.size () << std::endl;
    for (size_t i = 0; i < files.size (); i++)
    {
      std::cout << files[i] << std::endl;
      std::stringstream filestr;
      filestr << query_dir << "/" << files[i];
      std::string file = filestr.str ();
      files_to_recognize.push_back (file);
    }

    std::sort (files_to_recognize.begin (), files_to_recognize.end ());
  }
  else
  {
    files_to_recognize.push_back (query_dir);
  }

  for (size_t i = 0; i < files_to_recognize.size (); i++)
  {
    std::cout << files_to_recognize[i] << std::endl;
    //sample ply file
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    faat_pcl::rec_3d_framework::uniform_sampling (files_to_recognize[i], 100000, *scene_cloud, model_scale);

    pcl::visualization::PCLVisualizer vis ("test");
    int v1,v2;
    vis.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
    vis.createViewPort (0.5, 0.0, 1.0, 1.0, v2);
    vis.addPointCloud (scene_cloud, "scene", v1);

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree (0.01);
    octree.setInputCloud(scene_cloud);
    octree.addPointsFromInputCloud ();
    int leaf_count = octree.getLeafCount();
    std::cout << "leaf count" << leaf_count << std::endl;
    for(size_t k=0; k < model_clouds.size(); k++) {

      //octree.setOccupiedVoxelsAtPointsFromCloud(model_clouds[k]);
      int occupied = 0;
      for(size_t kk=0; kk < model_clouds[k]->points.size(); kk++) {
        if(octree.isVoxelOccupiedAtPoint(model_clouds[k]->points[kk])) {
          occupied++;
        }
      }

      if( (occupied / static_cast<float>(leaf_count)) > 0.8f) {
        std::cout << "good model:" << occupied << std::endl;
        vis.addPointCloud(model_clouds[k], "model", v2);
        vis.addCoordinateSystem (1.f);
        vis.spin();
        vis.removePointCloud("model", v2);
      } else {
        std::cout << occupied << " " << scene_cloud->points.size() << " " << occupied / static_cast<float>(scene_cloud->points.size()) << std::endl;
        //vis.spinOnce (10);
      }
      //vis.removePointCloud("model", v2);
    }
    //recognize

  }
}
