/*
 * icp_with_gc.cpp
 *
 *  Created on: Mar 20, 2013
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <faat_pcl/registration/mv_lm_icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>
#include "pcl/visualization/pcl_visualizer.h"
#include <faat_pcl/3d_rec_framework/feature_wrapper/normal_estimator.h>

namespace bf = boost::filesystem;
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
      std::string so_far = rel_path_so_far + (itr->path ().filename ()).string() + "/";
#else
      std::string so_far = rel_path_so_far + (itr->path ()).filename () + "/";
#endif

      bf::path curr_path = itr->path ();
      getModelsInDirectory (curr_path, so_far, relative_paths, ext);
    }
    else
    {
      //check that it is a ply file and then add, otherwise ignore..
      std::vector < std::string > strs;
#if BOOST_FILESYSTEM_VERSION == 3
      std::string file = (itr->path ().filename ()).string();
#else
      std::string file = (itr->path ()).filename ();
#endif

      boost::split (strs, file, boost::is_any_of ("."));
      std::string extension = strs[strs.size () - 1];

      if (extension.compare (ext) == 0)
      {
#if BOOST_FILESYSTEM_VERSION == 3
        std::string path = rel_path_so_far + (itr->path ().filename ()).string();
#else
        std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif

        relative_paths.push_back (path);
      }
    }
  }
}

using namespace pcl;

//./bin/icp_nl_multi_view -clouds ml_icp_cascade/test/ -vx_size 0.003 -use_normals 0 -max_corresp_dist 0.05 -vis_intermediate 1 -sparse 0 -dt_size 0.003
int
main (int argc, char ** argv)
{

  std::string clouds_dir;
  int icp_iterations_ = 30;
  float max_corresp_dist_ = 0.1f;
  bool use_cg = false;
  bool survival_of_the_fittest = true;
  bool point_to_plane_baseline = false;
  float voxel_grid_size = 0.005f;
  float dt_size = voxel_grid_size;
  float data_scale = 1.f;
  bool use_point_to_plane = false;
  bool use_normals = false;
  bool vis_intermediate_ = true;
  bool sparse = true;
  bool save_aligned_clouds = false;
  bool visualize = true;


  pcl::console::parse_argument (argc, argv, "-clouds", clouds_dir);
  pcl::console::parse_argument (argc, argv, "-icp_iterations", icp_iterations_);
  pcl::console::parse_argument (argc, argv, "-max_corresp_dist", max_corresp_dist_);
  pcl::console::parse_argument (argc, argv, "-vx_size", voxel_grid_size);
  pcl::console::parse_argument (argc, argv, "-dt_size", dt_size);
  pcl::console::parse_argument (argc, argv, "-data_scale", data_scale);
  pcl::console::parse_argument (argc, argv, "-p2p", use_point_to_plane);
  pcl::console::parse_argument (argc, argv, "-use_normals", use_normals);
  pcl::console::parse_argument (argc, argv, "-vis_intermediate", vis_intermediate_);
  pcl::console::parse_argument (argc, argv, "-sparse", sparse);
  pcl::console::parse_argument (argc, argv, "-save_aligned_clouds", save_aligned_clouds);

  typedef pcl::PointXYZ PointType;

  /*pcl::io::loadPCDFile (cloud1, *cloud_11);
  pcl::io::loadPCDFile (cloud2, *cloud_22);
  pcl::io::loadPCDFile (cloud3, *cloud_33);*/

  typedef pcl::PointXYZ PointT;
  boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
  normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
  normal_estimator->setCMR (false);
  normal_estimator->setDoVoxelGrid (true);
  normal_estimator->setRemoveOutliers (true);
  normal_estimator->setMinNRadius (27);
  normal_estimator->setValuesForCMRFalse (voxel_grid_size, 0.018f);

  /*
  //cube toy example
  cloud_11.reset(new pcl::PointCloud<PointType>);
  cloud_11->points.resize(8);
  cloud_11->points[0].getVector3fMap() = Eigen::Vector3f(-1,-1,-1);
  cloud_11->points[1].getVector3fMap() = Eigen::Vector3f(-1,-1,1);
  cloud_11->points[2].getVector3fMap() = Eigen::Vector3f(-1,1,-1);
  cloud_11->points[3].getVector3fMap() = Eigen::Vector3f(-1,1,1);
  cloud_11->points[4].getVector3fMap() = Eigen::Vector3f(1,-1,-1);
  cloud_11->points[5].getVector3fMap() = Eigen::Vector3f(1,-1,1);
  cloud_11->points[6].getVector3fMap() = Eigen::Vector3f(1,1,-1);
  cloud_11->points[7].getVector3fMap() = Eigen::Vector3f(1,1,1);
  */

  std::vector<pcl::PointCloud<PointType>::Ptr> clouds;
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normals_clouds;
  std::vector < std::string > files;
  std::vector < std::string > files_global;
  std::string start = "";
  std::string ext = std::string ("pcd");
  bf::path dir = clouds_dir;
  getModelsInDirectory (dir, start, files, ext);
  clouds.resize(files.size());
  files_global.resize(files.size());

  for(size_t i=0; i < files.size(); i++)
  {
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    std::stringstream filestr;
    filestr << clouds_dir << "/" << files[i];
    std::string file = filestr.str ();
    pcl::io::loadPCDFile (file, *cloud);
    files_global[i] = file;

    {
      if (data_scale != 1.f)
      {
        for (size_t k = 0; k < cloud->points.size (); k++)
        {
          cloud->points[k].getVector3fMap () *= data_scale;
        }
      }

      clouds[i].reset(new pcl::PointCloud<PointType>);
      pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
      normal_estimator->estimate(cloud, clouds[i], normals);
      normals_clouds.push_back(normals);
    }
  }

  if(visualize)
  {
    pcl::visualization::PCLVisualizer vis("test");
    for(size_t i=0; i < clouds.size(); i++)
    {
      std::stringstream cloud_name;
      cloud_name << "cloud_" << i;
      pcl::visualization::PointCloudColorHandlerRandom<PointType> rand_h(clouds[i]);
      vis.addPointCloud<PointType>(clouds[i], rand_h, cloud_name.str());
    }

    vis.addCoordinateSystem(0.1f);
    vis.spin();
  }

  std::vector<std::vector<bool> > A;
  A.resize(clouds.size());
  for(size_t i=0; i < clouds.size(); i++)
  {
    A[i].resize(clouds.size(), true);
  }

  std::vector<int> pointIdxNKNSearch;
  std::vector<float> pointNKNSquaredDistance;
  float inlier = 0.01f;
  for (size_t i = 0; i < clouds.size (); i++)
  {
    A[i][i] = false;
    pcl::octree::OctreePointCloudSearch<PointT> octree (0.003);
    octree.setInputCloud (clouds[i]);
    octree.addPointsFromInputCloud ();

    for (size_t j = i; j < clouds.size (); j++)
    {
      //compute overlap
      int overlap = 0;
      for (size_t kk = 0; kk < clouds[j]->points.size (); kk++)
      {
        if(pcl_isnan(clouds[j]->points[kk].x))
          continue;

        if (octree.nearestKSearch (clouds[j]->points[kk], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
          float d = sqrt (pointNKNSquaredDistance[0]);
          if (d < inlier)
          {
            overlap++;
          }
        }
      }

      float ov_measure_1 = overlap / static_cast<float>(clouds[j]->points.size());
      float ov_measure_2 = overlap / static_cast<float>(clouds[i]->points.size());
      float ff = 0.5f;
      if(!(ov_measure_1 > ff || ov_measure_2 > ff))
      {
        A[i][j] = false;
        A[j][i] = false;
      }
    }
  }

  for (size_t i = 0; i < clouds.size (); i++)
  {
    for (size_t j = 0; j < clouds.size (); j++)
      std::cout << A[i][j] << " ";

    std::cout << std::endl;
  }

  faat_pcl::registration::MVNonLinearICP<PointType> icp_nl(dt_size);
  icp_nl.setInlierThreshold(voxel_grid_size);
  icp_nl.setMaxCorrespondenceDistance(max_corresp_dist_);
  icp_nl.setClouds(clouds);
  icp_nl.setVisIntermediate(vis_intermediate_);
  icp_nl.setSparseSolver(sparse);

  if(use_normals)
    icp_nl.setInputNormals(normals_clouds);
  icp_nl.setAdjacencyMatrix(A);
  icp_nl.compute ();

  std::vector<Eigen::Matrix4f> transformations;
  icp_nl.getTransformation(transformations);

  if(visualize)
  {
    pcl::PointCloud<PointType>::Ptr big_cloud;
    pcl::PointCloud<PointType>::Ptr big_cloud_aligned;
    big_cloud.reset(new pcl::PointCloud<PointType>);
    big_cloud_aligned.reset(new pcl::PointCloud<PointType>);


    pcl::visualization::PCLVisualizer vis("aligned");
    int v1, v2;
    vis.createViewPort(0,0,0.5,1,v1);
    vis.createViewPort(0.5,0,1,1,v2);


    int t=0;
    for (size_t i = 0; i < clouds.size (); i++)
    {
      *big_cloud += *clouds[i];
      pcl::PointCloud<PointType>::Ptr trans_cloud;
      trans_cloud.reset(new pcl::PointCloud<PointType>);
      pcl::transformPointCloud(*clouds[i], *trans_cloud, transformations[i]);
      *big_cloud_aligned += *trans_cloud;
    }

    {
      pcl::visualization::PointCloudColorHandlerCustom<PointT> rand_h(big_cloud_aligned, 255, 0, 0);
      vis.addPointCloud<PointT>(big_cloud_aligned, rand_h, "aligned", v2);
    }

    {
      pcl::visualization::PointCloudColorHandlerCustom<PointT> rand_h(big_cloud, 0, 255, 0);
      vis.addPointCloud<PointT>(big_cloud, rand_h, "initial", v1);
    }

    vis.spin();
  }

  if(save_aligned_clouds)
  {
    for (size_t i = 0; i < clouds.size (); i++)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cloud;
      trans_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr initial_cloud;
      initial_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

      pcl::io::loadPCDFile(files_global[i], *initial_cloud);
      pcl::transformPointCloud(*initial_cloud, *trans_cloud, transformations[i]);

      std::stringstream cloud_name;
      cloud_name << "aligned_cloud_" << i << ".pcd";
      pcl::io::savePCDFileBinary(cloud_name.str().c_str(), *trans_cloud);
    }
  }
}
