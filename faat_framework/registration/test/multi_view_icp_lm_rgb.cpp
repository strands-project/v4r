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
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>
#include "pcl/visualization/pcl_visualizer.h"
#include <faat_pcl/3d_rec_framework/feature_wrapper/normal_estimator.h>
#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/integral_image_normal.h>
#include <numeric>

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
//  bool use_cg = false;
//  bool survival_of_the_fittest = true;
//  bool point_to_plane_baseline = false;
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

  typedef pcl::PointXYZRGB PointType;
  typedef pcl::PointXYZRGB PointT;
  boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
  normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
  normal_estimator->setCMR (false);
  normal_estimator->setDoVoxelGrid (true);
  normal_estimator->setRemoveOutliers (true);
  normal_estimator->setMinNRadius (27);
  normal_estimator->setValuesForCMRFalse (voxel_grid_size, 0.018f);

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

      pcl::PointCloud<pcl::Normal>::Ptr normals_oed (new pcl::PointCloud<pcl::Normal>);
      pcl::NormalEstimationOMP<PointType, pcl::Normal> normal_estimation;
      //normal_estimation.setNormalEstimationMethod (pcl::IntegralImageNormalEstimation<PointType, pcl::Normal>::AVERAGE_3D_GRADIENT);
      normal_estimation.setInputCloud (cloud);
      //normal_estimation.setNormalSmoothingSize (10.0);
      normal_estimation.setRadiusSearch(0.02f);
      //normal_estimation.setBorderPolicy (pcl::IntegralImageNormalEstimation<PointType, pcl::Normal>::BORDER_POLICY_MIRROR);
      normal_estimation.compute (*normals_oed);

      //compute canny edges
      pcl::OrganizedEdgeFromRGBNormals<pcl::PointXYZRGB, pcl::Normal, pcl::Label> oed;
      oed.setDepthDisconThreshold (0.03f);
      oed.setInputNormals(normals_oed);
      oed.setRGBCannyLowThreshold(200.f);
      oed.setRGBCannyHighThreshold (230.f);
      oed.setEdgeType (//pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDING
                       //| pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_NAN_BOUNDARY
                       //| pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDED
                       pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_RGB_CANNY
                       //| pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_HIGH_CURVATURE
                       );
      oed.setInputCloud (cloud);

      pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
      std::vector<pcl::PointIndices> indices2;
      oed.compute (*labels, indices2);

      pcl::PointCloud<PointType>::Ptr edges (new pcl::PointCloud<PointType>);
      std::cout << "Number of edge channels:" << indices2.size () << std::endl;
      for (size_t j = 0; j < indices2.size (); j++)
      {
        for (size_t i = 0; i < indices2[j].indices.size (); i++)
        {
          PointType pl;
          pl = cloud->points[indices2[j].indices[i]];

          if(pcl_isfinite(pl.z) && pcl_isfinite(pl.x) && pcl_isfinite(pl.y))
            edges->push_back (pl);
        }
      }

      //filter edge regions that are not dense enough
      pcl::PointCloud<PointType>::Ptr filtered_edges (new pcl::PointCloud<PointType>());

      /*pcl::RadiusOutlierRemoval<PointType> ror;
      ror.setRadiusSearch(0.01f);
      ror.setMinNeighborsInRadius(9);
      ror.setInputCloud(edges);
      ror.filter(*filtered_edges);*/

      pcl::StatisticalOutlierRemoval<PointType> ror;
      ror.setMeanK(10);
      ror.setStddevMulThresh(1.f);
      ror.setInputCloud(edges);
      ror.filter(*filtered_edges);

      pcl::PointCloud<PointType>::Ptr points (new pcl::PointCloud<PointType>(*filtered_edges));
//      float curv_thres = 0.01f;
      if(filtered_edges->points.size() > 0)
      {
        std::vector<int> pointIdxNKNSearch;
        std::vector<float> pointNKNSquaredDistance;
        float inlier = 0.02f;
        pcl::octree::OctreePointCloudSearch<PointT> octree (0.003);
        octree.setInputCloud (filtered_edges);
        octree.addPointsFromInputCloud ();

        for(size_t k=0; k < cloud->points.size(); k++)
        {
          if(pcl_isfinite(cloud->points[k].z) && pcl_isfinite(cloud->points[k].x) && pcl_isfinite(cloud->points[k].y))
          {


            /*if(!pcl_isnan(normals_oed->points[k].curvature) && normals_oed->points[k].curvature < curv_thres)
              continue;*/

            if (octree.nearestKSearch (cloud->points[k], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
              float d = sqrt (pointNKNSquaredDistance[0]);
              if (d > inlier)
              {
                points->push_back(cloud->points[k]);
              }
            }
          }
        }
      }
      else
      {
        PCL_WARN("There are no color edges...\n");
        for(size_t k=0; k < cloud->points.size(); k++)
        {
          if(pcl_isfinite(cloud->points[k].z) && pcl_isfinite(cloud->points[k].x) && pcl_isfinite(cloud->points[k].y))
          {
            /*if(pcl_isnan(normals_oed->points[k].curvature) || normals_oed->points[k].curvature < curv_thres)
              continue;*/

            points->push_back(cloud->points[k]);
          }
        }
      }

      /*clouds[i].reset(new pcl::PointCloud<PointType>());
      pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
      normal_estimator->estimate(cloud, clouds[i], normals);
      normals_clouds.push_back(normals);*/

      clouds[i].reset(new pcl::PointCloud<PointType>(*points));
    }
  }

  if(visualize)
  {
    pcl::visualization::PCLVisualizer vis("test");
    for(size_t i=0; i < clouds.size(); i++)
    {
      std::stringstream cloud_name;
      cloud_name << "cloud_" << i;
      pcl::visualization::PointCloudColorHandlerRGBField<PointType> rand_h(clouds[i]);
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

  PointT min_pt_all, max_pt_all;
  min_pt_all.x = min_pt_all.y = min_pt_all.z = std::numeric_limits<float>::max ();
  max_pt_all.x = max_pt_all.y = max_pt_all.z = (std::numeric_limits<float>::max () - 0.001f) * -1;

  for (size_t i = 0; i < clouds.size (); i++)
  {
    PointT min_pt, max_pt;
    pcl::getMinMax3D (*clouds[i], min_pt, max_pt);
    if (min_pt.x < min_pt_all.x)
      min_pt_all.x = min_pt.x;

    if (min_pt.y < min_pt_all.y)
      min_pt_all.y = min_pt.y;

    if (min_pt.z < min_pt_all.z)
      min_pt_all.z = min_pt.z;

    if (max_pt.x > max_pt_all.x)
      max_pt_all.x = max_pt.x;

    if (max_pt.y > max_pt_all.y)
      max_pt_all.y = max_pt.y;

    if (max_pt.z > max_pt_all.z)
      max_pt_all.z = max_pt.z;
  }

  float res_occupancy_grid_ = 0.01f;
  int size_x, size_y, size_z;
  size_x = static_cast<int> (std::ceil (std::abs (max_pt_all.x - min_pt_all.x) / res_occupancy_grid_)) + 1;
  size_y = static_cast<int> (std::ceil (std::abs (max_pt_all.y - min_pt_all.y) / res_occupancy_grid_)) + 1;
  size_z = static_cast<int> (std::ceil (std::abs (max_pt_all.z - min_pt_all.z) / res_occupancy_grid_)) + 1;

  for (size_t i = 0; i < clouds.size (); i++)
  {
    std::vector<int> complete_cloud_occupancy_by_RM_;
    complete_cloud_occupancy_by_RM_.resize (size_x * size_y * size_z, 0);
    for(size_t k=0; k < clouds[i]->points.size(); k++)
    {
      int pos_x, pos_y, pos_z;
      pos_x = static_cast<int> (std::floor ((clouds[i]->points[k].x - min_pt_all.x) / res_occupancy_grid_));
      pos_y = static_cast<int> (std::floor ((clouds[i]->points[k].y - min_pt_all.y) / res_occupancy_grid_));
      pos_z = static_cast<int> (std::floor ((clouds[i]->points[k].z - min_pt_all.z) / res_occupancy_grid_));
      int idx = pos_z * size_x * size_y + pos_y * size_x + pos_x;
      complete_cloud_occupancy_by_RM_[idx] = 1;
    }

    int total_points_i = std::accumulate(complete_cloud_occupancy_by_RM_.begin(),
                                          complete_cloud_occupancy_by_RM_.end(), 0);

    std::vector<int> complete_cloud_occupancy_by_RM_j;
    complete_cloud_occupancy_by_RM_j.resize (size_x * size_y * size_z, 0);

    for (size_t j = i; j < clouds.size (); j++)
    {
      int overlap = 0;
      std::map<int, bool> banned;
      std::map<int, bool>::iterator banned_it;

      for(size_t k=0; k < clouds[j]->points.size(); k++)
      {
        int pos_x, pos_y, pos_z;
        pos_x = static_cast<int> (std::floor ((clouds[j]->points[k].x - min_pt_all.x) / res_occupancy_grid_));
        pos_y = static_cast<int> (std::floor ((clouds[j]->points[k].y - min_pt_all.y) / res_occupancy_grid_));
        pos_z = static_cast<int> (std::floor ((clouds[j]->points[k].z - min_pt_all.z) / res_occupancy_grid_));
        int idx = pos_z * size_x * size_y + pos_y * size_x + pos_x;
        banned_it = banned.find (idx);
        if (banned_it == banned.end ())
        {
          complete_cloud_occupancy_by_RM_j[idx] = 1;
          if(complete_cloud_occupancy_by_RM_[idx] > 0)
            overlap++;
          banned[idx] = true;
        }
      }

      float total_points_j = std::accumulate(complete_cloud_occupancy_by_RM_j.begin(),
                                              complete_cloud_occupancy_by_RM_j.end(), 0);

      float ov_measure_1 = overlap / static_cast<float>(total_points_j);
      float ov_measure_2 = overlap / static_cast<float>(total_points_i);
      float ff = 0.3f;
      if(!(ov_measure_1 > ff || ov_measure_2 > ff))
      {
        A[i][j] = false;
        A[j][i] = false;
      }
    }
  }

  /*std::vector<int> pointIdxNKNSearch;
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
  }*/

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

    pcl::PointCloud<PointType>::Ptr big_cloud_aligned;
    big_cloud_aligned.reset(new pcl::PointCloud<PointType>);

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

      *big_cloud_aligned += *trans_cloud;
    }

    pcl::visualization::PCLVisualizer vis("aligned");
    int v1, v2;
    vis.createViewPort(0,0,0.5,1,v1);
    vis.createViewPort(0.5,0,1,1,v2);

    {
      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rand_h(big_cloud_aligned);
      vis.addPointCloud<PointT>(big_cloud_aligned, rand_h, "aligned", v1);
    }

    pcl::PointCloud<PointType>::Ptr filtered (new pcl::PointCloud<PointType>());
    pcl::StatisticalOutlierRemoval<PointType> ror;
    ror.setMeanK(10);
    ror.setStddevMulThresh(1.f);
    ror.setInputCloud(big_cloud_aligned);
    ror.filter(*filtered);

    {
      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rand_h(filtered);
      vis.addPointCloud<PointT>(filtered, rand_h, "filtered", v2);
    }

    vis.spin();
  }
}
