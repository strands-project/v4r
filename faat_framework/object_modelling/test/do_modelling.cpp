/*
 * do_modelling.cpp
 *
 *  Created on: Mar 15, 2013
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "seg_do_modelling.h"
#include <faat_pcl/object_modelling/object_modeller.h>
#include <faat_pcl/utils/filesystem_utils.h>

#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/integral_image_normal.h>

inline bool
readMatrixFromFile2 (std::string file, Eigen::Matrix4f & matrix, int ignore = 0)
{

  std::ifstream in;
  in.open (file.c_str (), std::ifstream::in);

  char linebuf[1024];
  in.getline (linebuf, 1024);
  std::string line (linebuf);
  std::vector < std::string > strs_2;
  boost::split (strs_2, line, boost::is_any_of (" "));

  int c = 0;
  for (int i = ignore; i < (ignore + 16); i++, c++)
  {
    matrix (c / 4, c % 4) = static_cast<float> (atof (strs_2[i].c_str ()));
  }

  return true;
}

int
main (int argc, char ** argv)
{
  float Z_DIST_ = 1.5f;
  std::string pcd_files_dir_;
  bool refine_feature_poses = false;
  bool sort_pcd_files_ = true;
  bool use_max_cluster_ = true;
  bool use_gc_icp_ = false;
  bool bf_pairwise = true;
  bool fittest = false;
  float data_scale = 1.f;
  float ov_percent = 0.5f;
  float x_limits = 0.4f;
  std::string graph_dir = "graph";
  int num_plane_inliers = 500;
  bool single_object = true;
  bool vis_final_ = true;
  bool pose_estimate = false;
  std::string aligned_output_dir = "";
  pcl::console::parse_argument (argc, argv, "-pcd_files_dir", pcd_files_dir_);
  pcl::console::parse_argument (argc, argv, "-refine_feature_poses", refine_feature_poses);
  pcl::console::parse_argument (argc, argv, "-sort_pcd_files", sort_pcd_files_);
  pcl::console::parse_argument (argc, argv, "-use_max_cluster", use_max_cluster_);
  pcl::console::parse_argument (argc, argv, "-use_gc_icp", use_gc_icp_);
  pcl::console::parse_argument (argc, argv, "-bf_pairwise", bf_pairwise);
  pcl::console::parse_argument (argc, argv, "-fittest", fittest);
  pcl::console::parse_argument (argc, argv, "-data_scale", data_scale);
  pcl::console::parse_argument (argc, argv, "-graph_dir", graph_dir);
  pcl::console::parse_argument (argc, argv, "-ov_percent", ov_percent);
  pcl::console::parse_argument (argc, argv, "-x_limits", x_limits);
  pcl::console::parse_argument (argc, argv, "-Z_DIST", Z_DIST_);
  pcl::console::parse_argument (argc, argv, "-num_plane_inliers", num_plane_inliers);
  pcl::console::parse_argument (argc, argv, "-single_object", single_object);
  pcl::console::parse_argument (argc, argv, "-vis_final", vis_final_);
  pcl::console::parse_argument (argc, argv, "-pose_estimate", pose_estimate);
  pcl::console::parse_argument (argc, argv, "-aligned_output_dir", aligned_output_dir);

  std::vector<std::string> files;
  std::string start = "";
  std::string ext = std::string ("pcd");
  bf::path dir = pcd_files_dir_;
  faat_pcl::utils::getFilesInDirectory (dir, start, files, ext);
  std::cout << "Number of scenes in directory is:" << files.size () << std::endl;

  typedef pcl::PointXYZRGB PointType;
  typedef pcl::PointXYZRGBNormal PointTypeNormal;

  std::vector< pcl::PointCloud<PointType>::Ptr > clouds_;
  clouds_.resize(files.size());

  if(sort_pcd_files_)
    std::sort(files.begin(), files.end());

  pcl::visualization::PCLVisualizer vis("");

  std::vector<pcl::PointCloud<PointType>::Ptr> range_images_;
  std::vector<pcl::PointCloud<PointType>::Ptr> edges_;
  std::vector<std::vector<int> > obj_indices_;
  std::vector<std::string> pose_files;

  for(size_t i=0; i < files.size(); i++)
  {
    pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType>);
    std::stringstream file_to_read;
    file_to_read << pcd_files_dir_ << "/" << files[i];
    pcl::io::loadPCDFile (file_to_read.str(), *scene);

    std::cout << file_to_read.str() << std::endl;
    if(pose_estimate)
    {
      std::string pose_file = file_to_read.str();
      boost::replace_all(pose_file, ".pcd", ".txt");
      boost::replace_all(pose_file, "cloud", "pose");
      bf::path pose_file_path = pose_file;
      if(bf::exists(pose_file_path))
      {
        pose_files.push_back(pose_file);
      }

      //compute depth edges discontinuities...
      pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
      pcl::IntegralImageNormalEstimation<PointType, pcl::Normal> normal_estimation;
      normal_estimation.setNormalEstimationMethod (pcl::IntegralImageNormalEstimation<PointType, pcl::Normal>::AVERAGE_3D_GRADIENT);
      normal_estimation.setInputCloud (scene);
      normal_estimation.setNormalSmoothingSize (10.0);
      normal_estimation.setBorderPolicy (pcl::IntegralImageNormalEstimation<PointType, pcl::Normal>::BORDER_POLICY_MIRROR);
      normal_estimation.compute (*normals);

      pcl::OrganizedEdgeFromRGBNormals<pcl::PointXYZRGB, pcl::Normal, pcl::Label> oed;
      oed.setDepthDisconThreshold (0.03f);
      /*oed.setHCCannyLowThreshold (0.4f);
      oed.setHCCannyHighThreshold (1.2f);*/
      oed.setInputNormals (normals);
      oed.setEdgeType (pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDING
                       //| pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_NAN_BOUNDARY
                       //| pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDED
                       //| pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_RGB_CANNY
                       //| pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_HIGH_CURVATURE
                       );
      oed.setInputCloud (scene);

      pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
      std::vector<pcl::PointIndices> indices2;
      oed.compute (*labels, indices2);

      pcl::PointCloud<PointType>::Ptr edges (new pcl::PointCloud<PointType>);
      std::cout << "Number of edge channels:" << indices2.size () << std::endl;
      for (size_t j = 0; j < indices2.size (); j++)
      {
        for (size_t i = 0; i < indices2[j].indices.size (); i++)
        {
          pcl::PointXYZRGB pl;
          pl = scene->points[indices2[j].indices[i]];
          edges->push_back (pl);
        }
      }

      edges_.push_back(edges);
    }

    //segment the object of interest
    pcl::PassThrough<PointType> pass_;
    pass_.setFilterLimits (0.f, Z_DIST_);
    pass_.setFilterFieldName ("z");
    pass_.setInputCloud (scene);
    pass_.setKeepOrganized (true);
    pass_.filter (*scene);

    if(x_limits > 0)
    {
      pass_.setInputCloud (scene);
      pass_.setFilterLimits (-x_limits, x_limits);
      pass_.setFilterFieldName ("x");
      pass_.filter (*scene);
    }

    std::vector<pcl::PointIndices> indices;
    Eigen::Vector4f table_plane;
    doSegmentation<PointType>(scene, indices, table_plane, num_plane_inliers);

    std::cout << "Number of clusters found:" << indices.size() << std::endl;
    range_images_.push_back(scene);

    if(single_object)
    {
      std::cout << "selecting max..." << std::endl;
      pcl::PointIndices max;
      for (size_t k = 0; k < indices.size (); k++)
      {
        if(max.indices.size() < indices[k].indices.size())
        {
          max = indices[k];
        }
      }

      pcl::PointCloud<PointType>::Ptr obj_interest (new pcl::PointCloud<PointType>);
      obj_interest->width = scene->width;
      obj_interest->height = scene->height;
      obj_interest->points.resize(scene->points.size());
      for(size_t k=0; k < obj_interest->points.size(); k++)
      {
        obj_interest->points[k].z = std::numeric_limits<float>::quiet_NaN();
      }

      for (size_t k = 0; k < max.indices.size (); k++)
      {
        obj_interest->points[max.indices[k]] = scene->points[max.indices[k]];
      }

      //pcl::copyPointCloud(*scene, max, *obj_interest);
      obj_indices_.push_back(indices[0].indices);
      clouds_[i] = obj_interest;
    }
    else
    {
      std::vector<int> obj_indices;
      pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
      for (size_t k = 0; k < indices.size (); k++)
      {
        pcl::PointCloud<PointType>::Ptr obj_interest (new pcl::PointCloud<PointType>);
        pcl::copyPointCloud(*scene, indices[k], *obj_interest);
        *cloud += *obj_interest;
        obj_indices.insert(obj_indices.end(), indices[k].indices.begin(), indices[k].indices.end());
      }
      obj_indices_.push_back(obj_indices);
      clouds_[i] = cloud;

    }
  }

  if(pose_estimate)
  {
    pcl::visualization::PCLVisualizer aligned_vis("aligned");
    for(size_t i=0; i < pose_files.size(); i++)
    {
      Eigen::Matrix4f pose;
      readMatrixFromFile2(pose_files[i], pose, 1);
      std::cout << pose << std::endl;
      std::cout << pose.inverse() << std::endl;
      pose = pose.inverse();
      pcl::PointCloud<PointType>::Ptr cloud_trans (new pcl::PointCloud<PointType>);
      pcl::transformPointCloud(*clouds_[i], *cloud_trans, pose);

      std::stringstream cloud_name;
      cloud_name << "cloud_" << i << ".pcd";

      std::stringstream out_name;
      out_name << aligned_output_dir << "/" << cloud_name.str();

      pcl::PointCloud<PointType>::Ptr cloud_sor (new pcl::PointCloud<PointType>);

      /*pcl::StatisticalOutlierRemoval<PointType> sor;
      sor.setMeanK (50);
      sor.setStddevMulThresh (2);
      sor.setInputCloud(cloud_trans);
      sor.filter(*cloud_sor);*/

      std::vector<int> pointIdxNKNSearch;
      std::vector<float> pointNKNSquaredDistance;
      pcl::octree::OctreePointCloudSearch<PointType> octree (0.003);
      octree.setInputCloud (edges_[i]);
      octree.addPointsFromInputCloud ();

      std::vector<int> indices_to_keep;
      std::vector<int> indices_set_to_nan;
      for(size_t k=0; k < clouds_[i]->points.size(); k++)
      {
        if(!pcl_isfinite(clouds_[i]->points[k].z))
          continue;

        if (octree.nearestKSearch (clouds_[i]->points[k], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
          float d = sqrt (pointNKNSquaredDistance[0]);
          if (d > 0.0015f)
          {
            indices_to_keep.push_back(static_cast<int>(k));
          }
          else
          {
            indices_set_to_nan.push_back(static_cast<int>(k));
          }
        }
      }

      //pcl::copyPointCloud(*cloud_trans, indices_to_keep, *cloud_sor);
      pcl::copyPointCloud(*cloud_trans, *cloud_sor);
      for(size_t k=0; k < indices_set_to_nan.size(); k++)
      {
        cloud_sor->points[indices_set_to_nan[k]].z = std::numeric_limits<float>::quiet_NaN();
      }

      pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (cloud_sor);
      aligned_vis.addPointCloud<PointType>(cloud_sor, handler_rgb, cloud_name.str());

      pcl::io::savePCDFileBinary(out_name.str().c_str(), *cloud_sor);
    }

    //aligned_vis.spin();
    exit(0);
  }

  faat_pcl::object_modelling::ObjectModeller<flann::L1, PointType, PointTypeNormal> om;

  for(size_t i=0; i < clouds_.size(); i++)
  {
    std::stringstream cloud_name;
    cloud_name << "cloud_" << i;
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (clouds_[i]);
    vis.addPointCloud<PointType>(clouds_[i], handler_rgb, cloud_name.str());

    om.addInputCloud(clouds_[i]);
    std::cout << files[i] << std::endl;
  }
  vis.spin();

  om.setVisFinal(vis_final_);
  om.setRangeImages(range_images_, obj_indices_, 525.f, 640.f, 480.f);
  om.setOverlapPercentage(ov_percent);
  om.setUseMaxClusterOnly(use_max_cluster_);
  om.setUseGCICP(use_gc_icp_);
  om.setBFPairwise(bf_pairwise);
  om.setICPGCSurvivalOfTheFittest(fittest);
  //om.computeFeatures();
  om.setRefineFeaturesPoses(refine_feature_poses);
  om.processClouds();
  //om.visualizeProcessed();
  //om.computePairWiseRelativePosesWithFeatures();
  //om.visualizePairWiseAlignment();
  om.computeRelativePosesWithICP();
  //om.visualizePairWiseAlignment();
  //om.globalSelectionWithMST(false);
  //om.visualizeGlobalAlignment(false);
  om.saveGraph(graph_dir);
  //om.visualizePairWiseAlignment();

  /*om.computeFeatures();
  om.setUseMaxClusterOnly(use_max_cluster_);
  om.setRefineFeaturesPoses(refine_feature_poses);
  om.visualizeProcessed();
  om.computePairWiseRelativePosesWithFeatures();
  //om.visualizePairWiseAlignment();
  //om.computeRelativePosesWithICP();
  //om.visualizePairWiseAlignment();
  om.globalSelectionWithMST(false);
  om.visualizeGlobalAlignment(true);
  om.visualizePairWiseAlignment();*/
}
