/*
 * do_modelling.cpp
 *
 *  Created on: Mar 15, 2013
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <faat_pcl/object_modelling/object_modeller.h>

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
  std::string graph_dir = "graph";
  float ov_percent = 0.5f;
  bool rotate_axis = false;
  float dt_vx_size_ = 0.003f;
  bool vis_final_ = true;

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
  pcl::console::parse_argument (argc, argv, "-rotate_axis", rotate_axis);
  pcl::console::parse_argument (argc, argv, "-dt_vx_size", dt_vx_size_);
  pcl::console::parse_argument (argc, argv, "-vis_final", vis_final_);

  std::vector<std::string> files;
  std::string start = "";
  std::string ext = std::string ("pcd");
  bf::path dir = pcd_files_dir_;
  faat_pcl::utils::getFilesInDirectory (dir, start, files, ext);
  std::cout << "Number of scenes in directory is:" << files.size () << std::endl;

  typedef pcl::PointXYZ PointType;
  typedef pcl::PointNormal PointTypeNormal;

  std::vector< pcl::PointCloud<PointType>::Ptr > clouds_;
  clouds_.resize(files.size());

  if(sort_pcd_files_)
    std::sort(files.begin(), files.end());
  else
    std::random_shuffle(files.begin(), files.end());

  pcl::visualization::PCLVisualizer vis("");

  for(size_t i=0; i < files.size(); i++)
  {
    pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType>);
    std::stringstream file_to_read;
    file_to_read << pcd_files_dir_ << "/" << files[i];
    pcl::io::loadPCDFile (file_to_read.str(), *scene);

    if(data_scale != 1.f)
    {
      for(size_t k=0; k < scene->points.size(); k++)
      {
        scene->points[k].getVector3fMap() *= data_scale;
      }
    }

    clouds_[i] = scene;
    std::cout << scene->isOrganized() << std::endl;
  }

  faat_pcl::object_modelling::ObjectModeller<flann::L1, PointType, PointTypeNormal> om;

  Eigen::Matrix4f m_x;
  m_x.setIdentity();
  m_x.block<3,3>(0,0) = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitY()).toRotationMatrix();

  for(size_t i=0; i < clouds_.size(); i++)
  {

    if(rotate_axis)
      pcl::transformPointCloud(*clouds_[i], *clouds_[i], m_x);

    std::stringstream cloud_name;
    cloud_name << "cloud_" << i;
    pcl::visualization::PointCloudColorHandlerRandom<PointType> handler_rgb (clouds_[i]);
    vis.addPointCloud<PointType>(clouds_[i], handler_rgb, cloud_name.str());

    om.addInputCloud(clouds_[i]);
    std::cout << files[i] << std::endl;
  }

  vis.addCoordinateSystem(0.5f);
  vis.spin();

  om.setDtVxSize(dt_vx_size_);
  om.setVisFinal(vis_final_);
  om.setOverlapPercentage(ov_percent);
  om.setUseMaxClusterOnly(use_max_cluster_);
  om.setUseGCICP(use_gc_icp_);
  om.setBFPairwise(bf_pairwise);
  om.setICPGCSurvivalOfTheFittest(fittest);
  //om.computeFeatures();
  om.processClouds();
  om.setRefineFeaturesPoses(refine_feature_poses);
  om.visualizeProcessed();
  //om.computePairWiseRelativePosesWithFeatures();
  //om.visualizePairWiseAlignment();
  om.computeRelativePosesWithICP();
  //om.visualizePairWiseAlignment();
  //om.globalSelectionWithMST(false);
  //om.visualizeGlobalAlignment(true);
  om.saveGraph(graph_dir);
  //om.visualizePairWiseAlignment();
}
