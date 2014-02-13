/*
 * objectness_3d.cpp
 *
 *  Created on: Oct 23, 2012
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <faat_pcl/segmentation/objectness_3d/objectness_3D.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/time.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/planar_polygon_fusion.h>
#include <pcl/segmentation/plane_coefficient_comparator.h>
#include <pcl/segmentation/euclidean_plane_coefficient_comparator.h>
#include <pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl/segmentation/edge_aware_plane_comparator.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#include "objectness_test.h"

int
main (int argc, char ** argv)
{
  int num_sampled_wins_ = 3000000;
  int num_wins_ = 500;
  std::string shrink_ = std::string ("0.66,0.66,0.5");
  int angle_incr_ = 15;
  float objectness_threshold_ = 0.75f;
  std::string pcd_file = "";
  float z_dist_ = 1.5f;
  int rows_ = 480;
  int cols_ = 640;
  int do_z = false;
  bool detect_clutter_ = true;
  bool optimize_ = true;
  float expand_factor = 1.5f;

  pcl::console::parse_argument (argc, argv, "-n_sampled_wins_", num_sampled_wins_);
  pcl::console::parse_argument (argc, argv, "-n_wins", num_wins_);
  pcl::console::parse_argument (argc, argv, "-shrink", shrink_);
  pcl::console::parse_argument (argc, argv, "-angle_incr", angle_incr_);
  pcl::console::parse_argument (argc, argv, "-objectness_threshold_", objectness_threshold_);
  pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
  pcl::console::parse_argument (argc, argv, "-z_dist", z_dist_);
  pcl::console::parse_argument (argc, argv, "-do_z", do_z);
  pcl::console::parse_argument (argc, argv, "-detect_clutter", detect_clutter_);
  pcl::console::parse_argument (argc, argv, "-optimize", optimize_);
  pcl::console::parse_argument (argc, argv, "-expand_factor", expand_factor);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz_points (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile (pcd_file, *xyz_points);
  Eigen::Vector4f table_plane;
  computeTablePlane (xyz_points, table_plane, z_dist_);

  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

  {
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimation;
    normal_estimation.setNormalEstimationMethod (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::AVERAGE_3D_GRADIENT);
    normal_estimation.setInputCloud (xyz_points);
    normal_estimation.setNormalSmoothingSize (10.0);
    normal_estimation.setBorderPolicy (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::BORDER_POLICY_MIRROR);
    normal_estimation.compute (*normals);
  }

  std::vector<int> edge_indices;
  pcl::PointCloud<pcl::PointXYZL>::Ptr label_cloud (new pcl::PointCloud<pcl::PointXYZL>);
  computeEdges (xyz_points, normals, label_cloud, edge_indices, table_plane, z_dist_, cols_);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz_points_andy (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PassThrough<pcl::PointXYZRGB> pass_;
  pass_.setFilterLimits (0.3f, z_dist_);
  pass_.setFilterFieldName ("z");
  pass_.setInputCloud (xyz_points);
  pass_.setKeepOrganized (true);
  pass_.filter (*xyz_points_andy);

  //filtering using x...
  pass_.setInputCloud (xyz_points_andy);
  pass_.setFilterLimits (-0.4f, 0.4f);
  pass_.setFilterFieldName ("x");
  pass_.filter (*xyz_points_andy);

  faat_pcl::segmentation::Objectness3D<pcl::PointXYZRGB> o3d (shrink_, num_sampled_wins_, num_wins_, 3, 41, angle_incr_, objectness_threshold_);
  o3d.setInputCloud (xyz_points_andy);
  o3d.setTablePlane (table_plane);
  o3d.setEdges (edge_indices);
  o3d.setEdgeLabelsCloud (label_cloud);
  o3d.setDoOptimize(optimize_);
  o3d.setExpandFactor(expand_factor);

  //smooth segmentation of the cloud
  if (detect_clutter_)
  {

    float size_voxels = 0.005f;
    pcl::PointCloud<pcl::PointXYZL>::Ptr clusters_cloud_;
    computeSuperPixels<pcl::PointXYZRGB>(xyz_points_andy, clusters_cloud_, size_voxels);
    o3d.setSmoothLabelsCloud (clusters_cloud_);
  }

  o3d.setVisualize (true);
  o3d.doZ (do_z);
  o3d.computeObjectness (true);

  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters;
  std::vector<pcl::PointIndices> indices;
  o3d.getObjectIndices (indices, xyz_points);
  for (size_t i = 0; i < indices.size (); i++)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud (*xyz_points, indices[i].indices, *cluster);
    clusters.push_back (cluster);
  }

  //visualize segmentation
  pcl::visualization::PCLVisualizer vis_ ("segmentation");
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (xyz_points);
  vis_.addPointCloud<pcl::PointXYZRGB> (xyz_points, handler_rgb, "scene_cloud_segmentation");
  vis_.spinOnce (100.f, true);

  for (size_t i = 0; i < clusters.size (); i++)
  {
    if (indices[i].indices.size () < 500)
      continue;

    std::stringstream name;
    name << "cluster_" << i;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud (*xyz_points, indices[i].indices, *cluster);

    pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZRGB> handler_rgb (cluster);
    vis_.addPointCloud<pcl::PointXYZRGB> (cluster, handler_rgb, name.str ());
  }

  vis_.spin ();
}
