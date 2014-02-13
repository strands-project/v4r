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
#include <faat_pcl/registration/fast_icp_with_gc.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
//#include <faat_pcl/registration/lm_icp.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace pcl;
int
main (int argc, char ** argv)
{
  std::string cloud1, cloud2;
  float chop = 1.25f;

  pcl::console::parse_argument (argc, argv, "-cloud1", cloud1);
  pcl::console::parse_argument (argc, argv, "-cloud2", cloud2);
  pcl::console::parse_argument (argc, argv, "-chop", chop);
  typedef pcl::PointXYZRGB PointType;
  pcl::PointCloud<PointType>::Ptr cloud_1 (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr cloud_2 (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr cloud_11 (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr cloud_22 (new pcl::PointCloud<PointType>);

  pcl::io::loadPCDFile (cloud1, *cloud_11);
  pcl::io::loadPCDFile (cloud2, *cloud_22);

  pcl::PassThrough<PointType> pass_;
  pass_.setFilterLimits (0.f, chop);
  pass_.setFilterFieldName ("z");
  pass_.setKeepOrganized (true);

  pass_.setInputCloud (cloud_11);
  pass_.filter (*cloud_1);

  pass_.setInputCloud (cloud_22);
  pass_.filter (*cloud_2);

  /*pcl::visualization::PCLVisualizer vis ("TEST");
  int v1, v2;
  vis.createViewPort (0, 0, 0.5, 1, v1);
  vis.createViewPort (0.5, 0, 1, 1, v2);
  vis.setBackgroundColor(255,255,255);
  vis.spinOnce(true);

  {
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler (cloud_1);
    vis.addPointCloud (cloud_1, handler, "cloud_1", v1);
  }

  {
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler (cloud_2);
    vis.addPointCloud (cloud_2, handler, "cloud_2");
  }

  vis.spin();*/

  faat_pcl::registration::FastIterativeClosestPointWithGC<PointType> icp;
  icp.setUseNormals(true);
  icp.setOverlapPercentage(0.3f);
  icp.setInputSource(cloud_1);
  icp.setInputTarget(cloud_2);
  icp.useStandardCG (true);
  icp.setKeepMaxHypotheses (1);
  icp.setMaximumIterations (5);
  icp.align();

  float w_after_icp_ = 0;
  Eigen::Matrix4f icp_trans = Eigen::Matrix4f::Identity();
  w_after_icp_ = icp.getFinalTransformation (icp_trans);
  std::cout << w_after_icp_ << std::endl;

  /*{
    pcl::transformPointCloud(*cloud_1, *cloud_11, icp_trans);
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler (cloud_11);
    vis.addPointCloud (cloud_11, handler, "cloud_11", v2);
  }

  vis.spin();*/
}
