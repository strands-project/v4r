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

#include <faat_pcl/object_modelling/ggo.h>

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
  bool rgb = false;
  float data_scale = 1.f;
  std::string graph_dir = "graph";

  pcl::console::parse_argument (argc, argv, "-graph_dir", graph_dir);
  pcl::console::parse_argument (argc, argv, "-rgb", rgb);

  if (rgb)
  {
    faat_pcl::object_modelling::ggo<pcl::PointXYZRGB> ggo;
    ggo.readGraph (graph_dir);
    ggo.process ();
  }
  else
  {
    faat_pcl::object_modelling::ggo<pcl::PointXYZ> ggo;
    ggo.readGraph (graph_dir);
    ggo.process ();
  }
}
