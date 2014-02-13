/*
 * test_linemod.cpp
 *
 *  Created on: Oct 29, 2013
 *      Author: aitor
 */

#include <pcl/recognition/linemod/line_rgbd.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <faat_pcl/recognition/hv/hv_go_3D.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <faat_pcl/3d_rec_framework/defines/faat_3d_rec_framework_defines.h>
#include <algorithm>    // std::random_shuffle
#include <faat_pcl/utils/pcl_opencv.h>
#include <faat_pcl/3d_rec_framework/pc_source/registered_views_source.h>
#include <faat_pcl/3d_rec_framework/pc_source/partial_pcd_source.h>
#include <faat_pcl/3d_rec_framework/pipeline/linemod3d_recognizer.h>

namespace bf = boost::filesystem;

struct camPosConstraints
{
  bool
  operator() (const Eigen::Vector3f & pos) const
  {
    if (pos[2] > 0)
      return true;

    return false;
  }
  ;
};

//./bin/linemod_framework -models_dir /home/aitor/data/willow_object_clouds/models_ml_final_reduced/ -training_dir_linemod /home/aitor/data/willow_linemod -training_input_structure /home/aitor/data/willow_structure_final/ -pcd_file /home/aitor/data/willow_test/T_28/cloud_0000000001.pcd

int
main (int argc, char ** argv)
{
  std::string training_input_structure;
  std::string detect_on_;
  int n_detections_to_show_ = 10;
  std::string path = "";
  std::string desc_name = "linemod";
  std::string training_dir_linemod = "";
  std::string pcd_file;
  bool use_3dmodels = false;
  std::string training_dir_linemod_3d_models;
  int tes_level_ = 1;
  float distance = 1.f;

  pcl::console::parse_argument (argc, argv, "-models_dir", path);
  pcl::console::parse_argument (argc, argv, "-training_input_structure", training_input_structure);
  pcl::console::parse_argument (argc, argv, "-training_dir_linemod", training_dir_linemod);

  pcl::console::parse_argument (argc, argv, "-detect_on_", detect_on_);
  pcl::console::parse_argument (argc, argv, "-n_detections", n_detections_to_show_);
  pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
  pcl::console::parse_argument (argc, argv, "-use_3dmodels", use_3dmodels);
  pcl::console::parse_argument (argc, argv, "-training_dir_linemod_3d_models", training_dir_linemod_3d_models);
  pcl::console::parse_argument (argc, argv, "-tes_level", tes_level_);
  pcl::console::parse_argument (argc, argv, "-distance", distance);

  typedef pcl::PointXYZRGBA PointT;
  boost::shared_ptr<faat_pcl::rec_3d_framework::Source<PointT> > cast_source;


  if(use_3dmodels)
  {
    boost::function<bool (const Eigen::Vector3f &)> campos_constraints;
    campos_constraints = camPosConstraints ();

    boost::shared_ptr<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, PointT, PointT> >
                                                                                                               source (
                                                                                                                       new faat_pcl::rec_3d_framework::PartialPCDSource<
                                                                                                                           pcl::PointXYZRGBNormal,
                                                                                                                           PointT, PointT>);
    source->setPath (path);
    source->setModelScale (1.f);
    source->setRadiusSphere (distance);
    source->setTesselationLevel (tes_level_);
    source->setDotNormal (0.f);
    source->setGenOrganized(true);
    source->setLoadViews (true);
    source->setCamPosConstraints (campos_constraints);
    source->setWindowSizeAndFocalLength(640, 480, 575.f);
    source->genInPlaneRotations(true, 45.f);
    source->setLoadIntoMemory (false);
    source->generate (training_dir_linemod_3d_models);
    cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, PointT, PointT> > (source);
  }
  else
  {
    boost::shared_ptr<faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> >
                                                                                                                                            mesh_source (
                                                                                                                                                         new faat_pcl::rec_3d_framework::RegisteredViewsSource<
                                                                                                                                                             pcl::PointXYZRGBNormal,
                                                                                                                                                             PointT, PointT>);
    mesh_source->setPath (path);
    mesh_source->setModelStructureDir (training_input_structure);
    mesh_source->generate (training_dir_linemod);

    cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > (mesh_source);
  }


  boost::shared_ptr<faat_pcl::rec_3d_framework::LineMod3DPipeline< PointT > > local;
  local.reset(new faat_pcl::rec_3d_framework::LineMod3DPipeline<PointT> ());
  local->setDataSource (cast_source);
  local->setTrainingDir (training_dir_linemod);
  if(use_3dmodels)
  {
    local->setTrainingDir (training_dir_linemod_3d_models);
  }
  local->setDescriptorName (desc_name);
  local->setUseCache (static_cast<bool> (false));
  local->setICPIterations (0);
  local->initialize (static_cast<bool> (false));

  pcl::PointCloud<PointT>::Ptr cloud_to_recog(new pcl::PointCloud<PointT>);
  pcl::io::loadPCDFile(pcd_file, *cloud_to_recog);
  std::cout << "finished loading file" << std::endl;
  local->setInputCloud(cloud_to_recog);
  local->recognize();
}
