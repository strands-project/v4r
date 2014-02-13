/*
 * labeling_from_orgt.cpp
 *
 *  Created on: Oct 15, 2013
 *      Author: aitor
 */

#include <faat_pcl/3d_rec_framework/tools/or_evaluator.h>
#include <faat_pcl/3d_rec_framework/segmentation/multiplane_segmentation.h>
#include <faat_pcl/3d_rec_framework/pc_source/partial_pcd_source.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/console/parse.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/passthrough.h>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>

namespace bf = boost::filesystem;

//./bin/labeling_from_orgt -input_dir /home/aitor/data/willow_dataset/ -models_dir /home/aitor/data/willow_object_clouds/models_ml_new/ -training_dir /home/aitor/data/mp_willow_trained -GT_DIR /home/aitor/data/willow_dataset_gt/ -z_dist 2 -output_dir /home/aitor/data/willow_dataset_labels/ -visualize 1

int
main (int argc, char ** argv)
{
  float Z_DIST_ = 2.5f;
  std::string GT_DIR_, models_dir_, input_dir_, output_dir_;
  std::string training_dir_;
  bool visualize = false;
  output_dir_ = "";

  pcl::console::parse_argument (argc, argv, "-GT_DIR", GT_DIR_);
  pcl::console::parse_argument (argc, argv, "-models_dir", models_dir_);
  pcl::console::parse_argument (argc, argv, "-input_dir", input_dir_);
  pcl::console::parse_argument (argc, argv, "-output_dir", output_dir_);
  pcl::console::parse_argument (argc, argv, "-training_dir", training_dir_);
  pcl::console::parse_argument (argc, argv, "-z_dist", Z_DIST_);
  pcl::console::parse_argument (argc, argv, "-visualize", visualize);

  if(output_dir_.compare("") != 0)
  {
    bf::path p = output_dir_;
    if(!bf::exists(p))
      bf::create_directory(p);
  }

  typedef pcl::PointXYZRGB PointT;
  boost::shared_ptr<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, PointT> >
                                                                                                   source (
                                                                                                           new faat_pcl::rec_3d_framework::PartialPCDSource<
                                                                                                               pcl::PointXYZRGBNormal, PointT>);
  source->setPath (models_dir_);
  source->setModelScale (1.f);
  source->setLoadViews (false);
  source->setLoadIntoMemory (false);
  source->generate (training_dir_);

  boost::shared_ptr<faat_pcl::rec_3d_framework::Source<PointT> > cast_source;
  cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, PointT> > (source);

  //take all scenes in input_dir_
  //load GT data for scene, backproject models and mark pixels
  //fit planes
  faat_pcl::rec_3d_framework::or_evaluator::OREvaluator<PointT> or_eval;
  or_eval.setGTDir (GT_DIR_);
  or_eval.setModelsDir (models_dir_);
  or_eval.setModelFileExtension ("pcd");
  or_eval.setReplaceModelExtension (false);

  bf::path input = input_dir_;
  std::vector<std::string> files_to_label;

  if (bf::is_directory (input_dir_))
  {
    std::vector<std::string> files;
    std::string start = "";
    std::string ext = std::string ("pcd");
    bf::path dir = input;
    faat_pcl::utils::getFilesInDirectory (dir, start, files, ext);
    std::cout << "Number of scenes in directory is:" << files.size () << std::endl;
    for (size_t i = 0; i < files.size (); i++)
    {
      typename pcl::PointCloud<PointT>::Ptr scene_cloud (new pcl::PointCloud<PointT>);
      std::cout << files[i] << std::endl;
      std::stringstream filestr;
      filestr << input_dir_ << files[i];
      std::string file = filestr.str ();
      files_to_label.push_back (file);
    }

    std::sort (files_to_label.begin (), files_to_label.end ());
    or_eval.setScenesDir (input_dir_);
    or_eval.setDataSource (cast_source);
    or_eval.loadGTData ();
  }
  else
  {
    PCL_ERROR("Expecting a directory as input!!, aborting\n");
    exit (-1);
  }

  pcl::visualization::PCLVisualizer vis ("Labeling results");
  int v1,v2,v3;
  vis.createViewPort (0.0, 0.0, 0.33, 1, v1);
  vis.createViewPort (0.33, 0, 0.66, 1, v2);
  vis.createViewPort (0.66, 0, 1, 1, v3);

  std::cout << files_to_label.size () << std::endl;
  for (size_t i = 0; i < files_to_label.size (); i++)
  {
    std::cout << "recognizing " << files_to_label[i] << std::endl;
    typename pcl::PointCloud<PointT>::Ptr scene (new pcl::PointCloud<PointT>);
    pcl::io::loadPCDFile (files_to_label[i], *scene);

    pcl::PassThrough<PointT> pass_;
    pass_.setFilterLimits (0.f, Z_DIST_);
    pass_.setFilterFieldName ("z");
    pass_.setKeepOrganized (true);
    pass_.setInputCloud (scene);
    pass_.filter (*scene);

    {
      pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler (scene);
      vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud_z_coloured", v1);
    }

    std::string file_to_recognize(files_to_label[i]);
    boost::replace_all (file_to_recognize, input_dir_, "");
    boost::replace_all (file_to_recognize, ".pcd", "");

    std::string id_1 = file_to_recognize;
    or_eval.visualizeGroundTruth(vis, id_1, v2);

    faat_pcl::MultiPlaneSegmentation<PointT> mps;
    mps.setInputCloud(scene);
    mps.setMinPlaneInliers(1000);
    mps.setMergePlanes(true);
    std::vector<faat_pcl::PlaneModel<PointT> > planes_found;
    mps.segment();
    planes_found = mps.getModels();

    if(planes_found.size() == 0 && scene->isOrganized())
    {
      mps.segment(true);
      planes_found = mps.getModels();
    }

    for(size_t kk=0; kk < planes_found.size(); kk++)
    {
      std::stringstream pname;
      pname << "plane_" << kk;

      pcl::visualization::PointCloudColorHandlerRandom<PointT> scene_handler(planes_found[kk].plane_cloud_);
      vis.addPointCloud<PointT> (planes_found[kk].plane_cloud_, scene_handler, pname.str(), v2);

      pname << "chull";
      vis.addPolygonMesh (*planes_found[kk].convex_hull_, pname.str(), v2);
    }

    pcl::PointCloud<pcl::PointXYZL>::Ptr labels(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::copyPointCloud(*scene, *labels);
    int UNKNOWN_LABEL_ = 0;
    int PLANE_LABEL_ = 1;
    int KNOWN_LABEL_ = 2;

    for(size_t j=0; j < labels->points.size(); j++)
      labels->points[j].label = UNKNOWN_LABEL_;

    for(size_t kk=0; kk < planes_found.size(); kk++)
    {
      for(size_t j=0; j < planes_found[kk].inliers_.indices.size(); j++)
      {
        labels->points[planes_found[kk].inliers_.indices[j]].label = PLANE_LABEL_;
      }
    }

    int cx_, cy_;
    float focal_length_ = 525.f;
    float thres = 0.01f;

    cx_ = 640;
    cy_ = 480;

    float cx, cy;
    cx = static_cast<float> (cx_) / 2.f - 0.5f;
    cy = static_cast<float> (cy_) / 2.f - 0.5f;

    std::vector< pcl::PointCloud<PointT>::Ptr > model_clouds;
    or_eval.getModelsForScene(id_1, model_clouds, 0.003f);

    std::vector< pcl::PointCloud<PointT>::Ptr > model_clouds_full_res;
    or_eval.getModelsForScene(id_1, model_clouds_full_res, -1);

    pcl::IterativeClosestPoint < PointT, PointT > icp;
    icp.setInputTarget (scene);

    for(size_t j=0; j < model_clouds.size(); j++)
    {
      //do a small icp stage
      /*icp.setInputSource (model_clouds[j]);
      icp.setMaxCorrespondenceDistance(0.015f);
      icp.setMaximumIterations(5);
      icp.setRANSACIterations(1000);
      icp.setEuclideanFitnessEpsilon(1e-9);
      icp.setTransformationEpsilon(1e-9);
      pcl::PointCloud < PointT >::Ptr model_aligned( new pcl::PointCloud<PointT> );
      icp.align (*model_aligned);

      Eigen::Matrix4f icp_trans;
      icp_trans = icp.getFinalTransformation();

      pcl::transformPointCloud(*model_clouds_full_res[j], *model_aligned, icp_trans);*/

      pcl::PointCloud < PointT >::Ptr model_aligned( new pcl::PointCloud<PointT>(*model_clouds_full_res[j]) );

      //backproject points and assign label if points is close enough
      for(size_t k=0; k < model_aligned->points.size(); k++)
      {
          float x = model_aligned->points[k].x;
          float y = model_aligned->points[k].y;
          float z = model_aligned->points[k].z;
          int u = static_cast<int> (focal_length_ * x / z + cx);
          int v = static_cast<int> (focal_length_ * y / z + cy);

          if (u >= cx_ || v >= cy_ || u < 0 || v < 0)
            continue;

          if (!pcl_isfinite(labels->at(u,v).z))
            continue;

          if( (model_aligned->points[k].getVector3fMap() - labels->at(u,v).getVector3fMap()).norm() > thres)
            continue;

          labels->at(u, v).label = KNOWN_LABEL_;
      }
    }

    {
      pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> scene_handler (labels, "label");
      vis.addPointCloud<pcl::PointXYZL> (labels, scene_handler, "labels", v3);
    }

    if(output_dir_.compare("") != 0)
    {

      std::string file_to_recognize(files_to_label[i]);
      boost::replace_all (file_to_recognize, input_dir_, "");

      std::stringstream path;
      path << output_dir_;
      std::vector<std::string> strs;
      boost::split (strs, file_to_recognize, boost::is_any_of ("/"));

      if(strs.size() > 1)
      {
        path << strs[strs.size() - 2];
      }
      bf::path p = path.str();
      if(!bf::exists(p))
        bf::create_directory(p);

      {
        std::stringstream path;
        path << output_dir_ << file_to_recognize;
        std::cout << path.str() << std::endl;
        pcl::io::savePCDFileBinary(path.str(), *labels);
      }
    }

    if(visualize)
      vis.spin();
    else
      vis.spinOnce(500, true);

    vis.removeAllPointClouds();
    vis.removeAllShapes();
  }
}
