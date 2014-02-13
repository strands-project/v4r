/*
 * mit.cpp
 *
 *  Created on: Feb 20, 2013
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <faat_pcl/3d_rec_framework/pc_source/registered_views_pp_source.h>
#include <faat_pcl/3d_rec_framework/pc_source/partial_pcd_source.h>
#include <faat_pcl/3d_rec_framework/pipeline/local_recognizer.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/shot_local_estimator_omp.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/fpfh_local_estimator_omp.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/recognition/cg/correspondence_grouping.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <faat_pcl/recognition/cg/graph_geometric_consistency.h>
#include <pcl/recognition/hv/hv_papazov.h>
#include <faat_pcl/recognition/hv/hv_go.h>
#include <pcl/recognition/hv/greedy_verification.h>
#include <pcl/apps/dominant_plane_segmentation.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/fast_bilateral.h>

struct XYZRedGreenBlue
 {
   float x;                  // preferred way of adding a XYZ+padding
   float y;
   float z;
   //PCL_ADD_POINT4D;
   float red;
   float green;
   float blue;
   //EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
 }; //EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

 POINT_CLOUD_REGISTER_POINT_STRUCT (XYZRedGreenBlue,           // here we assume a XYZ + "test" (as fields)
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (float, red, red)
                                    (float, green, green)
                                    (float, blue, blue)
)

inline void
transformXYZRedGreenBlueToXYZRGB(pcl::PointCloud<XYZRedGreenBlue> & orig, pcl::PointCloud<pcl::PointXYZRGB> & dst)
{
   pcl::copyPointCloud(orig, dst);
   for(size_t i=0; i < orig.points.size(); i++)
   {
        //dst.points[i].getVector3fMap() = orig.points[i].getVector3fMap();
        dst.points[i].r = static_cast<int>(orig.points[i].red);
        dst.points[i].g = static_cast<int>(orig.points[i].green);
        dst.points[i].b = static_cast<int>(orig.points[i].blue);
   }
}

/*inline void
transformXYZRedGreenBlueToXYZRGB(pcl::PointCloud<XYZRedGreenBlue> & orig, pcl::PointCloud<pcl::PointXYZRGB> & dst)
{
   dst.points.resize(orig.points.size());

   dst.width = orig.width;
   dst.height = orig.height;

   for(size_t u=0; u < orig.width; u++) {
     for(size_t v=0; v < orig.height; v++) {
       dst.at(u,v).r = static_cast<int>(orig.at(u,v).red);
       dst.at(u,v).g = static_cast<int>(orig.at(u,v).green);
       dst.at(u,v).b = static_cast<int>(orig.at(u,v).blue);
     }
   }

   for(size_t i=0; i < orig.points.size(); i++) {
     dst.points[i].getVector3fMap() = orig.points[i].getVector3fMap();
     int ii = static_cast<int>(i);
     int u = ii / orig.width;
     int v = ii % orig.height;

     dst.at(v,u).r = static_cast<int>(orig.points[i].red);
     dst.at(v,u).g = static_cast<int>(orig.points[i].green);
     dst.at(v,u).b = static_cast<int>(orig.points[i].blue);

     dst.points[orig.points.size() - (i+1)].r = static_cast<int>(orig.points[i].red);
     dst.points[orig.points.size() - (i+1)].g = static_cast<int>(orig.points[i].green);
     dst.points[orig.points.size() - (i+1)].b = static_cast<int>(orig.points[i].blue);
     std::cout << "r" << (int)(dst.points[i].r) << " " <<  orig.points[i].red << std::endl;
     std::cout << "g" << (int)(dst.points[i].g) << " " <<  orig.points[i].green << std::endl;
     std::cout << "b" << (int)(dst.points[i].b) << " " <<  orig.points[i].blue << std::endl;
   }
}*/

inline bool
sortFiles (const std::string & file1, const std::string & file2)
{
  std::vector < std::string > strs1;
  boost::split (strs1, file1, boost::is_any_of ("/"));

  std::vector < std::string > strs2;
  boost::split (strs2, file2, boost::is_any_of ("/"));

  std::string id_1 = strs1[strs1.size () - 1];
  std::string id_2 = strs2[strs2.size () - 1];

  size_t pos1 = id_1.find (".pcd");
  size_t pos2 = id_2.find (".pcd");

  id_1 = id_1.substr (0, pos1);
  id_2 = id_2.substr (0, pos2);

  id_1 = id_1.substr (2);
  id_2 = id_2.substr (2);

  return atoi (id_1.c_str ()) < atoi (id_2.c_str ());
}

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

void
getDesiredPaths (bf::path & path_with_views, std::vector<std::string> & view_filenames, std::string & view_prefix_, std::string & ext)
{
  int number_of_views = 0;
  bf::directory_iterator end_itr;
  for (bf::directory_iterator itr (path_with_views); itr != end_itr; ++itr)
  {
    if (!(bf::is_directory (*itr)))
    {
      std::vector < std::string > strs;
      std::vector < std::string > strs_;

#if BOOST_FILESYSTEM_VERSION == 3
      std::string file = (itr->path ().filename ()).string();
#else
      std::string file = (itr->path ()).filename ();
#endif

      boost::split (strs, file, boost::is_any_of ("."));
      boost::split (strs_, file, boost::is_any_of ("_"));

      std::string extension = strs[strs.size () - 1];

      if ((extension.compare (ext) == 0) && ( boost::algorithm::starts_with(strs_[0], view_prefix_) || (strs_[0].compare (view_prefix_) == 0) ))
      {
#if BOOST_FILESYSTEM_VERSION == 3
        view_filenames.push_back ((itr->path ().filename ()).string());
#else
        view_filenames.push_back ((itr->path ()).filename ());
#endif

        number_of_views++;
      }
    }
  }
}

template<typename PointT>
void visualizeGroundTruth(std::string & scene_file, pcl::visualization::PCLVisualizer & vis, int viewport, std::string & models_folder,
                          std::vector<typename pcl::PointCloud<PointT>::Ptr> & gt_models_aligned,
                          std::vector<std::string> & gt_ids)
{
  std::vector < std::string > strs;
  boost::split (strs, scene_file, boost::is_any_of ("/"));
  std::string cloud_fn = strs[strs.size () - 1];

  std::string model_path (scene_file);
  boost::replace_all (model_path, cloud_fn, "");

  strs.clear();
  boost::split (strs, cloud_fn, boost::is_any_of ("."));
  std::string cloud_n = strs[0];

  std::cout << scene_file << " " << cloud_fn << " " << cloud_n << std::endl;

  std::vector < std::string > pose_filenames;
  bf::path model_dir = model_path;
  std::string ext = std::string("txt");
  getDesiredPaths(model_dir, pose_filenames, cloud_n, ext);

  for(size_t i=0; i < pose_filenames.size(); i++)
  {

    std::vector < std::string > strs;
    boost::split (strs, pose_filenames[i], boost::is_any_of ("_"));
    std::string model_name = strs[1];
    boost::replace_all (model_name, ".txt", "");

    std::stringstream file_to_read;
    file_to_read << model_path << "/" << pose_filenames[i];
    std::cout << file_to_read.str() << " " << model_name << std::endl;

    std::string file_pose(file_to_read.str());
    //std::ifstream infile(file_pose.c_str(), ifstream::in);
    boost::replace_all (file_pose, "///", "/");
    std::ifstream in;
    std::cout << "Trying to open..." << file_pose << std::endl;
    in.open (file_pose.c_str (), std::ifstream::in);

    if(in) {

      char linebuf[256];
      in.getline (linebuf, 256);
      std::string pose_line (linebuf);
      std::cout << pose_line << std::endl;

      in.close();

      strs.clear();
      boost::split (strs, pose_line, boost::is_any_of (" "));

      std::vector<float> non_empty;
      for(size_t k=0; k < strs.size(); k++)
      {
        if(strs[k] != "") {
          non_empty.push_back(atof(strs[k].c_str()));
        }
      }

      Eigen::Vector3f trans(non_empty[0],non_empty[1],non_empty[2]);
      Eigen::Quaternionf rot(non_empty[3],non_empty[4],non_empty[5],non_empty[6]);

      Eigen::Matrix3f rot_mat = rot.toRotationMatrix();
      Eigen::Matrix4f pose_mat;
      pose_mat.block<3,3>(0,0) = rot_mat;
      pose_mat.block<3,1>(0,3) = trans;

      typename pcl::PointCloud<PointT>::Ptr gt_model (new pcl::PointCloud<PointT>);
      std::stringstream  model_file;
      model_file << models_folder << "/" << model_name << ".pcd";
      pcl::io::loadPCDFile (model_file.str(), *gt_model);

      pcl::transformPointCloud(*gt_model, *gt_model, pose_mat);

      std::stringstream name;
      name << "gt_model" << i;
      pcl::visualization::PointCloudColorHandlerRandom<PointT> random_handler (gt_model);
      vis.addPointCloud<PointT> (gt_model, random_handler, name.str (), viewport);

      gt_models_aligned.push_back(gt_model);
      gt_ids.push_back(model_name);
    } else {
      std::cout << "Could not read file..." << file_to_read.str() << std::endl;
    }

  }
}


bool ADD_GT_HYPOTHESES_ = false;
std::string results_folder = "/home/aitor/data/jared/MIT-clutter-2012-test_results/";

template<template<class > class DistT, typename PointT, typename FeatureT>
void
recognizeAndVisualize (typename faat_pcl::rec_3d_framework::LocalRecognitionPipeline<DistT, PointT, FeatureT> & local,
                          std::string & scene_file,
                          boost::shared_ptr<faat_pcl::GlobalHypothesesVerification<PointT, PointT> > & go,
                          std::string & models_folder)
{

  typename boost::shared_ptr<faat_pcl::rec_3d_framework::Source<PointT> > model_source_ = local.getDataSource ();
  typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointInTPtr;
  typedef faat_pcl::rec_3d_framework::Model<PointT> ModelT;
  typedef boost::shared_ptr<ModelT> ModelTPtr;

  pcl::visualization::PCLVisualizer vis ("Recognition results");
  /*int v1, v2, v3;
  vis.createViewPort (0.0, 0.0, 0.33, 1.0, v1);
  vis.createViewPort (0.33, 0, 0.66, 1.0, v2);
  vis.createViewPort (0.66, 0, 1.0, 1.0, v3);*/

  int v1, v2, v3, v4;
  vis.createViewPort (0.0, 0.0, 0.25, 1.0, v1);
  vis.createViewPort (0.25, 0.0, 0.5, 1.0, v2);
  vis.createViewPort (0.5, 0.0, 0.75, 1.0, v3);
  vis.createViewPort (0.75, 0.0, 1.0, 1.0, v4);

  bf::path input = scene_file;
  std::vector<std::string> files_to_recognize;

  if (bf::is_directory (input))
  {
    std::vector < std::string > files;
    std::string start = "";
    std::string ext = std::string ("pcd");
    bf::path dir = input;
    getModelsInDirectory (dir, start, files, ext);
    std::cout << "Number of scenes in directory is:" << files.size () << std::endl;
    for (size_t i = 0; i < files.size (); i++)
    {
      typename pcl::PointCloud<PointT>::Ptr scene_cloud (new pcl::PointCloud<PointT>);
      std::cout << files[i] << std::endl;
      std::stringstream filestr;
      filestr << scene_file << "/" << files[i];
      std::string file = filestr.str ();
      files_to_recognize.push_back (file);
    }

    std::sort(files_to_recognize.begin(),files_to_recognize.end());
  }
  else
  {
    files_to_recognize.push_back (scene_file);
  }

  for(size_t i=0; i < files_to_recognize.size(); i++) {
    typename pcl::PointCloud<PointT>::Ptr scene (new pcl::PointCloud<PointT>);
    pcl::io::loadPCDFile (files_to_recognize[i], *scene);

    /*pcl::PointCloud<XYZRedGreenBlue>::Ptr scene_with_rgb (new pcl::PointCloud<XYZRedGreenBlue>);
    pcl::io::loadPCDFile (files_to_recognize[i], *scene_with_rgb);*/

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile (files_to_recognize[i], *scene_rgb);
    //transformXYZRedGreenBlueToXYZRGB(*scene_with_rgb, *scene_rgb);

    /*typename pcl::PointCloud<PointT>::Ptr scene_fbf (new pcl::PointCloud<PointT>);
    pcl::FastBilateralFilter<PointT> fbf;
    fbf.setInputCloud(scene);
    fbf.setSigmaS(3);
    fbf.setSigmaR(0.01);
    fbf.filter(*scene_fbf);*/

    typename pcl::PointCloud<PointT>::Ptr scene_pass_through (new pcl::PointCloud<PointT>);
    pcl::PassThrough<PointT> pass_;
    pass_.setFilterLimits (0.f, 2.5f);
    pass_.setFilterFieldName ("z");
    //pass_.setInputCloud (scene_fbf);
    pass_.setInputCloud (scene);
    pass_.setKeepOrganized (false);
    pass_.filter (*scene_pass_through);

    typename pcl::PointCloud<PointT>::Ptr scene_vx_grid (new pcl::PointCloud<PointT>);
    float voxel_grid_size = 0.003f;
    pcl::VoxelGrid<PointT> grid_;
    grid_.setInputCloud (scene_pass_through);
    grid_.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
    grid_.setDownsampleAllData (true);
    grid_.filter (*scene_vx_grid);

    pcl::apps::DominantPlaneSegmentation<PointT> dps;
    dps.setMaxZBounds(2.f);
    dps.setInputCloud(scene_vx_grid);
    dps.compute_table_plane();
    Eigen::Vector4f table_plane;
    dps.getTableCoefficients(table_plane);

    std::vector<int> indices;
    {
      for (int i = 0; i < scene_vx_grid->points.size (); i++)
      {
        Eigen::Vector3f xyz_p = scene_vx_grid->points[i].getVector3fMap ();

        if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
          continue;

        float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

        if (val >= 0.005 && xyz_p[2] < 1.2f)
        {
          indices.push_back (static_cast<int> (i));
        }
      }
    }

    typename pcl::PointCloud<PointT>::Ptr scene_no_plane (new pcl::PointCloud<PointT>);
    //pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler (scene_vx_grid, 125, 125, 125);
    //vis.addPointCloud<PointT> (scene_vx_grid, scene_handler, "scene_cloud", v1);

    {
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (scene_rgb);
      vis.addPointCloud<pcl::PointXYZRGB> (scene_rgb, handler_rgb, "scene_rgb", v1);
    }

    {
      pcl::copyPointCloud(*scene_vx_grid, indices, *scene_no_plane);
      pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler (scene_vx_grid, 125, 125, 125);
      //vis.addPointCloud<PointT> (scene_vx_grid, scene_handler, "scene_cloud_no_plane", v2);
    }

    local.setIndices(indices);
    local.setInputCloud (scene_vx_grid);
    {
      pcl::ScopeTime ttt ("Recognition");
      local.recognize ();
    }

    std::cout << "to recognize:" << files_to_recognize[i] << std::endl;
    std::string file = files_to_recognize[i];
    boost::replace_all (file, scene_file, results_folder);

    std::stringstream save_to;
    std::string cloud_name = file;
    boost::replace_all (cloud_name, ".pcd", "");
    save_to << cloud_name << "_";
    std::cout << save_to.str() << std::endl;

    vis.addText (files_to_recognize[i], 1, 30, 14, 1, 0, 0, "scene_text", v1);

    std::vector<typename pcl::PointCloud<PointT>::Ptr> gt_models_aligned;
    std::vector<std::string> gt_ids;
    visualizeGroundTruth<PointT>(files_to_recognize[i], vis, v4, models_folder, gt_models_aligned, gt_ids);

    //visualize results
    boost::shared_ptr < std::vector<ModelTPtr> > models = local.getModels ();
    boost::shared_ptr < std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms = local.getTransforms ();

    std::vector<std::string> ids;
    std::vector<ConstPointInTPtr> aligned_models;
    //aligned_models.resize (models->size ());

    for (size_t j = 0; j < models->size (); j++) {
      ids.push_back(models->at(j)->id_);

      ConstPointInTPtr model_cloud = models->at (j)->getAssembled (go->getResolution());
      typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
      pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (j));

      //aligned_models[j] = (model_aligned);
      aligned_models.push_back(model_aligned);
    }

    if(ADD_GT_HYPOTHESES_)
    {
      for (size_t j = 0; j < gt_models_aligned.size (); j++)
      {
        ids.push_back(gt_ids[j]);
        aligned_models.push_back(gt_models_aligned[j]);
      }
    }

    go->setSceneCloud (scene_vx_grid);
    std::vector<bool> mask_hv;
    go->addModels (aligned_models, true);
    go->setObjectIds(ids);
    go->verify ();
    go->getMask (mask_hv);

    for (size_t j = 0; j < aligned_models.size (); j++)
    {

      if(!mask_hv[j])
        continue;

      std::stringstream name;
      name << "cloud_" << j;

      /*ConstPointInTPtr model_cloud = models->at (j)->getAssembled (0.003f);
      typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
      pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (j));*/

      pcl::visualization::PointCloudColorHandlerRandom<PointT> random_handler (aligned_models[j]);
      vis.addPointCloud<PointT> (aligned_models[j], random_handler, name.str (), v2);

      std::stringstream nametext;
      nametext << "cloud_" << j;
      nametext << "_text";
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid (*aligned_models[j], centroid);
      pcl::PointXYZ ptext;
      ptext.getVector4fMap () = centroid;

      std::string text = ids[j];
      boost::replace_all (text, ".pcd", "");

      vis.addText3D (text, ptext, 0.01, 255, 255, 0, nametext.str (), v2);

      std::stringstream save_to_pose;
      save_to_pose << save_to.str();
      save_to_pose << text << ".txt";
      std::cout << save_to_pose.str() << std::endl;

      faat_pcl::rec_3d_framework::PersistenceUtils::writeMatrixToFile(save_to_pose.str(), transforms->at(j));
      /*pcl::visualization::PointCloudColorHandlerCustom<PointT> random_handler (model_aligned, r, g, b);
      vis.addPointCloud<PointT> (model_aligned, random_handler, name.str (), v2);*/
    }

    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr smooth_cloud_ =  go->getSmoothClusters();
      pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> random_handler (smooth_cloud_, "intensity");
      vis.addPointCloud<pcl::PointXYZI> (smooth_cloud_, random_handler, "smooth_cloud", v3);
    }

    vis.setBackgroundColor(0.0,0.0,0.0);
    /*if(PLAY_) {
      vis.spinOnce (100.f, true);
    } else {*/
      vis.spin ();
    //}


    vis.removeAllPointClouds(v4);
    vis.removePointCloud ("smooth_cloud");
    vis.removePointCloud ("scene_rgb");
    vis.removePointCloud ("scene_cloud_no_plane");
    vis.removePointCloud ("scene_cloud");
    vis.removeShape ("scene_text");
    vis.removeAllPointClouds(v2);
    vis.removeAllShapes();

    /*for (size_t j = 0; j < models->size (); j++)
    {
      std::stringstream name;
      name << "cloud_" << j;
      vis.removePointCloud (name.str ());
    }*/
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
  float CG_SIZE_ = 0.005f;
  int CG_THRESHOLD_ = 3;
  float keypoint_density = 0.01f;
  float support_radius = 0.04f;
  float thres_acc_hyp_ = 0.f;
  bool use_cb_shot = false;
  int icp_iterations = 10;
  float go_inlier_threshold = 0.0075f;
  float go_regularizer = 2.5f;
  float go_clutter_regularizer = 2.5f;
  int go_opt_type = 0;
  float GO_RES_ = 0.005f;
  float go_radius_clutter = 0.05f;
  bool use_vertices = true;
  float keypoint_density_recog_ = 0.005f;
  float ransac_threshold_cg_ = CG_THRESHOLD_;
  int kdtree_splits = 128;

  std::string flann_index_fn = std::string("index_flann.txt");
  std::string cb_index_fn =std::string("index_codebook.txt");

  std::string desc_used = "shot";
  bool use_gc_graph = true;
  pcl::console::parse_argument (argc, argv, "-use_gc_graph", use_gc_graph);
  pcl::console::parse_argument (argc, argv, "-use_cb_shot", use_cb_shot);
  pcl::console::parse_argument (argc, argv, "-flann_index_fn", flann_index_fn);
  pcl::console::parse_argument (argc, argv, "-cb_index_fn", cb_index_fn);
  pcl::console::parse_argument (argc, argv, "-gc_size", CG_SIZE_);
  pcl::console::parse_argument (argc, argv, "-gc_threshold", CG_THRESHOLD_);
  pcl::console::parse_argument (argc, argv, "-training_dir", training_dir);
  pcl::console::parse_argument (argc, argv, "-models_dir", models_dir);
  pcl::console::parse_argument (argc, argv, "-query_dir", query_dir);
  pcl::console::parse_argument (argc, argv, "-force_retrain", force_retrain);
  pcl::console::parse_argument (argc, argv, "-icp_iterations", icp_iterations);
  pcl::console::parse_argument (argc, argv, "-use_vertices", use_vertices);

  pcl::console::parse_argument (argc, argv, "-go_inlier_threshold", go_inlier_threshold);
  pcl::console::parse_argument (argc, argv, "-go_regularizer", go_regularizer);
  pcl::console::parse_argument (argc, argv, "-go_clutter_regularizer", go_clutter_regularizer);
  pcl::console::parse_argument (argc, argv, "-go_opt_type", go_opt_type);
  pcl::console::parse_argument (argc, argv, "-go_res", GO_RES_);
  pcl::console::parse_argument (argc, argv, "-go_radius_clutter", go_radius_clutter);
  pcl::console::parse_argument (argc, argv, "-go_radius_normals", radius_normals_go_);
  pcl::console::parse_argument (argc, argv, "-desc_used", desc_used);
  pcl::console::parse_argument (argc, argv, "-support_radius", support_radius);
  pcl::console::parse_argument (argc, argv, "-thres_acc_hyp", thres_acc_hyp_);
  pcl::console::parse_argument (argc, argv, "-keypoint_density_recog", keypoint_density_recog_);
  pcl::console::parse_argument (argc, argv, "-ransac_threshold_cg_", ransac_threshold_cg_);
  pcl::console::parse_argument (argc, argv, "-kdtree_splits", kdtree_splits);

  typedef pcl::PointXYZRGB PointType;

  /*{
    boost::shared_ptr<faat_pcl::rec_3d_framework::RegisteredViewsWithPPSource<pcl::PointXYZRGB> > regviewspp (new faat_pcl::rec_3d_framework::RegisteredViewsWithPPSource<pcl::PointXYZRGB>);
    regviewspp->setPath (models_dir);
    std::string view_prefix = "cloud";
    regviewspp->setPrefix(view_prefix);
    regviewspp->generate (training_dir);
    exit(-1);
  }*/

  boost::shared_ptr<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, PointType, PointType> >
                                                                                                            source (
                                                                                                                    new faat_pcl::rec_3d_framework::PartialPCDSource<
                                                                                                                    pcl::PointXYZRGBNormal,
                                                                                                                    PointType, PointType>);
  source->setPath (models_dir);
  source->setModelScale (1.f);
  source->setRadiusSphere (1.f);
  source->setTesselationLevel (1);
  source->setUseVertices (use_vertices);
  source->setDotNormal (-1.f);
  source->setLoadViews (true);
  source->generate (training_dir);

  boost::shared_ptr<faat_pcl::rec_3d_framework::Source<PointType> > cast_source;
  cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, PointType, PointType  > > (source);

  //configure normal estimator
  boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointType, pcl::Normal> > normal_estimator;
  normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointType, pcl::Normal>);
  normal_estimator->setCMR (false);
  normal_estimator->setDoVoxelGrid (true);
  normal_estimator->setRemoveOutliers (false);
  normal_estimator->setValuesForCMRFalse (0.003f, 0.025f);

  //configure cg algorithm (geometric consistency grouping)
  boost::shared_ptr<pcl::CorrespondenceGrouping<PointType, PointType> > cast_cg_alg;


  if(!use_gc_graph){
    boost::shared_ptr<pcl::GeometricConsistencyGrouping<PointType, PointType> > gcg_alg (new pcl::GeometricConsistencyGrouping<PointType, PointType>);

    gcg_alg->setGCThreshold (CG_THRESHOLD_);
    gcg_alg->setGCSize (CG_SIZE_);
    //gcg_alg->setRansacThreshold(ransac_threshold_cg_);

    cast_cg_alg = boost::static_pointer_cast<pcl::CorrespondenceGrouping<PointType, PointType> > (gcg_alg);
  }
  else
  {
    boost::shared_ptr<faat_pcl::GraphGeometricConsistencyGrouping<PointType, PointType> > gcg_alg (
                                                                                                 new faat_pcl::GraphGeometricConsistencyGrouping<PointType,
                                                                                                 PointType>);
    gcg_alg->setGCThreshold (CG_THRESHOLD_);
    gcg_alg->setGCSize (CG_SIZE_);
    gcg_alg->setRansacThreshold (ransac_threshold_cg_);
    gcg_alg->setUseGraph(true);
    cast_cg_alg = boost::static_pointer_cast<pcl::CorrespondenceGrouping<PointType, PointType> > (gcg_alg);
  }

  //configure keypoint extractors
  boost::shared_ptr<faat_pcl::rec_3d_framework::UniformSamplingExtractor<PointType> >
                                                                                 uniform_keypoint_extractor (
                                                                                                             new faat_pcl::rec_3d_framework::UniformSamplingExtractor<
                                                                                                             PointType>);

  std::vector<typename boost::shared_ptr<faat_pcl::rec_3d_framework::KeypointExtractor<PointType> > > keypoint_extractors;

  boost::shared_ptr<faat_pcl::rec_3d_framework::KeypointExtractor<PointType> > keypoint_extractor;
  keypoint_extractor = boost::static_pointer_cast<faat_pcl::rec_3d_framework::KeypointExtractor<PointType> > (uniform_keypoint_extractor);
  keypoint_extractors.push_back (keypoint_extractor);

  uniform_keypoint_extractor->setSamplingDensity (keypoint_density);
  uniform_keypoint_extractor->setFilterPlanar (true);

  boost::shared_ptr<faat_pcl::GlobalHypothesesVerification<PointType, PointType> > go (
                                                                                            new faat_pcl::GlobalHypothesesVerification<PointType,PointType>);
  go->setResolution (GO_RES_);
  //go->setMaxIterations (parameters_for_go.go_iterations);
  go->setInlierThreshold (go_inlier_threshold);
  go->setRadiusClutter (go_radius_clutter);
  go->setRegularizer (go_regularizer);
  go->setClutterRegularizer (go_clutter_regularizer);
  go->setDetectClutter (true);
  go->setOcclusionThreshold (0.01f);
  go->setOptimizerType(go_opt_type);
  go->setUseReplaceMoves(1);
  //go->setInitialTemp(parameters_for_go.init_temp);
  go->setRadiusNormals(0.015f);
  //go->setRequiresNormals(parameters_for_go.require_normals);
  go->setInitialStatus(false);

  /*eps_angle_threshold_ = 0.25;
  min_points_ = 20;
  curvature_threshold_ = 0.04f;
  cluster_tolerance_ = 0.015f;*/
  go->setSmoothSegParameters(0.05f, 0.025, GO_RES_ * 4.f);

  if(desc_used == "shot")
  {
    boost::shared_ptr<faat_pcl::rec_3d_framework::SHOTLocalEstimationOMP<PointType, pcl::Histogram<352> > > estimator;
    estimator.reset (new faat_pcl::rec_3d_framework::SHOTLocalEstimationOMP<PointType, pcl::Histogram<352> >);
    estimator->setNormalEstimator (normal_estimator);
    estimator->setKeypointExtractors (keypoint_extractors);
    estimator->setSupportRadius (support_radius);
    estimator->setAdaptativeMLS (false);

    std::string desc_name = "shot";
    boost::shared_ptr<faat_pcl::rec_3d_framework::LocalEstimator<PointType, pcl::Histogram<352> > > cast_estimator;
    cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::LocalEstimator<PointType, pcl::Histogram<352> > > (estimator);

    faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointType, pcl::Histogram<352> > shot_local_;
    shot_local_.setUseCodebook(use_cb_shot);
    shot_local_.setIndexFN(flann_index_fn);
    shot_local_.setCodebookFN(cb_index_fn);
    shot_local_.setDataSource (cast_source);
    shot_local_.setTrainingDir (training_dir);
    shot_local_.setDescriptorName (desc_name);
    shot_local_.setFeatureEstimator (cast_estimator);
    shot_local_.setCGAlgorithm (cast_cg_alg);
    shot_local_.setThresholdAcceptHyp (thres_acc_hyp_);
    shot_local_.setUseCache (true);
    shot_local_.setVoxelSizeICP (0.005f);
    shot_local_.initialize (static_cast<bool> (force_retrain));
    shot_local_.setICPIterations (icp_iterations);
    shot_local_.setKdtreeSplits (kdtree_splits);
    shot_local_.setICPType(1);

    //boost::shared_ptr<pcl::HypothesisVerification<PointType, PointType> > cast_hv_alg;
    //cast_hv_alg = boost::static_pointer_cast<pcl::HypothesisVerification<PointType, PointType> > (go);
    //shot_local_.setHVAlgorithm(cast_hv_alg);

    //get whatever files to recognize in query_dir...
    uniform_keypoint_extractor->setSamplingDensity (keypoint_density_recog_);
    recognizeAndVisualize<flann::L1, PointType, pcl::Histogram<352> > (shot_local_, query_dir, go, models_dir);
  }
  else if(desc_used == "fpfh")
  {

    boost::shared_ptr<faat_pcl::rec_3d_framework::FPFHLocalEstimationOMP<PointType, pcl::FPFHSignature33 > > estimator;
    estimator.reset (new faat_pcl::rec_3d_framework::FPFHLocalEstimationOMP<PointType, pcl::FPFHSignature33 >);
    estimator->setNormalEstimator (normal_estimator);
    estimator->setKeypointExtractors (keypoint_extractors);
    estimator->setSupportRadius (support_radius);

    std::string desc_name = "fpfh";
    boost::shared_ptr<faat_pcl::rec_3d_framework::LocalEstimator<PointType, pcl::FPFHSignature33 > > cast_estimator;
    cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::LocalEstimator<PointType, pcl::FPFHSignature33 > > (estimator);

    faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointType, pcl::FPFHSignature33 > shot_local_;
    shot_local_.setUseCodebook(use_cb_shot);
    shot_local_.setIndexFN(flann_index_fn);
    shot_local_.setCodebookFN(cb_index_fn);
    shot_local_.setDataSource (cast_source);
    shot_local_.setTrainingDir (training_dir);
    shot_local_.setDescriptorName (desc_name);
    shot_local_.setFeatureEstimator (cast_estimator);
    shot_local_.setCGAlgorithm (cast_cg_alg);
    shot_local_.setThresholdAcceptHyp (thres_acc_hyp_);
    shot_local_.setUseCache (true);
    shot_local_.setVoxelSizeICP (0.005f);
    shot_local_.initialize (static_cast<bool> (force_retrain));
    shot_local_.setICPIterations (icp_iterations);
    shot_local_.setKdtreeSplits (128);
    shot_local_.setICPType(1);

    //boost::shared_ptr<pcl::HypothesisVerification<PointType, PointType> > cast_hv_alg;
    //cast_hv_alg = boost::static_pointer_cast<pcl::HypothesisVerification<PointType, PointType> > (go);
    //shot_local_.setHVAlgorithm(cast_hv_alg);

    //get whatever files to recognize in query_dir...
    uniform_keypoint_extractor->setSamplingDensity (keypoint_density_recog_);
    recognizeAndVisualize<flann::L1, PointType, pcl::FPFHSignature33 > (shot_local_, query_dir, go, models_dir);
  }
}
