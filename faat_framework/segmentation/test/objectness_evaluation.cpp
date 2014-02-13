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

#define VISUALIZE_TOTAL_ 1

namespace bf = boost::filesystem;

void
getFilesInDirectory (bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext)
{
  bf::directory_iterator end_itr;
  for (bf::directory_iterator itr (dir); itr != end_itr; ++itr)
  {
    //check if its a directory, then get models in it
    if (bf::is_directory (*itr))
    {
#if BOOST_FILESYSTEM_VERSION == 3
      std::string so_far = rel_path_so_far + (itr->path ().filename ()).string () + "/";
#else
      std::string so_far = rel_path_so_far + (itr->path ()).filename () + "/";
#endif

      bf::path curr_path = itr->path ();
      getFilesInDirectory (curr_path, so_far, relative_paths, ext);
    }
    else
    {
      //check that it is a ply file and then add, otherwise ignore..
      std::vector<std::string> strs;
#if BOOST_FILESYSTEM_VERSION == 3
      std::string file = (itr->path ().filename ()).string ();
#else
      std::string file = (itr->path ()).filename ();
#endif

      boost::split (strs, file, boost::is_any_of ("."));
      std::string extension = strs[strs.size () - 1];

      if (extension.compare (ext) == 0)
      {
#if BOOST_FILESYSTEM_VERSION == 3
        std::string path = rel_path_so_far + (itr->path ().filename ()).string ();
#else
        std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif

        relative_paths.push_back (path);
      }
    }
  }
}

//./bin/objectness_3d_evaluation -test_directory /home/aitor/data/willow_test/T_07/ -labels_gt_directory /home/aitor/data/willow_ground_truth/T_07/ -angle_incr 10 -objectness_threshold_ 0.25 -z_dist 1.15 -detect_clutter 1 -optimize 1 -n_wins 50 -visualize 1 -shrink 0.66,0.66,0.66 -expand_factor 2

int
main (int argc, char ** argv)
{
  int num_sampled_wins_ = 3000000;
  int num_wins_ = 500;
  std::string shrink_ = std::string ("0.66,0.66,0.5");
  int angle_incr_ = 15;
  float objectness_threshold_ = 0.75f;
  float z_dist_ = 1.5f;
  int rows_ = 480;
  int cols_ = 640;
  int do_z = false;
  bool detect_clutter_ = true;
  bool optimize_ = true;
  float expand_factor = 1.5f;
  bool visualize = false;
  std::string eval_file = "willow_eval.txt";
  std::string pcd_file = "";
  std::string test_directory = "";
  std::string labels_gt_directory = "";
  bool do_cuda = true;
  bool objectness_ = true;
  int best_nwins = -1;
  bool cut_x = false;
  int opt_type = 1;

  pcl::console::parse_argument (argc, argv, "-n_sampled_wins_", num_sampled_wins_);
  pcl::console::parse_argument (argc, argv, "-n_wins", num_wins_);
  pcl::console::parse_argument (argc, argv, "-shrink", shrink_);
  pcl::console::parse_argument (argc, argv, "-angle_incr", angle_incr_);
  pcl::console::parse_argument (argc, argv, "-objectness_threshold_", objectness_threshold_);
  pcl::console::parse_argument (argc, argv, "-test_directory", test_directory);
  pcl::console::parse_argument (argc, argv, "-labels_gt_directory", labels_gt_directory);
  pcl::console::parse_argument (argc, argv, "-z_dist", z_dist_);
  pcl::console::parse_argument (argc, argv, "-do_z", do_z);
  pcl::console::parse_argument (argc, argv, "-detect_clutter", detect_clutter_);
  pcl::console::parse_argument (argc, argv, "-optimize", optimize_);
  pcl::console::parse_argument (argc, argv, "-expand_factor", expand_factor);
  pcl::console::parse_argument (argc, argv, "-visualize", visualize);
  pcl::console::parse_argument (argc, argv, "-eval_file", eval_file);
  pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
  pcl::console::parse_argument (argc, argv, "-do_cuda", do_cuda);
  pcl::console::parse_argument (argc, argv, "-objectness", objectness_);
  pcl::console::parse_argument (argc, argv, "-best_nwins", best_nwins);
  pcl::console::parse_argument (argc, argv, "-cut_x", cut_x);
  pcl::console::parse_argument (argc, argv, "-opt_type", opt_type);

  std::vector<std::string> files;
  std::vector<std::string> files_gt;

  bool PCD_FILE_DEFINED_ = false;

  if (pcd_file.compare ("") == 0)
  {

    {
      bf::path obj_name_path = test_directory;
      std::vector<std::string> files_intern;
      std::string start = "";
      std::string ext = std::string ("pcd");
      getFilesInDirectory (obj_name_path, start, files_intern, ext);

      for (size_t i = 0; i < files_intern.size (); i++)
      {
        std::cout << files_intern[i] << std::endl;
        std::stringstream pc_name;
        pc_name << test_directory << "/" << files_intern[i];

        std::string nn = pc_name.str ();
        files.push_back (nn);
      }
    }

    {
      bf::path obj_name_path = labels_gt_directory;
      std::vector<std::string> files_intern;
      std::string start = "";
      std::string ext = std::string ("pcd");
      getFilesInDirectory (obj_name_path, start, files_intern, ext);

      for (size_t i = 0; i < files_intern.size (); i++)
      {
        std::cout << files_intern[i] << std::endl;
        std::stringstream pc_name;
        pc_name << labels_gt_directory << "/" << files_intern[i];

        std::string nn = pc_name.str ();
        files_gt.push_back (nn);
      }
    }

    std::sort (files.begin (), files.end ());
    std::sort (files_gt.begin (), files_gt.end ());

  }
  else
  {
    PCD_FILE_DEFINED_ = true;
    files.push_back(pcd_file);
    std::vector<std::string> strs;
    boost::split (strs, pcd_file, boost::is_any_of ("/"));
    std::string file;
    file.append(strs[strs.size() - 2]);
    file.append("/");

    std::vector<std::string> strs2;
    boost::split (strs2, strs[strs.size() - 1], boost::is_any_of ("."));
    std::string gt_file;
    gt_file.append(file);
    gt_file.append(strs2[0]);
    gt_file.append("_labels.pcd");
    std::cout << gt_file << std::endl;

    std::stringstream gt_file_complete;
    gt_file_complete << labels_gt_directory << "/" << gt_file;

    std::string nn = gt_file_complete.str ();
    std::cout << nn << std::endl;
    files_gt.push_back(nn);
  }

  ofstream out (eval_file.c_str ());

#if VISUALIZE_TOTAL_ > 0
  pcl::visualization::PCLVisualizer vis_ ("segmentation");
#endif
  int tp_total = 0;
  int fp_total = 0;
  int fn_total = 0;

  std::vector<int> tp_per_object;
  std::vector<int> fp_per_object;
  std::vector<int> fn_per_object;

  std::vector<float> precision_per_scene;
  std::vector<float> recall_per_scene;

  for (size_t i = 0; i < files.size (); i++)
  {
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Processing file:" << files[i] << std::endl;
    std::cout << "************************************************************************" << std::endl;

#if VISUALIZE_TOTAL_ > 0
    vis_.removeAllPointClouds ();
    vis_.removeAllShapes ();

    int v1, v2, v3;
    vis_.createViewPort (0.0, 0.0, 0.33, 1.0, v1);
    vis_.createViewPort (0.33, 0.0, 0.66, 1.0, v2);
    vis_.createViewPort (0.66, 0.0, 1.0, 1.0, v3);
#endif

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz_points (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile (files[i], *xyz_points);
    Eigen::Vector4f table_plane;
    computeTablePlane (xyz_points, table_plane, z_dist_);

    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

    {
      pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimation;
      normal_estimation.setNormalEstimationMethod (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::COVARIANCE_MATRIX);
      normal_estimation.setInputCloud (xyz_points);
      normal_estimation.setNormalSmoothingSize (15.0);
      normal_estimation.setBorderPolicy (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::BORDER_POLICY_MIRROR);
      /*pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimation;
      normal_estimation.setInputCloud (xyz_points);
      normal_estimation.setRadiusSearch(0.015f);*/
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
    if(cut_x) {
      pass_.setInputCloud (xyz_points_andy);
      pass_.setFilterLimits (-0.35f, 0.35f);
      pass_.setFilterFieldName ("x");
      pass_.filter (*xyz_points_andy);
    } else {
      pass_.setInputCloud (xyz_points_andy);
      pass_.setFilterLimits (-0.35f, 0.35f);
      pass_.setFilterFieldName ("y");
      pass_.filter (*xyz_points_andy);
    }

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters;
    std::vector<pcl::PointIndices> indices;

    if(objectness_)
    {
      faat_pcl::segmentation::Objectness3D<pcl::PointXYZRGB> o3d (shrink_, num_sampled_wins_, num_wins_, 3, 40, angle_incr_, objectness_threshold_);
      o3d.setBestWins(best_nwins);
      o3d.setInputCloud (xyz_points_andy);
      o3d.setTablePlane (table_plane);
      o3d.setEdges (edge_indices);
      o3d.setEdgeLabelsCloud (label_cloud);
      o3d.setDoOptimize (optimize_);
      o3d.setExpandFactor (expand_factor);
      o3d.setDoCuda(do_cuda);
      o3d.setOptType(opt_type);
      //smooth segmentation of the cloud
      if (detect_clutter_)
      {
        float size_voxels = 0.005f;
        pcl::PointCloud<pcl::PointXYZL>::Ptr clusters_cloud_;
        computeSuperPixels<pcl::PointXYZRGB> (xyz_points_andy, clusters_cloud_, size_voxels);
        o3d.setSmoothLabelsCloud (clusters_cloud_);
      }

      o3d.setVisualize (visualize);
      o3d.doZ (do_z);
      {
        pcl::ScopeTime t("total time...");
        o3d.computeObjectness (true);
      }
      //o3d.visualizeBoundingBoxes(vis_);

      o3d.getObjectIndices (indices, xyz_points);
    }
    else
    {
      std::cout << "Start segmentation..." << std::endl;
      pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
      ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
      ne.setMaxDepthChangeFactor (0.02f);
      ne.setNormalSmoothingSize (20.0f);
      ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::BORDER_POLICY_IGNORE);
      ne.setInputCloud (xyz_points);
      pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
      ne.compute (*normal_cloud);

      int num_plane_inliers = 5000;

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz_points_andy (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PassThrough<pcl::PointXYZRGB> pass_;
      pass_.setFilterLimits (0.f, z_dist_);
      pass_.setFilterFieldName ("z");
      pass_.setInputCloud (xyz_points);
      pass_.setKeepOrganized (true);
      pass_.filter (*xyz_points_andy);

      pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZRGB, pcl::Normal, pcl::Label> mps;
      mps.setMinInliers (num_plane_inliers);
      mps.setAngularThreshold (0.017453 * 1.5f); // 2 degrees
      mps.setDistanceThreshold (0.01); // 1cm
      mps.setInputNormals (normal_cloud);
      mps.setInputCloud (xyz_points_andy);

      std::vector<pcl::PlanarRegion<pcl::PointXYZRGB>, Eigen::aligned_allocator<pcl::PlanarRegion<pcl::PointXYZRGB> > > regions;
      std::vector<pcl::ModelCoefficients> model_coefficients;
      std::vector<pcl::PointIndices> inlier_indices;
      pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
      std::vector<pcl::PointIndices> label_indices;
      std::vector<pcl::PointIndices> boundary_indices;

      pcl::PlaneRefinementComparator<pcl::PointXYZRGB, pcl::Normal, pcl::Label>::Ptr ref_comp (
                                                                                               new pcl::PlaneRefinementComparator<pcl::PointXYZRGB,
                                                                                                   pcl::Normal, pcl::Label> ());
      ref_comp->setDistanceThreshold (0.01f, true);
      ref_comp->setAngularThreshold (0.017453 * 10);
      mps.setRefinementComparator (ref_comp);
      mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);

      std::cout << "Number of planes found:" << model_coefficients.size () << std::endl;

      int table_plane_selected = 0;
      int max_inliers_found = -1;
      std::vector<int> plane_inliers_counts;
      plane_inliers_counts.resize (model_coefficients.size ());

      for (size_t i = 0; i < model_coefficients.size (); i++)
      {
        Eigen::Vector4f table_plane = Eigen::Vector4f (model_coefficients[i].values[0], model_coefficients[i].values[1],
                                                       model_coefficients[i].values[2], model_coefficients[i].values[3]);

        std::cout << "Number of inliers for this plane:" << inlier_indices[i].indices.size () << std::endl;
        int remaining_points = 0;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_points (new pcl::PointCloud<pcl::PointXYZRGB> (*xyz_points_andy));
        for (int j = 0; j < plane_points->points.size (); j++)
        {
          Eigen::Vector3f xyz_p = plane_points->points[j].getVector3fMap ();

          if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
            continue;

          float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

          if (std::abs (val) > 0.01)
          {
            plane_points->points[j].x = std::numeric_limits<float>::quiet_NaN ();
            plane_points->points[j].y = std::numeric_limits<float>::quiet_NaN ();
            plane_points->points[j].z = std::numeric_limits<float>::quiet_NaN ();
          }
          else
            remaining_points++;
        }

        plane_inliers_counts[i] = remaining_points;

        if (remaining_points > max_inliers_found)
        {
          table_plane_selected = i;
          max_inliers_found = remaining_points;
        }
      }

      size_t itt = static_cast<size_t> (table_plane_selected);
      Eigen::Vector4f table_plane = Eigen::Vector4f (model_coefficients[itt].values[0], model_coefficients[itt].values[1],
                                                     model_coefficients[itt].values[2], model_coefficients[itt].values[3]);

      Eigen::Vector3f normal_table = Eigen::Vector3f (model_coefficients[itt].values[0], model_coefficients[itt].values[1],
                                                      model_coefficients[itt].values[2]);

      int inliers_count_best = plane_inliers_counts[itt];

      //check that the other planes with similar normal are not higher than the table_plane_selected
      for (size_t i = 0; i < model_coefficients.size (); i++)
      {
        Eigen::Vector4f model = Eigen::Vector4f (model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2],
                                                 model_coefficients[i].values[3]);

        Eigen::Vector3f normal = Eigen::Vector3f (model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2]);

        int inliers_count = plane_inliers_counts[i];

        std::cout << "Dot product is:" << normal.dot (normal_table) << std::endl;
        if ((normal.dot (normal_table) > 0.95) && (inliers_count_best * 0.5 <= inliers_count))
        {
          //check if this plane is higher, projecting a point on the normal direction
          std::cout << "Check if plane is higher, then change table plane" << std::endl;
          std::cout << model[3] << " " << table_plane[3] << std::endl;
          if (model[3] < table_plane[3])
          {
            PCL_WARN ("Changing table plane...");
            table_plane_selected = i;
            table_plane = model;
            normal_table = normal;
            inliers_count_best = inliers_count;
          }
        }
      }

      table_plane = Eigen::Vector4f (model_coefficients[table_plane_selected].values[0], model_coefficients[table_plane_selected].values[1],
                                     model_coefficients[table_plane_selected].values[2], model_coefficients[table_plane_selected].values[3]);

      //cluster..
      typename pcl::EuclideanClusterComparator<pcl::PointXYZRGB, pcl::Normal, pcl::Label>::Ptr
                                                                                               euclidean_cluster_comparator_ (
                                                                                                                              new pcl::EuclideanClusterComparator<
                                                                                                                                  pcl::PointXYZRGB,
                                                                                                                                  pcl::Normal,
                                                                                                                                  pcl::Label> ());

      //create two labels, 1 one for points belonging to or under the plane, 1 for points above the plane
      label_indices.resize (2);

      for (int j = 0; j < xyz_points_andy->points.size (); j++)
      {
        Eigen::Vector3f xyz_p = xyz_points_andy->points[j].getVector3fMap ();

        if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
          continue;

        float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

        if (val >= 0.01)
        {
          labels->points[j].label = 1;
          label_indices[0].indices.push_back (j);
        }
        else
        {
          labels->points[j].label = 0;
          label_indices[1].indices.push_back (j);
        }
      }

      std::vector<bool> plane_labels;
      plane_labels.resize (label_indices.size (), false);
      plane_labels[0] = true;

      euclidean_cluster_comparator_->setInputCloud (xyz_points_andy);
      euclidean_cluster_comparator_->setLabels (labels);
      euclidean_cluster_comparator_->setExcludeLabels (plane_labels);
      euclidean_cluster_comparator_->setDistanceThreshold (0.005f, false);

      pcl::PointCloud<pcl::Label> euclidean_labels;
      std::vector<pcl::PointIndices> euclidean_label_indices;
      pcl::OrganizedConnectedComponentSegmentation<pcl::PointXYZRGB, pcl::Label> euclidean_segmentation (euclidean_cluster_comparator_);
      euclidean_segmentation.setInputCloud (xyz_points_andy);
      euclidean_segmentation.segment (euclidean_labels, euclidean_label_indices);

      for (size_t i = 0; i < euclidean_label_indices.size (); i++)
      {
        if (euclidean_label_indices[i].indices.size () > 50)
        {
          indices.push_back (euclidean_label_indices[i]);
        }
      }

      PCL_INFO ("Got %d euclidean clusters!\n", clusters.size ());
      std::cout << "Regions:" << regions.size () << std::endl;
    }

    for (size_t j = 0; j < indices.size (); j++)
    {
      typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::copyPointCloud (*xyz_points, indices[j].indices, *cluster);
      clusters.push_back (cluster);
    }

    //visualize segmentation
#if VISUALIZE_TOTAL_ > 0
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (xyz_points);
    vis_.addPointCloud<pcl::PointXYZRGB> (xyz_points, handler_rgb, "scene_cloud_segmentation", v1);
    vis_.spinOnce (100.f, true);
#endif

    for (size_t j = 0; j < clusters.size (); j++)
    {
      if (indices[j].indices.size () < 500)
        continue;

      std::stringstream name;
      name << "cluster_" << j;

      typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::copyPointCloud (*xyz_points, indices[j].indices, *cluster);

/*#if VISUALIZE_TOTAL_ > 0
      pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZRGB> handler_rgb (cluster);
      vis_.addPointCloud<pcl::PointXYZRGB> (cluster, handler_rgb, name.str (), v3);
#endif*/
    }

    //create labels cloud to compare with gt
    pcl::PointCloud<pcl::PointXYZL>::Ptr result_labels (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZL> (*xyz_points, *result_labels);
    for (size_t k = 0; k < result_labels->points.size (); k++)
    {
      result_labels->points[k].label = 0;
    }

    int l = 1;
    for (size_t j = 0; j < clusters.size (); j++)
    {
      if (indices[j].indices.size () < 500)
        continue;

      for (size_t k = 0; k < indices[j].indices.size (); k++)
      {
        result_labels->points[indices[j].indices[k]].label = l;
      }

      l++;
    }

    if(!PCD_FILE_DEFINED_) {
      pcl::PointCloud<pcl::PointXYZL>::Ptr gt_labels (new pcl::PointCloud<pcl::PointXYZL>);
      pcl::io::loadPCDFile (files_gt[i], *gt_labels);
      std::vector<pcl::PointIndices> gt_labels_indices;
      std::map<int, int>::iterator it;

      {
        std::map<int, int> label_found;
        int nlabels_found = 0;
        for (size_t k = 0; k < gt_labels->points.size (); k++)
        {
          if (gt_labels->points[k].label != 0)
          {
            if ((it = label_found.find (gt_labels->points[k].label)) == label_found.end ())
            {
              label_found[gt_labels->points[k].label] = nlabels_found;
              pcl::PointIndices ind;
              ind.indices.push_back (k);
              gt_labels_indices.push_back (ind);
              nlabels_found++;
            }
            else
            {
              gt_labels_indices[it->second].indices.push_back (k);
            }
          }
        }
      }

      std::map<int, pcl::PointIndices> gt_resultlabels_indices;
      std::map<int, pcl::PointIndices>::iterator it2;

      {
        for (size_t k = 0; k < result_labels->points.size (); k++)
        {
          if (gt_labels->points[k].label != 0)
          {
            if ((it2 = gt_resultlabels_indices.find (result_labels->points[k].label)) == gt_resultlabels_indices.end ())
            {
              pcl::PointIndices ind;
              ind.indices.push_back (k);
              gt_resultlabels_indices[result_labels->points[k].label] = ind;
            }
            else
            {
              it2->second.indices.push_back (k);
            }
          }
        }
      }

      std::cout << "Num labels found:" << gt_labels_indices.size () << std::endl;

      if(best_nwins == -1) {
        int tp_scene, fp_scene, fn_scene;
        tp_scene = fp_scene = fn_scene = 0;

        for (size_t j = 0; j < gt_labels_indices.size (); j++)
        {
          //count max label in the results_label cloud
          int max_id = 0;
          std::map<int, int> nlabels_in_results;
          for (size_t k = 0; k < gt_labels_indices[j].indices.size (); k++)
          {
            if ((it = nlabels_in_results.find (result_labels->points[gt_labels_indices[j].indices[k]].label)) == nlabels_in_results.end ())
            {
              nlabels_in_results[result_labels->points[gt_labels_indices[j].indices[k]].label] = 1;
            }
            else
            {
              it->second++;
            }
          }

          int max_p = 0;
          for (it = nlabels_in_results.begin (); it != nlabels_in_results.end (); it++)
          {
            if (it->second > max_p)
            {
              max_id = it->first;
              max_p = it->second;
            }
          }

          //max are the tp
          //fn are gt_labels_indices[j].indices.size () - max
          //fp are the number of labels in result minus max
          int tp = max_p;
          int fn = static_cast<int> (gt_labels_indices[j].indices.size ()) - tp;
          it2 = gt_resultlabels_indices.find (max_id);
          int fp = it2->second.indices.size () - tp;

          tp_total += tp;
          fp_total += fp;
          fn_total += fn;

          tp_per_object.push_back (tp);
          fn_per_object.push_back (fn);
          fp_per_object.push_back (fp);

          tp_scene += tp;
          fn_scene += fn;
          fp_scene += fp;
          //std::cout << max_id << " <--->" << nlabels_in_results[max_id] << std::endl;
          //std::cout << "tp:" << tp << " fn:" << fn << " fp:" << fp << std::endl;
        }

        precision_per_scene.push_back(tp_scene / static_cast<float>(tp_scene + fp_scene));
        recall_per_scene.push_back(tp_scene / static_cast<float>(tp_scene + fn_scene));
        //visualize gt
  #if VISUALIZE_TOTAL_ > 0
        {
          pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> handler_rgb (gt_labels, "label");
          vis_.addPointCloud<pcl::PointXYZL> (gt_labels, handler_rgb, "gt_labels", v2);
        }
  #endif

        std::cout << "tp:" << tp_total << " fn:" << fn_total << " fp:" << fp_total << std::endl;
        std::cout << "precision: " << tp_total / static_cast<float> (tp_total + fp_total) << std::endl;
        std::cout << "recall: " << tp_total / static_cast<float> (tp_total + fn_total) << std::endl;

        out << files_gt[i] << "\t" <<  precision_per_scene[i] << "\t" << recall_per_scene[i] << std::endl;

      } else {
         //get the gt label that intersects with the result label
         std::map<int, int> nlabels_in_results;
         std::map<int, int>::iterator it;

         for (size_t k = 0; k < indices[0].indices.size (); k++) {
           it = nlabels_in_results.find(gt_labels->points[indices[0].indices[k]].label);
           if(it == nlabels_in_results.end()) {
             nlabels_in_results[gt_labels->points[indices[0].indices[k]].label] = 1;
           } else {
             it->second++;
           }
         }

         int max_id = 0;
         int max_p = 0;
         int sum = 0;
         for (it = nlabels_in_results.begin (); it != nlabels_in_results.end (); it++)
         {
           if (it->second > max_p)
           {
             max_id = it->first;
             max_p = it->second;
           }

           sum += it->second;
         }

         std::map<int, int> gt_labels_size;
         for (size_t k = 0; k < gt_labels->points.size (); k++)
         {
           if ((it = gt_labels_size.find (gt_labels->points[k].label)) == gt_labels_size.end ())
           {
             gt_labels_size[gt_labels->points[k].label] = 1;
           }
           else
           {
             it->second++;
           }
         }

         it = gt_labels_size.find(max_id);
         int gt_label_size = it->second;

         //std::cout << max_id << " " << max_p << " " << sum << " " << indices[0].indices.size () << " " << gt_label_size << std::endl;
         out << files_gt[i] << "\t" <<  max_p / static_cast<float>(sum) << "\t" << max_p / static_cast<float>(gt_label_size) << std::endl;
      }

  #if VISUALIZE_TOTAL_ > 0
      //visualize result labels
      {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> handler_rgb (result_labels, "label");
        vis_.addPointCloud<pcl::PointXYZL> (result_labels, handler_rgb, "result_labels", v3);
      }

      if (visualize)
      {
        vis_.spin ();
      }
      else
      {
        vis_.spinOnce ();
      }
  #endif
    }
    else
    {
      pcl::visualization::PCLVisualizer vis_("segmentation");
      vis_.removeAllPointClouds ();
      vis_.removeAllShapes ();

      int v1, v2;
      vis_.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
      vis_.createViewPort (0.5, 0.0, 1.0, 1.0, v2);

      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (xyz_points);
      vis_.addPointCloud<pcl::PointXYZRGB> (xyz_points, handler_rgb, "scene_cloud_segmentation", v1);

      pcl::PointCloud<pcl::PointXYZL>::Ptr result_labels (new pcl::PointCloud<pcl::PointXYZL>);
      pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZL> (*xyz_points, *result_labels);
      for (size_t k = 0; k < result_labels->points.size (); k++)
      {
        result_labels->points[k].label = 0;
      }

      int l = 1;
      for (size_t j = 0; j < clusters.size (); j++)
      {
        if (indices[j].indices.size () < 50)
          continue;

        for (size_t k = 0; k < indices[j].indices.size (); k++)
        {
          result_labels->points[indices[j].indices[k]].label = l;
        }

        l++;
      }

      {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> handler_rgb (result_labels, "label");
        vis_.addPointCloud<pcl::PointXYZL> (result_labels, handler_rgb, "result_labels", v2);
      }
      vis_.spin();

    }
  }

  out << "tp:" << tp_total << " fn:" << fn_total << " fp:" << fp_total << std::endl;
  out << "precision: " << tp_total / static_cast<float> (tp_total + fp_total) << std::endl;
  out << "recall: " << tp_total / static_cast<float> (tp_total + fn_total) << std::endl;

  /*for(size_t i=0; i < precision_per_scene.size(); i++) {
    out << files_gt[i] << "\t" <<  precision_per_scene[i] << "\t" << recall_per_scene[i] << std::endl;
  }*/
}
