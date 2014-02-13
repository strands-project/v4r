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
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <faat_pcl/utils/filesystem_utils.h>

#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/integral_image_normal.h>
#include "seg_do_modelling.h"
#include <faat_pcl/registration/mv_lm_icp.h>
#include <pcl/features/normal_3d_omp.h>

struct IndexPoint
{
  int idx;
};

POINT_CLOUD_REGISTER_POINT_STRUCT (IndexPoint,
    (int, idx, idx)
)

inline bool
readMatrixFromFile2 (std::string file, Eigen::Matrix4f & matrix, int ignore = 0)
{

  std::ifstream in;
  in.open (file.c_str (), std::ifstream::in);

  char linebuf[1024];
  in.getline (linebuf, 1024);
  std::string line (linebuf);
  std::vector<std::string> strs_2;
  boost::split (strs_2, line, boost::is_any_of (" "));

  int c = 0;
  for (int i = ignore; i < (ignore + 16); i++, c++)
  {
    matrix (c / 4, c % 4) = static_cast<float> (atof (strs_2[i].c_str ()));
  }

  return true;
}

inline bool
writeMatrixToFile (std::string file, Eigen::Matrix4f & matrix)
{
  std::ofstream out (file.c_str ());
  if (!out)
  {
    std::cout << "Cannot open file.\n";
    return false;
  }

  for (size_t i = 0; i < 4; i++)
  {
    for (size_t j = 0; j < 4; j++)
    {
      out << matrix (i, j);
      if (!(i == 3 && j == 3))
        out << " ";
    }
  }
  out.close ();

  return true;
}

void transformNormals(pcl::PointCloud<pcl::Normal>::Ptr & normals_cloud,
                         pcl::PointCloud<pcl::Normal>::Ptr & normals_aligned,
                         Eigen::Matrix4f & transform)
{
  normals_aligned.reset (new pcl::PointCloud<pcl::Normal>);
  normals_aligned->points.resize (normals_cloud->points.size ());
  normals_aligned->width = normals_cloud->width;
  normals_aligned->height = normals_cloud->height;
  for (size_t k = 0; k < normals_cloud->points.size (); k++)
  {
    Eigen::Vector3f nt (normals_cloud->points[k].normal_x, normals_cloud->points[k].normal_y, normals_cloud->points[k].normal_z);
    normals_aligned->points[k].normal_x = static_cast<float> (transform (0, 0) * nt[0] + transform (0, 1) * nt[1]
        + transform (0, 2) * nt[2]);
    normals_aligned->points[k].normal_y = static_cast<float> (transform (1, 0) * nt[0] + transform (1, 1) * nt[1]
        + transform (1, 2) * nt[2]);
    normals_aligned->points[k].normal_z = static_cast<float> (transform (2, 0) * nt[0] + transform (2, 1) * nt[1]
        + transform (2, 2) * nt[2]);
  }
}

//./bin/object_modelling_willow -pcd_files_dir /home/aitor/willow_challenge_ros_code/read_willow_data/train/object_15/ -Z_DIST 0.8 -num_plane_inliers 2000 -max_corresp_dist 0.01 -vx_size 0.003 -dt_size 0.003 -visualize 0 -fast_overlap 1 -aligned_output_dir /home/aitor/data/willow_structure/object_15.pcd -aligned_model_saved_to /home/aitor/data/willow_object_clouds/models_ml_new/object_15.pcd

int
main (int argc, char ** argv)
{
  float Z_DIST_ = 1.5f;
  std::string pcd_files_dir_;
  bool sort_pcd_files_ = true;
  bool use_max_cluster_ = true;
  float data_scale = 1.f;
  float x_limits = 0.4f;
  int num_plane_inliers = 500;
  bool single_object = true;
  bool vis_final_ = true;
  std::string aligned_output_dir = "";
  float max_corresp_dist_ = 0.1f;
  float voxel_grid_size = 0.005f;
  float dt_size = voxel_grid_size;
  float visualize = false;
  bool fast_overlap = true;
  std::string aligned_model_saved_to = "test.pcd";
  int ignore = 1;

  pcl::console::parse_argument (argc, argv, "-ignore", ignore);
  pcl::console::parse_argument (argc, argv, "-pcd_files_dir", pcd_files_dir_);
  pcl::console::parse_argument (argc, argv, "-sort_pcd_files", sort_pcd_files_);
  pcl::console::parse_argument (argc, argv, "-use_max_cluster", use_max_cluster_);
  pcl::console::parse_argument (argc, argv, "-data_scale", data_scale);
  pcl::console::parse_argument (argc, argv, "-x_limits", x_limits);
  pcl::console::parse_argument (argc, argv, "-Z_DIST", Z_DIST_);
  pcl::console::parse_argument (argc, argv, "-num_plane_inliers", num_plane_inliers);
  pcl::console::parse_argument (argc, argv, "-single_object", single_object);
  pcl::console::parse_argument (argc, argv, "-vis_final", vis_final_);
  pcl::console::parse_argument (argc, argv, "-aligned_output_dir", aligned_output_dir);

  pcl::console::parse_argument (argc, argv, "-max_corresp_dist", max_corresp_dist_);
  pcl::console::parse_argument (argc, argv, "-vx_size", voxel_grid_size);
  pcl::console::parse_argument (argc, argv, "-dt_size", dt_size);
  pcl::console::parse_argument (argc, argv, "-visualize", visualize);
  pcl::console::parse_argument (argc, argv, "-fast_overlap", fast_overlap);
  pcl::console::parse_argument (argc, argv, "-aligned_model_saved_to", aligned_model_saved_to);

  std::vector<std::string> files;
  std::string start = "";
  std::string ext = std::string ("pcd");
  bf::path dir = pcd_files_dir_;
  faat_pcl::utils::getFilesInDirectory (dir, start, files, ext);
  std::cout << "Number of scenes in directory is:" << files.size () << std::endl;

  typedef pcl::PointXYZRGB PointType;
  typedef pcl::PointXYZRGBNormal PointTypeNormal;

  std::vector<pcl::PointCloud<PointType>::Ptr> clouds_;
  clouds_.resize (files.size ());

  if (sort_pcd_files_)
    std::sort (files.begin (), files.end ());

  pcl::visualization::PCLVisualizer vis ("");

  std::vector<pcl::PointCloud<PointType>::Ptr> range_images_;
  std::vector<pcl::PointCloud<PointType>::Ptr> edges_;
  std::vector<std::vector<int> > obj_indices_;
  std::vector<std::string> pose_files;
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr > clouds_normals_;

  for (size_t i = 0; i < files.size (); i++)
  {
    pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr rimage (new pcl::PointCloud<PointType>);
    std::stringstream file_to_read;
    file_to_read << pcd_files_dir_ << "/" << files[i];
    pcl::io::loadPCDFile (file_to_read.str (), *scene);
    pcl::copyPointCloud(*scene, *rimage);

    std::cout << file_to_read.str () << std::endl;
    std::string pose_file = file_to_read.str ();
    boost::replace_all (pose_file, ".pcd", ".txt");
    boost::replace_all (pose_file, "cloud", "pose");
    bf::path pose_file_path = pose_file;
    if (bf::exists (pose_file_path))
    {
      pose_files.push_back (pose_file);
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

    edges_.push_back (edges);
    range_images_.push_back (rimage);

    //segment the object of interest
    pcl::PassThrough<PointType> pass_;
    pass_.setFilterLimits (0.f, Z_DIST_);
    pass_.setFilterFieldName ("z");
    pass_.setInputCloud (scene);
    pass_.setKeepOrganized (true);
    pass_.filter (*scene);

    if (x_limits > 0)
    {
      pass_.setInputCloud (scene);
      pass_.setFilterLimits (-x_limits, x_limits);
      pass_.setFilterFieldName ("x");
      pass_.filter (*scene);
    }

    std::vector<pcl::PointIndices> indices;
    Eigen::Vector4f table_plane;
    doSegmentation<PointType> (scene, indices, table_plane, num_plane_inliers);

    std::cout << "Number of clusters found:" << indices.size () << std::endl;

    if (single_object)
    {
      std::cout << "selecting max..." << std::endl;
      pcl::PointIndices max;
      for (size_t k = 0; k < indices.size (); k++)
      {
        if (max.indices.size () < indices[k].indices.size ())
        {
          max = indices[k];
        }
      }

      pcl::PointCloud<PointType>::Ptr obj_interest (new pcl::PointCloud<PointType>);
      obj_interest->width = scene->width;
      obj_interest->height = scene->height;
      obj_interest->points.resize (scene->points.size ());
      obj_interest->is_dense = false;
      for (size_t k = 0; k < obj_interest->points.size (); k++)
      {
        obj_interest->points[k].x = std::numeric_limits<float>::quiet_NaN ();
        obj_interest->points[k].y = std::numeric_limits<float>::quiet_NaN ();
        obj_interest->points[k].z = std::numeric_limits<float>::quiet_NaN ();
        obj_interest->points[k].rgb = std::numeric_limits<float>::quiet_NaN ();
      }

      for (size_t k = 0; k < max.indices.size (); k++)
      {
        obj_interest->points[max.indices[k]] = scene->points[max.indices[k]];
      }

      //pcl::copyPointCloud(*scene, max, *obj_interest);
      obj_indices_.push_back (indices[0].indices);
      clouds_[i] = obj_interest;

      {
        pcl::NormalEstimationOMP<PointType, pcl::Normal> normal_estimation;
        normal_estimation.setInputCloud (obj_interest);
        normal_estimation.setRadiusSearch(0.02f);
        normal_estimation.compute (*normals);
        clouds_normals_.push_back(normals);
      }
    }
    else
    {
      std::vector<int> obj_indices;
      pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
      for (size_t k = 0; k < indices.size (); k++)
      {
        pcl::PointCloud<PointType>::Ptr obj_interest (new pcl::PointCloud<PointType>);
        pcl::copyPointCloud (*scene, indices[k], *obj_interest);
        *cloud += *obj_interest;
        obj_indices.insert (obj_indices.end (), indices[k].indices.begin (), indices[k].indices.end ());
      }
      obj_indices_.push_back (obj_indices);
      clouds_[i] = cloud;

    }
  }

  std::vector<Eigen::Matrix4f> poses_;
  std::vector<pcl::PointCloud<PointType>::Ptr> clouds_aligned_;
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr > clouds_normals_aligned_;

  {
    pcl::visualization::PCLVisualizer aligned_vis ("aligned");
    for (size_t i = 0; i < pose_files.size (); i++)
    {
      std::cout << pose_files[i] << std::endl;
      Eigen::Matrix4f pose;
      readMatrixFromFile2 (pose_files[i], pose, ignore);
      std::cout << pose << std::endl;
      std::cout << pose.inverse () << std::endl;
      if(ignore == 1) //TODO: Attention here, this might be tricky... (willow, my modelling)
        pose = pose.inverse ();
      pcl::PointCloud<PointType>::Ptr cloud_trans (new pcl::PointCloud<PointType>);
      pcl::transformPointCloud (*clouds_[i], *cloud_trans, pose);

      std::stringstream cloud_name;
      cloud_name << "cloud_" << i << ".pcd";

      pcl::PointCloud<PointType>::Ptr cloud_sor (new pcl::PointCloud<PointType>);

      std::vector<int> pointIdxNKNSearch;
      std::vector<float> pointNKNSquaredDistance;
      pcl::octree::OctreePointCloudSearch<PointType> octree (0.003);
      octree.setInputCloud (edges_[i]);
      octree.addPointsFromInputCloud ();

      std::vector<int> indices_set_to_nan;
      for (size_t k = 0; k < clouds_[i]->points.size (); k++)
      {
        if (!pcl_isfinite(clouds_[i]->points[k].z))
          continue;

        if (octree.nearestKSearch (clouds_[i]->points[k], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
          float d = sqrt (pointNKNSquaredDistance[0]);
          if (d < 0.0015f)
            indices_set_to_nan.push_back (static_cast<int> (k));
        }
      }

      pcl::copyPointCloud (*cloud_trans, *cloud_sor);
      for (size_t k = 0; k < indices_set_to_nan.size (); k++)
      {
        cloud_sor->points[indices_set_to_nan[k]].x = std::numeric_limits<float>::quiet_NaN ();
        cloud_sor->points[indices_set_to_nan[k]].y = std::numeric_limits<float>::quiet_NaN ();
        cloud_sor->points[indices_set_to_nan[k]].z = std::numeric_limits<float>::quiet_NaN ();
        cloud_sor->points[indices_set_to_nan[k]].rgb = std::numeric_limits<float>::quiet_NaN ();
        clouds_normals_[i]->points[indices_set_to_nan[k]].normal_x = std::numeric_limits<float>::quiet_NaN ();
        clouds_normals_[i]->points[indices_set_to_nan[k]].normal_y = std::numeric_limits<float>::quiet_NaN ();
        clouds_normals_[i]->points[indices_set_to_nan[k]].normal_z = std::numeric_limits<float>::quiet_NaN ();
        clouds_normals_[i]->points[indices_set_to_nan[k]].curvature = std::numeric_limits<float>::quiet_NaN ();
      }

      pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (cloud_sor);
      aligned_vis.addPointCloud<PointType> (cloud_sor, handler_rgb, cloud_name.str ());
      cloud_name << "_normals";

      pcl::PointCloud<pcl::Normal>::Ptr aligned_normals (new pcl::PointCloud<pcl::Normal>);
      transformNormals(clouds_normals_[i], aligned_normals, pose);
      aligned_vis.addPointCloudNormals<PointType, pcl::Normal> (cloud_sor, aligned_normals, 100, 0.01, cloud_name.str ());

      clouds_aligned_.push_back (cloud_sor);
      clouds_normals_aligned_.push_back(aligned_normals);
      poses_.push_back (pose);
    }

    if(visualize)
    aligned_vis.spin ();
  }

  //do multiview ICP
  std::vector<pcl::PointCloud<PointType>::Ptr> clouds; //clouds after LM ICP
  clouds.resize(files.size());
  for (size_t i = 0; i < files.size (); i++)
  {
    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
    cloud = clouds_aligned_[i];

    {
      //compute canny edges
      pcl::OrganizedEdgeFromRGB<pcl::PointXYZRGB, pcl::Label> oed;
      oed.setDepthDisconThreshold (0.03f);
      oed.setRGBCannyLowThreshold (150.f);
      oed.setRGBCannyHighThreshold (200.f);
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

          if (pcl_isfinite(pl.z) && pcl_isfinite(pl.x) && pcl_isfinite(pl.y))
            edges->push_back (pl);
        }
      }

      //filter edge regions that are not dense enough
      pcl::PointCloud<PointType>::Ptr filtered_edges (new pcl::PointCloud<PointType> ());

      pcl::StatisticalOutlierRemoval<PointType> ror;
      ror.setMeanK (10);
      ror.setStddevMulThresh (1.f);
      ror.setInputCloud (edges);
      ror.filter (*filtered_edges);

      pcl::PointCloud<PointType>::Ptr points (new pcl::PointCloud<PointType> (*filtered_edges));
      float curv_thres = 0.01f;
      if (filtered_edges->points.size () > 0)
      {
        std::vector<int> pointIdxNKNSearch;
        std::vector<float> pointNKNSquaredDistance;
        float inlier = 0.02f;
        pcl::octree::OctreePointCloudSearch<PointType> octree (0.003);
        octree.setInputCloud (filtered_edges);
        octree.addPointsFromInputCloud ();

        for (size_t k = 0; k < cloud->points.size (); k++)
        {
          if (pcl_isfinite(cloud->points[k].z) && pcl_isfinite(cloud->points[k].x) && pcl_isfinite(cloud->points[k].y))
          {

            /*if(!pcl_isnan(normals_oed->points[k].curvature) && normals_oed->points[k].curvature < curv_thres)
             continue;*/

            if (octree.nearestKSearch (cloud->points[k], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
              float d = sqrt (pointNKNSquaredDistance[0]);
              if (d > inlier)
              {
                points->push_back (cloud->points[k]);
              }
            }
          }
        }
      }
      else
      {
        PCL_WARN("There are no color edges...\n");
        for (size_t k = 0; k < cloud->points.size (); k++)
        {
          if (pcl_isfinite(cloud->points[k].z) && pcl_isfinite(cloud->points[k].x) && pcl_isfinite(cloud->points[k].y))
          {
            /*if(pcl_isnan(normals_oed->points[k].curvature) || normals_oed->points[k].curvature < curv_thres)
             continue;*/

            points->push_back (cloud->points[k]);
          }
        }
      }

      /*clouds[i].reset(new pcl::PointCloud<PointType>());
       pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
       normal_estimator->estimate(cloud, clouds[i], normals);
       normals_clouds.push_back(normals);*/

      clouds[i].reset (new pcl::PointCloud<PointType> (*points));
    }
  }

  std::vector<std::vector<bool> > A;
  A.resize (clouds.size ());
  for (size_t i = 0; i < clouds.size (); i++)
  {
    A[i].resize (clouds.size (), true);
  }

  float ff = 0.3f;
  if(fast_overlap)
  {
    PointType min_pt_all, max_pt_all;
    min_pt_all.x = min_pt_all.y = min_pt_all.z = std::numeric_limits<float>::max ();
    max_pt_all.x = max_pt_all.y = max_pt_all.z = (std::numeric_limits<float>::max () - 0.001f) * -1;

    for (size_t i = 0; i < clouds.size (); i++)
    {
      PointType min_pt, max_pt;
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
      for (size_t k = 0; k < clouds[i]->points.size (); k++)
      {
        int pos_x, pos_y, pos_z;
        pos_x = static_cast<int> (std::floor ((clouds[i]->points[k].x - min_pt_all.x) / res_occupancy_grid_));
        pos_y = static_cast<int> (std::floor ((clouds[i]->points[k].y - min_pt_all.y) / res_occupancy_grid_));
        pos_z = static_cast<int> (std::floor ((clouds[i]->points[k].z - min_pt_all.z) / res_occupancy_grid_));
        int idx = pos_z * size_x * size_y + pos_y * size_x + pos_x;
        complete_cloud_occupancy_by_RM_[idx] = 1;
      }

      int total_points_i = std::accumulate (complete_cloud_occupancy_by_RM_.begin (), complete_cloud_occupancy_by_RM_.end (), 0);

      std::vector<int> complete_cloud_occupancy_by_RM_j;
      complete_cloud_occupancy_by_RM_j.resize (size_x * size_y * size_z, 0);

      for (size_t j = i; j < clouds.size (); j++)
      {
        int overlap = 0;
        std::map<int, bool> banned;
        std::map<int, bool>::iterator banned_it;

        for (size_t k = 0; k < clouds[j]->points.size (); k++)
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
            if (complete_cloud_occupancy_by_RM_[idx] > 0)
              overlap++;
            banned[idx] = true;
          }
        }

        float total_points_j = std::accumulate (complete_cloud_occupancy_by_RM_j.begin (), complete_cloud_occupancy_by_RM_j.end (), 0);

        float ov_measure_1 = overlap / static_cast<float> (total_points_j);
        float ov_measure_2 = overlap / static_cast<float> (total_points_i);
        if (!(ov_measure_1 > ff || ov_measure_2 > ff))
        {
          A[i][j] = false;
          A[j][i] = false;
        }
      }
    }
  }
  else
  {
    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;
    float inlier = max_corresp_dist_;
    for (size_t i = 0; i < clouds.size (); i++)
    {
      A[i][i] = false;
      pcl::octree::OctreePointCloudSearch<PointType> octree (0.003);
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
        if(!(ov_measure_1 > ff || ov_measure_2 > ff))
        {
          A[i][j] = false;
          A[j][i] = false;
        }
      }
    }
  }

  for (size_t i = 0; i < clouds.size (); i++)
  {
    for (size_t j = 0; j < clouds.size (); j++)
      std::cout << A[i][j] << " ";

    std::cout << std::endl;
  }

  if(visualize)
  {
    pcl::PointCloud<PointType>::Ptr big_cloud;
    big_cloud.reset(new pcl::PointCloud<PointType>);

    pcl::visualization::PCLVisualizer vis("aligned");
    int v1, v2;
    vis.createViewPort(0,0,0.5,1,v1);
    vis.createViewPort(0.5,0,1,1,v2);


    int t=0;
    for (size_t i = 0; i < clouds.size (); i++)
    {
      *big_cloud += *clouds[i];
    }

    {
      pcl::visualization::PointCloudColorHandlerCustom<PointType> rand_h(big_cloud, 0, 255, 0);
      vis.addPointCloud<PointType>(big_cloud, rand_h, "initial", v1);
    }

    vis.spin();
  }

  faat_pcl::registration::MVNonLinearICP<PointType> icp_nl (dt_size);
  icp_nl.setInlierThreshold (voxel_grid_size);
  icp_nl.setMaxCorrespondenceDistance (max_corresp_dist_);
  icp_nl.setClouds (clouds);
  icp_nl.setVisIntermediate (false);
  icp_nl.setSparseSolver (true);
  icp_nl.setAdjacencyMatrix (A);
  icp_nl.compute ();

  std::vector<Eigen::Matrix4f> transformations;
  icp_nl.getTransformation (transformations);

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
      pcl::visualization::PointCloudColorHandlerCustom<PointType> rand_h(big_cloud_aligned, 255, 0, 0);
      vis.addPointCloud<PointType>(big_cloud_aligned, rand_h, "aligned", v2);
    }

    {
      pcl::visualization::PointCloudColorHandlerCustom<PointType> rand_h(big_cloud, 0, 255, 0);
      vis.addPointCloud<PointType>(big_cloud, rand_h, "initial", v1);
    }

    vis.spin();
  }

  {

    pcl::PointCloud<PointType>::Ptr big_cloud;
    pcl::PointCloud<PointType>::Ptr big_cloud_aligned;
    big_cloud.reset(new pcl::PointCloud<PointType>);
    big_cloud_aligned.reset(new pcl::PointCloud<PointType>);
    pcl::PointCloud<pcl::Normal>::Ptr big_cloud_aligned_normals;
    big_cloud_aligned_normals.reset(new pcl::PointCloud<pcl::Normal>);

    for (size_t i = 0; i < clouds.size (); i++)
    {
      poses_[i] = transformations[i] * poses_[i];
    }

    for (size_t i = 0; i < clouds.size (); i++)
    {
      *big_cloud += *clouds_aligned_[i];

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cloud;
      trans_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr initial_cloud;
      initial_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>(*clouds_aligned_[i]));

      //pcl::io::loadPCDFile(files_global[i], *initial_cloud);
      pcl::transformPointCloud(*initial_cloud, *trans_cloud, transformations[i]);

      /*std::stringstream cloud_name;
      cloud_name << "aligned_cloud_" << i << ".pcd";
      pcl::io::savePCDFileBinary(cloud_name.str().c_str(), *trans_cloud);*/

      *big_cloud_aligned += *trans_cloud;

      pcl::PointCloud<pcl::Normal>::Ptr aligned_normals (new pcl::PointCloud<pcl::Normal>);
      transformNormals(clouds_normals_aligned_[i], aligned_normals, transformations[i]);
      *big_cloud_aligned_normals += *aligned_normals;

    }

    pcl::PointCloud<PointType>::Ptr filtered (new pcl::PointCloud<PointType>());
    pcl::StatisticalOutlierRemoval<PointType> ror(true);
    ror.setMeanK(10);
    ror.setStddevMulThresh(1.f);
    ror.setInputCloud(big_cloud_aligned);
    ror.setNegative(true);
    ror.filter(*filtered);

    pcl::PointIndices::Ptr removed(new pcl::PointIndices);
    ror.getRemovedIndices(*removed);

    std::cout << removed->indices.size() << std::endl;

    /*pcl::ExtractIndices<pcl::Normal> extract;
    extract.setInputCloud(big_cloud_aligned_normals);
    extract.setNegative(true);
    extract.setIndices(removed);*/
    pcl::PointCloud<pcl::Normal>::Ptr big_cloud_aligned_normals_final;
    big_cloud_aligned_normals_final.reset(new pcl::PointCloud<pcl::Normal>);
    //extract.filter(*big_cloud_aligned_normals_final);

    pcl::copyPointCloud(*big_cloud_aligned_normals, *removed, *big_cloud_aligned_normals_final);
    pcl::copyPointCloud(*big_cloud_aligned, *removed, *filtered);

    if(visualize)
    {
      pcl::visualization::PCLVisualizer vis("aligned");
      int v1, v2, v3;
      vis.createViewPort(0,0,0.33,1,v1);
      vis.createViewPort(0.33,0,0.66,1,v2);
      vis.createViewPort(0.66,0,1,1,v3);

      {
        pcl::visualization::PointCloudColorHandlerRGBField<PointType> rand_h(big_cloud_aligned);
        vis.addPointCloud<PointType>(big_cloud_aligned, rand_h, "aligned", v1);
      }

      {
        pcl::visualization::PointCloudColorHandlerRGBField<PointType> rand_h(filtered);
        vis.addPointCloud<PointType>(filtered, rand_h, "filtered", v2);
      }

      {
        pcl::visualization::PointCloudColorHandlerRGBField<PointType> rand_h(big_cloud);
        vis.addPointCloud<PointType>(big_cloud, rand_h, "initial", v3);
      }

      vis.addCoordinateSystem(0.1f);
      vis.spin();
    }

    if(visualize)
    {
      std::cout << filtered->points.size() << " -- " << big_cloud_aligned_normals_final->points.size() << std::endl;
      std::cout << big_cloud_aligned->points.size() << " -- " << big_cloud_aligned_normals->points.size() << std::endl;
      //visualize final model with normals
      pcl::visualization::PCLVisualizer vis("FINAL");
      pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (filtered);
      vis.addPointCloud<PointType> (filtered, handler_rgb, "final");
      vis.addPointCloudNormals<PointType, pcl::Normal> (filtered, big_cloud_aligned_normals_final, 100, 0.01, "final_model");
      vis.addCoordinateSystem(0.1f);
      vis.spin();
    }

    pcl::PointCloud<PointTypeNormal>::Ptr final (new pcl::PointCloud<PointTypeNormal>());
    final->width = filtered->width;
    final->height = filtered->height;
    final->points.resize (filtered->points.size ());
    final->is_dense = false;
    for (size_t k = 0; k < final->points.size (); k++)
    {
      final->points[k].x = filtered->points[k].x;
      final->points[k].y = filtered->points[k].y;
      final->points[k].z = filtered->points[k].z;
      final->points[k].rgb = filtered->points[k].rgb;
      final->points[k].normal_x = big_cloud_aligned_normals_final->points[k].normal_x;
      final->points[k].normal_y = big_cloud_aligned_normals_final->points[k].normal_y;
      final->points[k].normal_z = big_cloud_aligned_normals_final->points[k].normal_z;
      final->points[k].curvature = big_cloud_aligned_normals_final->points[k].curvature;
    }

    pcl::PassThrough<PointTypeNormal> pass_;
    pass_.setFilterLimits (0.f, 10.f);
    pass_.setFilterFieldName ("z");
    pass_.setInputCloud (final);
    pcl::PointCloud<PointTypeNormal>::Ptr final2 (new pcl::PointCloud<PointTypeNormal>());
    pass_.filter (*final2);

    float voxel_grid_size = 0.001f;
    pcl::VoxelGrid<PointTypeNormal> grid_;
    grid_.setInputCloud (final2);
    grid_.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
    grid_.setDownsampleAllData (true);
    grid_.filter (*final);
    std::cout << final->points.size() << " -- " << final2->points.size() << std::endl;
    pcl::io::savePCDFileBinary(aligned_model_saved_to, *final);

    bf::path aod = aligned_output_dir;
    if(!bf::exists(aod))
    {
      bf::create_directory(aod);
    }

    //save the original clouds, the masks and refined pose files
    for(size_t k=0; k < range_images_.size(); k++)
    {
      //write original cloud
      {
        std::stringstream temp;
        temp << aligned_output_dir << "/cloud_";
        temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".pcd";
        std::string scene_name;
        temp >> scene_name;
        std::cout << scene_name << std::endl;
        pcl::io::savePCDFileBinary(scene_name, *range_images_[k]);
      }

      //write pose
      {
        std::stringstream temp;
        temp << aligned_output_dir << "/pose_";
        temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".txt";
        std::string scene_name;
        temp >> scene_name;
        std::cout << scene_name << std::endl;
        writeMatrixToFile(scene_name, poses_[k]);
        std::cout << poses_[k] << std::endl;
      }

      //write object indices
      {
        std::stringstream temp;
        temp << aligned_output_dir << "/object_indices_";
        temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".pcd";
        std::string scene_name;
        temp >> scene_name;
        std::cout << scene_name << std::endl;
        pcl::PointCloud<IndexPoint> obj_indices_cloud;
        obj_indices_cloud.width = obj_indices_[k].size();
        obj_indices_cloud.height = 1;
        obj_indices_cloud.points.resize(obj_indices_cloud.width);
        for(size_t kk=0; kk < obj_indices_[k].size(); kk++)
          obj_indices_cloud.points[kk].idx = obj_indices_[k][kk];

        pcl::io::savePCDFileBinary(scene_name, obj_indices_cloud);
      }
    }
  }
}
