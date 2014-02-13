/*
 * OBJECTNESS_TEST_H_
 *
 *  Created on: Nov 6, 2012
 *      Author: aitor
 */

#ifndef OBJECTNESS_TEST_H_
#define OBJECTNESS_TEST_H_

#include <pcl/common/time.h>

void
computeTablePlane (pcl::PointCloud<pcl::PointXYZRGB>::Ptr & xyz_points, Eigen::Vector4f & table_plane, float z_dist_ = 1.5f)
{
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

  //pcl::visualization::PCLVisualizer plane_vis ("Plane visualizer");
  for (size_t i = 0; i < model_coefficients.size (); i++)
  {
    //plane_vis.removeAllPointClouds ();
    Eigen::Vector4f table_plane = Eigen::Vector4f (model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2],
                                                   model_coefficients[i].values[3]);

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
  table_plane = Eigen::Vector4f (model_coefficients[itt].values[0], model_coefficients[itt].values[1], model_coefficients[itt].values[2],
                                 model_coefficients[itt].values[3]);

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

  std::cout << "Table plane computed... " << std::endl;
}

inline void
computeEdges (pcl::PointCloud<pcl::PointXYZRGB>::Ptr & xyz_points, pcl::PointCloud<pcl::Normal>::Ptr & normals,
              pcl::PointCloud<pcl::PointXYZL>::Ptr & label_cloud, std::vector<int> & edge_indices,
              Eigen::Vector4f & table_plane, float z_dist_, int cols_)
{

  //compute organized edges
  pcl::OrganizedEdgeFromRGBNormals<pcl::PointXYZRGB, pcl::Normal, pcl::Label> oed;
  oed.setDepthDisconThreshold (0.03f);
  /*oed.setHCCannyLowThreshold (0.4f);
  oed.setHCCannyHighThreshold (1.2f);*/
  oed.setInputNormals (normals);
  oed.setEdgeType (
                   pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDING
                       | pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_NAN_BOUNDARY
                       | pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDED
                       //| pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_RGB_CANNY
                       | pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_HIGH_CURVATURE);
  oed.setInputCloud (xyz_points);

  pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
  std::vector<pcl::PointIndices> indices2;

  {
    pcl::ScopeTime t ("computing edges...");
    oed.compute (*labels, indices2);
  }

  std::cout << "Number of edge channels:" << indices2.size () << std::endl;
  for (size_t j = 0; j < indices2.size (); j++)
  {
    for (size_t i = 0; i < indices2[j].indices.size (); i++)
    {
      pcl::PointXYZL pl;
      pl.getVector3fMap () = xyz_points->points[indices2[j].indices[i]].getVector3fMap ();
      pl.label = labels->points[indices2[j].indices[i]].label; //static_cast<uint32_t>(j);

      Eigen::Vector3f xyz_p = pl.getVector3fMap ();

      if (xyz_p[2] > z_dist_)
        continue;

      float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

      if (val < 0.0)
        continue;

      if ((val < 0.01) && (pl.label & pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDED))
        continue;

      if ((val < 0.01) && (pl.label & pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_RGB_CANNY))
        continue;

      /*if ((val > 0.0075) && (pl.label & pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_HIGH_CURVATURE))
       {
       continue;
       }*/

      if ((val < 0.005) && (pl.label & pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDING))
        continue;

      if (pl.label & pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_HIGH_CURVATURE)
      {
        //check that there are no EDGELABEL_OCCLUDED close to this edges
        int uc, vc;
        int ws = 3;
        int ws2 = std::floor (static_cast<float> (ws) / 2.f);

        uc = indices2[j].indices[i] / cols_;
        vc = indices2[j].indices[i] % cols_;

        bool found = false;
        for (int u = (uc - ws2); u <= (uc + ws2); u++)
        {
          for (int v = (vc - ws2); v <= (vc + ws2); v++)
          {
            if (labels->at (v, u).label & pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDED)
            {
              found = true;
            }
          }
        }

        if (found)
        {
          continue;
        }
      }

      label_cloud->push_back (pl);
      edge_indices.push_back (indices2[j].indices[i]);
    }
  }

}

template<typename PointT>
  inline void
  computeSuperPixels (typename pcl::PointCloud<PointT>::Ptr & cloud, pcl::PointCloud<pcl::PointXYZL>::Ptr & clusters_cloud_,
                      float voxel_size = 0.005f)
  {
    typename pcl::PointCloud<PointT>::Ptr cloud_downsampled_;
    cloud_downsampled_.reset (new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setInputCloud (cloud);
    voxel_grid.setDownsampleAllData (true);
    voxel_grid.setLeafSize (voxel_size, voxel_size, voxel_size);
    voxel_grid.filter (*cloud_downsampled_);
    //initialize kdtree for search
    typename pcl::search::KdTree<PointT>::Ptr scene_downsampled_tree_;
    scene_downsampled_tree_.reset (new pcl::search::KdTree<PointT>);
    scene_downsampled_tree_->setInputCloud (cloud_downsampled_);

    pcl::PointCloud<pcl::Normal>::Ptr scene_normals_;
    scene_normals_.reset (new pcl::PointCloud<pcl::Normal> ());

    typename pcl::search::KdTree<PointT>::Ptr normals_tree (new pcl::search::KdTree<PointT>);
    normals_tree->setInputCloud (cloud_downsampled_);

    typedef pcl::NormalEstimation<PointT, pcl::Normal> NormalEstimator_;
    NormalEstimator_ n3d;
    n3d.setRadiusSearch (0.015f);
    n3d.setSearchMethod (normals_tree);
    n3d.setInputCloud (cloud_downsampled_);
    n3d.compute (*scene_normals_);

    std::vector<pcl::PointIndices> clusters;
    double eps_angle_threshold = 0.1;
    int min_points = 50;
    float curvature_threshold = 0.035f;
    float inliers_threshold_ = voxel_size;
    faat_pcl::segmentation::extractEuclideanClustersSmooth<PointT, pcl::Normal> (*cloud_downsampled_, *scene_normals_, inliers_threshold_ * 2.f,
                                                                                 scene_downsampled_tree_, clusters, eps_angle_threshold,
                                                                                 curvature_threshold, min_points);

    clusters_cloud_.reset (new pcl::PointCloud<pcl::PointXYZL>);
    clusters_cloud_->points.resize (cloud_downsampled_->points.size ());
    clusters_cloud_->width = cloud_downsampled_->width;
    clusters_cloud_->height = 1;

    for (size_t i = 0; i < cloud_downsampled_->points.size (); i++)
    {
      pcl::PointXYZL p;
      p.getVector3fMap () = cloud_downsampled_->points[i].getVector3fMap ();
      p.label = 0.f;
      clusters_cloud_->points[i] = p;
    }

    srand (
    time (NULL));
    std::random_shuffle (clusters.begin (), clusters.end ());
    int label = 1;
    for (size_t i = 0; i < clusters.size (); i++)
    {
      std::cout << clusters[i].indices.size () << std::endl;
      for (size_t j = 0; j < clusters[i].indices.size (); j++)
      {
        clusters_cloud_->points[clusters[i].indices[j]].label = label;
      }

      label++;
    }
  }

#endif /* OBJECTNESS_TEST_H_ */
