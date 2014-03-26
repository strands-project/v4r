/*
 * segmenter.h
 *
 *  Created on: Sep 7, 2013
 *      Author: aitor
 */

#ifndef SOC_SEGMENTER_H_
#define SOC_SEGMENTER_H_

#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/planar_polygon_fusion.h>
#include <pcl/segmentation/plane_coefficient_comparator.h>
#include <pcl/segmentation/euclidean_plane_coefficient_comparator.h>
#include <pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl/segmentation/edge_aware_plane_comparator.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>

template<typename PointT>
void
doSegmentation (typename pcl::PointCloud<PointT>::Ptr & xyz_points,
                  std::vector<pcl::PointIndices> & indices,
                  Eigen::Vector4f & table_plane,
                  float Z_DIST_ = 1.2f)
{
  std::cout << "Start segmentation..." << std::endl;
  pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
  ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
  ne.setMaxDepthChangeFactor (0.02f);
  ne.setNormalSmoothingSize (20.0f);
  ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<PointT, pcl::Normal>::BORDER_POLICY_IGNORE);
  ne.setInputCloud (xyz_points);
  pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
  ne.compute (*normal_cloud);

  int num_plane_inliers = 2500;

  typename pcl::PointCloud<PointT>::Ptr xyz_points_andy (new pcl::PointCloud<PointT>);
  pcl::PassThrough<PointT> pass_;
  pass_.setFilterLimits (0.f, Z_DIST_);
  pass_.setFilterFieldName ("z");
  pass_.setInputCloud (xyz_points);
  pass_.setKeepOrganized (true);
  pass_.filter (*xyz_points_andy);

  pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
  mps.setMinInliers (num_plane_inliers);
  mps.setAngularThreshold (0.017453 * 5.f); // 2 degrees
  mps.setDistanceThreshold (0.01); // 1cm
  mps.setInputNormals (normal_cloud);
  mps.setInputCloud (xyz_points_andy);

  std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
  std::vector<pcl::ModelCoefficients> model_coefficients;
  std::vector<pcl::PointIndices> inlier_indices;
  pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
  std::vector<pcl::PointIndices> label_indices;
  std::vector<pcl::PointIndices> boundary_indices;

  typename pcl::PlaneRefinementComparator<PointT, pcl::Normal, pcl::Label>::Ptr ref_comp (
                                                                                           new pcl::PlaneRefinementComparator<PointT,
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
    typename pcl::PointCloud<PointT>::Ptr plane_points (new pcl::PointCloud<PointT> (*xyz_points_andy));
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
  table_plane = Eigen::Vector4f (model_coefficients[itt].values[0], model_coefficients[itt].values[1],
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
  typename pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr
                                                                                           euclidean_cluster_comparator_ (
                                                                                                                          new pcl::EuclideanClusterComparator<
                                                                                                                              PointT,
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

    if (val >= 0.005f)
    {
      /*plane_points->points[j].x = std::numeric_limits<float>::quiet_NaN ();
       plane_points->points[j].y = std::numeric_limits<float>::quiet_NaN ();
       plane_points->points[j].z = std::numeric_limits<float>::quiet_NaN ();*/
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
  euclidean_cluster_comparator_->setDistanceThreshold (0.035f, true);

  pcl::PointCloud<pcl::Label> euclidean_labels;
  std::vector<pcl::PointIndices> euclidean_label_indices;
  pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label> euclidean_segmentation (euclidean_cluster_comparator_);
  euclidean_segmentation.setInputCloud (xyz_points_andy);
  euclidean_segmentation.segment (euclidean_labels, euclidean_label_indices);

  for (size_t i = 0; i < euclidean_label_indices.size (); i++)
  {
    if (euclidean_label_indices[i].indices.size () > 100)
    {
      indices.push_back (euclidean_label_indices[i]);
    }
  }
}

#endif /* SOC_SEGMENTER_H_ */
