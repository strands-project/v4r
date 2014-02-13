/*
 * multiplane_segmentation.hpp
 *
 *  Created on: Sep 25, 2013
 *      Author: aitor
 */

#ifndef FAAT_PCL_3D_REC_FRAMEWORK_MULTIPLANE_SEGMENTATION_HPP_
#define FAAT_PCL_3D_REC_FRAMEWORK_MULTIPLANE_SEGMENTATION_HPP_

#include <faat_pcl/3d_rec_framework/segmentation/multiplane_segmentation.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
//#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/planar_polygon_fusion.h>
#include <pcl/segmentation/plane_coefficient_comparator.h>
#include <pcl/segmentation/euclidean_plane_coefficient_comparator.h>
#include <pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl/segmentation/edge_aware_plane_comparator.h>
#include <pcl/features/integral_image_normal.h>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/connected_components.hpp>

template<typename PointT>
void
faat_pcl::MultiPlaneSegmentation<PointT>::segment(bool force_unorganized)
{
  models_.clear();

  if(input_->isOrganized() && !force_unorganized)
  {
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
    if(!normals_set_)
    {
        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
        ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
        ne.setMaxDepthChangeFactor (0.02f);
        ne.setNormalSmoothingSize (20.0f);
        ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<PointT, pcl::Normal>::BORDER_POLICY_IGNORE);
        ne.setInputCloud (input_);
        ne.compute (*normal_cloud);
    }
    else
    {
        normal_cloud = normal_cloud_;
    }

    pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
    mps.setMinInliers (min_plane_inliers_);
    mps.setAngularThreshold (0.017453 * 2.f); // 2 degrees
    mps.setDistanceThreshold (0.01); // 1cm
    mps.setMaximumCurvature(0.002);
    mps.setInputNormals (normal_cloud);
    mps.setInputCloud (input_);

    std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
    std::vector<pcl::ModelCoefficients> model_coefficients;
    std::vector<pcl::PointIndices> inlier_indices;
    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
    std::vector<pcl::PointIndices> label_indices;
    std::vector<pcl::PointIndices> boundary_indices;

    typename pcl::PlaneRefinementComparator<PointT, pcl::Normal, pcl::Label>::Ptr ref_comp (
                                                                                             new pcl::PlaneRefinementComparator<PointT,
                                                                                                 pcl::Normal, pcl::Label> ());
    ref_comp->setDistanceThreshold (0.01f, false);
    ref_comp->setAngularThreshold (0.017453 * 2.f);
    mps.setRefinementComparator (ref_comp);
    mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);
    //mps.segment (model_coefficients, inlier_indices);

    //std::cout << model_coefficients.size() << std::endl;

    if(merge_planes_)
    {
      //sort planes by size
      //check if the first plane can be merged against the others, if yes, define a new plane combining both and add it to the cue

      typedef boost::adjacency_matrix<boost::undirectedS, int> GraphPlane;
      GraphPlane mergeable_planes (model_coefficients.size ());
      for(size_t i=0; i < model_coefficients.size(); i++)
      {

        Eigen::Vector3f plane_i = Eigen::Vector3f (model_coefficients[i].values[0], model_coefficients[i].values[1],
                                                   model_coefficients[i].values[2]);

        plane_i.normalize();
        for(size_t j=(i+1); j < model_coefficients.size(); j++)
        {
          Eigen::Vector3f plane_j = Eigen::Vector3f (model_coefficients[j].values[0], model_coefficients[j].values[1],
                                                     model_coefficients[j].values[2]);

          plane_j.normalize();
          if(plane_i.dot(plane_j) > 0.9)
          {
            if(std::abs(model_coefficients[i].values[3] - model_coefficients[j].values[3]) < 0.01)
            {
              boost::add_edge (static_cast<int>(i), static_cast<int>(j), mergeable_planes);
            }
          }
        }
      }

      boost::vector_property_map<int> components (boost::num_vertices (mergeable_planes));
      int n_cc = static_cast<int> (boost::connected_components (mergeable_planes, &components[0]));

      std::vector<int> cc_sizes;
      std::vector< std::vector<int> > cc_to_model_coeff;
      cc_sizes.resize (n_cc, 0);
      cc_to_model_coeff.resize(n_cc);

      for (size_t i = 0; i < model_coefficients.size (); i++)
      {
        cc_sizes[components[i]]++;
        cc_to_model_coeff[components[i]].push_back(i);
      }

      std::vector<pcl::ModelCoefficients> new_model_coefficients;
      std::vector<pcl::PointIndices> new_inlier_indices;

      for(size_t i=0; i < cc_sizes.size(); i++)
      {
        if(cc_sizes[i] < 2)
        {
          new_model_coefficients.push_back(model_coefficients[cc_to_model_coeff[i][0]]);
          new_inlier_indices.push_back(inlier_indices[cc_to_model_coeff[i][0]]);
          continue;
        }

        //std::cout << "going to merge CC:" << cc_sizes[i] << std::endl;
        pcl::ModelCoefficients model_coeff;
        model_coeff.values.resize(4);

        for(size_t k=0; k < 4; k++)
          model_coeff.values[k] = 0.f;

        pcl::PointIndices merged_indices;
        for(size_t j=0; j < cc_to_model_coeff[i].size(); j++)
        {
          for(size_t k=0; k < 4; k++)
            model_coeff.values[k] += model_coefficients[cc_to_model_coeff[i][j]].values[k];

          merged_indices.indices.insert(merged_indices.indices.end(), inlier_indices[cc_to_model_coeff[i][j]].indices.begin(),
                                                                      inlier_indices[cc_to_model_coeff[i][j]].indices.end());
        }

        for(size_t k=0; k < 4; k++)
          model_coeff.values[k] /= static_cast<float>(cc_to_model_coeff[i].size());

        new_model_coefficients.push_back(model_coeff);
        new_inlier_indices.push_back(merged_indices);
      }

      model_coefficients = new_model_coefficients;
      inlier_indices = new_inlier_indices;
    }

    for(size_t i=0; i < model_coefficients.size(); i++)
    {
      PlaneModel<PointT> pm;
      pm.coefficients_ = model_coefficients[i];
      pm.plane_cloud_.reset(new PointTCloud);
      pm.cloud_.reset(new PointTCloud);
      pcl::copyPointCloud(*input_, *pm.cloud_);
      pcl::copyPointCloud(*input_, inlier_indices[i], *pm.plane_cloud_);
      pm.inliers_ = inlier_indices[i];
      pm.projectPlaneCloud(resolution_);
      //convex hull
      pcl::ConvexHull<PointT> convex_hull;
      convex_hull.setInputCloud (pm.plane_cloud_);
      convex_hull.setDimension (2);
      convex_hull.setComputeAreaVolume (false);
      pcl::PolygonMeshPtr mesh_out(new pcl::PolygonMesh);
      convex_hull.reconstruct (*mesh_out);
      pm.convex_hull_ = mesh_out;

      models_.push_back(pm);
    }
  }
  else
  {
    std::cout << "Unorganized" << std::endl;
    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<PointT> vg;
    PointTCloudPtr cloud_filtered (new PointTCloud);
    vg.setInputCloud (input_);
    float leaf_size_ = 0.005f;
    float dist_threshold_ = 0.01f;
    vg.setLeafSize (leaf_size_, leaf_size_, leaf_size_);
    vg.filter (*cloud_filtered);
    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl;

    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (1000);
    seg.setDistanceThreshold (dist_threshold_);

    bool plane_found = true;

    PointTCloudPtr cloud_plane (new PointTCloud);
    std::vector<pcl::PointIndices> plane_inliers_;

    //pcl::visualization::PCLVisualizer vis("vis");
    while (plane_found)
    {
      // Segment the largest planar component from the remaining cloud
      seg.setInputCloud (cloud_filtered);
      seg.segment (*inliers, *coefficients);
      if (static_cast<int>(inliers->indices.size ()) < min_plane_inliers_)
      {
        std::cout << "Could not estimate a planar model big enough for the given dataset." << std::endl;
        plane_found = false;
        break;
      }

      // Extract the planar inliers from the input cloud
      pcl::ExtractIndices<PointT> extract;
      extract.setInputCloud (cloud_filtered);
      extract.setIndices (inliers);
      extract.setNegative (false);
      // Get the points associated with the planar surface
      extract.filter (*cloud_plane);
      std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

      //save coefficients
        PlaneModel<PointT> pm;
        pm.coefficients_ = *coefficients;
        pm.plane_cloud_.reset(new PointTCloud);
        pm.cloud_.reset(new PointTCloud);
        pcl::copyPointCloud(*cloud_filtered, *pm.cloud_);
        pcl::copyPointCloud(*cloud_plane, *pm.plane_cloud_);
        pm.inliers_ = *inliers;
        pm.projectPlaneCloud(resolution_);

        //convex hull
        pcl::ConvexHull<PointT> convex_hull;
        convex_hull.setInputCloud (pm.plane_cloud_);
        convex_hull.setDimension (2);
        convex_hull.setComputeAreaVolume (false);
        pcl::PolygonMeshPtr mesh_out(new pcl::PolygonMesh);
        convex_hull.reconstruct (*mesh_out);
        pm.convex_hull_ = mesh_out;
        models_.push_back(pm);

      /*vis.addPointCloud(cloud_plane);
      vis.spin();
      vis.removeAllPointClouds();*/
      // Remove the planar inliers, extract the rest
      extract.setNegative (true);
      PointTCloudPtr cloud_f(new PointTCloud);
      extract.filter (*cloud_f);
      *cloud_filtered = *cloud_f;
    }

    std::cout << "Number of planes found:" << models_.size() << "organized:" << static_cast<int>(input_->isOrganized() && !force_unorganized) << std::endl;
  }
}

#endif /* FAAT_PCL_3D_REC_FRAMEWORK_MULTIPLANE_SEGMENTATION_HPP_ */
