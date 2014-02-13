/*
 * ggo.cpp
 *
 *  Created on: Apr 17, 2013
 *      Author: aitor
 */

#include <faat_pcl/object_modelling/ggo.h>
#include <boost/filesystem.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <faat_pcl/registration/visibility_reasoning.h>
#include <boost/graph/connected_components.hpp>
#include <pcl/surface/convex_hull.h>
#include <faat_pcl/object_modelling/impl/prim.hpp>
#include <pcl/registration/icp.h>
#include <pcl/registration/elch.h>
#include <pcl/filters/filter.h>
#include <faat_pcl/registration/mv_lm_icp.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/normal_estimator.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <faat_pcl/registration/uniform_sampling.h>

namespace bf = boost::filesystem;

template <typename PointT>
faat_pcl::object_modelling::ggo<PointT>::ggo ()
{
  max_poses_ = 3;
}

template <typename PointT>
mets::gol_type
faat_pcl::object_modelling::ggo<PointT>::evaluateSolution (const std::vector<bool> & active, int changed,
                                                   std::vector<boost::graph_traits<Graph>::edge_descriptor> & edge_iterators, Graph & ST_so_far_)
{
  float sum_fsv_fraction = 0.f; //pairwise (directly connected) fsv penalization...
  float sum_osv_fraction = 0.f; //pairwise (directly connected) osv penalization...
  float sum_overlap = 0.f;
  float global_fsv_fraction = 0.f; //directly and non-directly connected fsv penalization
  float global_osv_fraction = 0.f; //directly and non-directly connected osv penalization
  int count = 0;
  int nedges_used = 0;
  float sum_pw_registration = 0.f;
  float color_weight_sum = 0.f;
  float min_overlap_ = 0.3f;
  float osv_cutoff = 0.1f;
  float fsv_cutoff = 0.01f;

  boost::property_map<Graph, boost::edge_weight_t>::type EdgeWeightMap = get (boost::edge_weight_t (), LR_graph_);

  for (size_t i = 0; i < active.size (); i++)
  {
    if (active[i])
    {
      nedges_used++;

      int min = std::min (boost::source (edge_iterators[i], LR_graph_), boost::target (edge_iterators[i], LR_graph_));
      int max = std::max (boost::source (edge_iterators[i], LR_graph_), boost::target (edge_iterators[i], LR_graph_));
      float fsv = pairwise_registration_[min][max][(get (EdgeWeightMap, edge_iterators[i])).first].fsv_fraction_;
      float osv = pairwise_registration_[min][max][(get (EdgeWeightMap, edge_iterators[i])).first].osv_fraction_;
      float overlap = std::min(pairwise_registration_[min][max][(get (EdgeWeightMap, edge_iterators[i])).first].overlap_factor_, min_overlap_);
      float col_error = pairwise_registration_[min][max][(get (EdgeWeightMap, edge_iterators[i])).first].color_error_;
      color_weight_sum += col_error;
      sum_fsv_fraction += std::max(fsv, fsv_cutoff) - fsv_cutoff;
      sum_osv_fraction += osv;
      sum_overlap += overlap;
      //sum_pw_registration += overlap * (1.f - fsv) * (1.f - osv);
      sum_pw_registration += overlap * col_error;
      //std::cout << min << "-" << max << " " << (get (EdgeWeightMap, edge_iterators[i])).first << " " << pairwise_registration_[min][max][(get (EdgeWeightMap, edge_iterators[i])).first].fsv_fraction_ << std::endl;
    }
  }

  /*Eigen::Vector4f cam (0, 0, 0, 1);
  cam = absolute_poses_[i] * cam;*/

  boost::vector_property_map<int> components (num_vertices (ST_so_far_));
  int n_cc = static_cast<int> (boost::connected_components (ST_so_far_, &components[0]));
  //std::cout << "Number of connected components..." << n_cc << std::endl;

  //for each connected component (3 or more elements) compute FSV fraction using absolute poses
  std::vector<int> cc_sizes;
  std::vector<std::vector<boost::graph_traits<Graph>::vertex_descriptor> > ccs;
  cc_sizes.resize (n_cc, 0);
  ccs.resize (n_cc);

  float degree = 0;
  for (size_t i = 0; i < (num_vertices (ST_so_far_)); i++)
  {
    cc_sizes[components[i]]++;
    ccs[components[i]].push_back (static_cast<int> (i));
    degree += (boost::degree(i, ST_so_far_));
    //std::cout << boost::degree(i, ST_so_far_) << " ";
  }

  //std::cout << std::endl;

  degree /= static_cast<float>((num_vertices (ST_so_far_)));

  float dev_avg_degree = 0.f;
  for (size_t i = 0; i < (num_vertices (ST_so_far_)); i++)
  {
    dev_avg_degree += std::abs( boost::degree(i, ST_so_far_) - degree);
  }

  //std::cout << dev_avg_degree << std::endl;

//degree /= static_cast<float>((num_vertices (ST_so_far_)));

  float dev_avg_distance_cam = 0.f;
  float dist_cameras = 0.f;
  float avg_dist_nn_cam_all = 0.f;
  std::vector<float> distance_bet_cameras;
  for (size_t i = 0; i < cc_sizes.size (); i++)
  {
    if (cc_sizes[i] > 2)
    {
      //compute absolute for connected subgraph and compute fsv fraction with all views
      graph_traits<Graph>::vertex_descriptor root = 0;
      std::vector<Eigen::Matrix4f> absolute_poses;
      std::vector<boost::graph_traits<Graph>::vertex_descriptor> cc;
      computeAbsolutePosesFromMST (ST_so_far_, root, absolute_poses, ccs[i]);

      //absolute poses align ccs[i] to ccs[0]
      //transform all clouds to common RF
      for (size_t k = 0; k < absolute_poses.size (); k++)
      {
        for (size_t j = (k + 1); j < absolute_poses.size (); j++)
        {

          float overlap = pairwise_registration_[k][j][(get (EdgeWeightMap, edge_iterators[i])).first].overlap_factor_;
          if(overlap < min_overlap_)
            continue;

          //transformation from cloud k to cloud_j
          Eigen::Matrix4f k_to_j;
          k_to_j = absolute_poses[j].inverse () * absolute_poses[k];
          faat_pcl::registration::VisibilityReasoning<PointT> vr (focal_length_, cx_, cy_);
          vr.setThresholdTSS(0.01f);
          float fsv_ij = vr.computeFSV (range_images_[ccs[i][k]], clouds_[ccs[i][j]], k_to_j.inverse ());
          float fsv_ji = vr.computeFSV (range_images_[ccs[i][j]], clouds_[ccs[i][k]], k_to_j);
          float fsv = std::max (fsv_ij, fsv_ji);
          global_fsv_fraction += std::max(fsv, fsv_cutoff) - fsv_cutoff;

          float osv_ij = vr.computeOSV (range_images_[ccs[i][k]], clouds_[ccs[i][j]], k_to_j.inverse ());
          float osv_ji = vr.computeOSV (range_images_[ccs[i][j]], clouds_[ccs[i][k]], k_to_j);
          float osv = std::max (osv_ij, osv_ji);
          global_osv_fraction += osv;
          count++;
        }
      }
    }
  }

  //cost function
  float normalize = static_cast<float>(nedges_used);
  //float to_max = color_weight_sum / normalize + sum_overlap / normalize; //dist_between_cameras; // sum_overlap; //nedges_used * 0.f;
  float to_max = sum_pw_registration / normalize;
  float to_min =  sum_fsv_fraction / normalize + n_cc * 0.1f; // + dev_avg_degree * 0.5f;
  std::cout << n_cc << " " << dev_avg_degree * 0.5f << std::endl;
  if (count > 0)
  {
    to_min += (global_fsv_fraction / static_cast<float> (count));
    //to_min += dev_avg_distance_cam;
    //to_max += avg_dist_nn_cam_all; // / static_cast<float> (count);
    //to_min += (global_fsv_fraction);
    //to_min += (global_osv_fraction / static_cast<float> (count));
  }

  std::cout << "sum_fsv_fraction:" << sum_fsv_fraction << " global:" << global_fsv_fraction << " " << " " << " degree:" << degree << std::endl;
  std::cout << sum_overlap / normalize << std::endl;
  std::cout << "cost:" << to_max * -1.f + to_min << std::endl;
  return to_max * -1.f + to_min;
}

template <typename PointT>
void
faat_pcl::object_modelling::ggo<PointT>::computeHuberST (Graph & local_registration, Graph & mst)
{
  boost::graph_traits<Graph>::edge_iterator ei, ei_end;
  std::vector<std::pair<boost::graph_traits<Graph>::edge_iterator, float> > sorted_edges;
  boost::property_map<Graph, boost::edge_weight_t>::type EdgeWeightMap = get (boost::edge_weight_t (), local_registration);

  for (boost::tie (ei, ei_end) = boost::edges (local_registration); ei != ei_end; ++ei)
  {
    std::pair<boost::graph_traits<Graph>::edge_iterator, float> p;
    p.first = ei;
    p.second = static_cast<float> ((get (EdgeWeightMap, *ei)).second );
    sorted_edges.push_back (p);
  }

  std::sort (
             sorted_edges.begin (),
             sorted_edges.end (),
             boost::bind (&std::pair<boost::graph_traits<Graph>::edge_iterator, float>::second, _1)
                 < boost::bind (&std::pair<boost::graph_traits<Graph>::edge_iterator, float>::second, _2));

  //for (boost::tie(ei, ei_end) = boost::out_edges(start, mst); ei != ei_end; ++ei)
  for (size_t e = 0; e < sorted_edges.size (); e++)
  {
    boost::graph_traits<Graph>::vertex_descriptor target = boost::target (*(sorted_edges[e].first), local_registration);
    bool found = false;
    int target_v = static_cast<int> (target);
    bfs_time_visitor vis (found, target_v);
    boost::breadth_first_search (mst, boost::source (*(sorted_edges[e].first), local_registration), boost::visitor (vis));

    if (!vis.getFound ()) //vertices are not connected, add edge
    {
      //moves_m.push_back (new move (i));
      boost::add_edge (static_cast<int> (boost::source (*(sorted_edges[e].first), local_registration)),
                       static_cast<int> (boost::target (*(sorted_edges[e].first), local_registration)), mst);

      //check global consistence
      float global_fsv_fraction = 0.f; //directly and non-directly connected fsv penalization
      int count = 0;

      boost::vector_property_map<int> components (num_vertices (mst));
      int n_cc = static_cast<int> (boost::connected_components (mst, &components[0]));

      //for each connected component (3 or more elements) compute FSV fraction using absolute poses
      std::vector<int> cc_sizes;
      std::vector<std::vector<boost::graph_traits<Graph>::vertex_descriptor> > ccs;
      cc_sizes.resize (n_cc, 0);
      ccs.resize (n_cc);

      for (size_t i = 0; i < (num_vertices (mst)); i++)
      {
        cc_sizes[components[i]]++;
        ccs[components[i]].push_back (static_cast<int> (i));
      }

      bool globally_consistent = true;

      for (size_t i = 0; (i < cc_sizes.size ()) && globally_consistent; i++)
      {
        if (cc_sizes[i] > 2)
        {
          //compute absolute for connected subgraph and compute fsv fraction with all views
          graph_traits<Graph>::vertex_descriptor root = 0;
          std::vector<Eigen::Matrix4f> absolute_poses;
          std::vector<boost::graph_traits<Graph>::vertex_descriptor> cc;
          computeAbsolutePosesFromMST (mst, root, absolute_poses, ccs[i]);

          //absolute poses align ccs[i] to ccs[0]
          //transform all clouds to common RF
          for (size_t k = 0; (k < absolute_poses.size ()) && (globally_consistent); k++)
          {
            for (size_t j = (k + 1); (j < absolute_poses.size ()) && (globally_consistent); j++)
            {
              //transformation from cloud k to cloud_j
              Eigen::Matrix4f k_to_j;
              k_to_j = absolute_poses[j].inverse () * absolute_poses[k];
              faat_pcl::registration::VisibilityReasoning<PointT> vr (focal_length_, cx_, cy_);
              float fsv_ij = vr.computeFSV (range_images_[ccs[i][k]], clouds_[ccs[i][j]], k_to_j.inverse ());
              float fsv_ji = vr.computeFSV (range_images_[ccs[i][j]], clouds_[ccs[i][k]], k_to_j);
              float fsv = std::max (fsv_ij, fsv_ji);
              std::cout << fsv << std::endl;
              if (fsv > 0.01f)
              {
                globally_consistent = false;
              }
            }
          }
        }
      }

      if (!globally_consistent)
      {
        std::cout << "remove edge..." << std::endl;
        boost::remove_edge (static_cast<int> (boost::source (*(sorted_edges[e].first), local_registration)),
                            static_cast<int> (boost::target (*(sorted_edges[e].first), local_registration)), mst);
      }
    }
    else
    {

    }
  }
}

template <typename PointT>
void
faat_pcl::object_modelling::ggo<PointT>::computeAbsolutePosesRecursive
                                                (Graph & mst,
                                                boost::graph_traits<Graph>::vertex_descriptor & start,
                                                boost::graph_traits<Graph>::vertex_descriptor & coming_from,
                                                Eigen::Matrix4f accum,
                                                std::vector<Eigen::Matrix4f> & absolute_poses,
                                                std::vector<boost::graph_traits<Graph>::vertex_descriptor> & cc,
                                                std::vector<boost::graph_traits<Graph>::vertex_descriptor> & graph_to_cc)

{
  if (boost::degree (start, mst) == 1)
  {
    //check if target is like coming_from
    boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
    for (boost::tie (ei, ei_end) = boost::out_edges (start, mst); ei != ei_end; ++ei)
    {
      if (target (*ei, mst) == coming_from)
        return;
    }
  }

  //std::cout << "out degree:" << boost::out_degree (start, mst) << " vertex:" << graph_to_cc[start] << " " << start << std::endl;

  //iterate over edges and call recursive
  //TODO: Sort this edges if out_degree > 1
  boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
  std::vector<std::pair<boost::graph_traits<Graph>::out_edge_iterator, float> > sorted_edges;
  boost::property_map<Graph, boost::edge_weight_t>::type EdgeWeightMap = get (boost::edge_weight_t (), mst);

  for (boost::tie (ei, ei_end) = boost::out_edges (start, mst); ei != ei_end; ++ei)
  {

    if (target (*ei, mst) == coming_from)
    {
      continue;
    }
    std::pair<boost::graph_traits<Graph>::out_edge_iterator, float> p;
    p.first = ei;
    p.second = static_cast<float> ((get (EdgeWeightMap, *ei)).second);
    sorted_edges.push_back (p);
  }

  std::sort (
             sorted_edges.begin (),
             sorted_edges.end (),
             boost::bind (&std::pair<boost::graph_traits<Graph>::out_edge_iterator, float>::second, _1)
                 < boost::bind (&std::pair<boost::graph_traits<Graph>::out_edge_iterator, float>::second, _2));

  //for (boost::tie(ei, ei_end) = boost::out_edges(start, mst); ei != ei_end; ++ei)
  //std::cout << "edges coming out:" <<  sorted_edges.size() << std::endl;
  for (size_t i = 0; i < sorted_edges.size (); i++)
  {
    Eigen::Matrix4f internal_accum;
    boost::graph_traits<Graph>::vertex_descriptor dst = target (*(sorted_edges[i].first), mst);
    int idx_k = (get (EdgeWeightMap, *(sorted_edges[i].first))).first;
    //std::cout << "going to:" << dst << " using pose:" << idx_k << std::endl;

    if (dst < start)
    {
      //std::cout << "(Inverse...) Number of correspondences:" << pairwise_corresp_clusters_[dst][start][0].size() << std::endl;
      internal_accum = accum * pairwise_poses_[dst][start][idx_k].inverse ();
      absolute_poses[graph_to_cc[dst]] = internal_accum;
    }
    else
    {
      //std::cout << "Number of correspondences:" << pairwise_corresp_clusters_[start][dst][0].size() << std::endl;
      internal_accum = accum * pairwise_poses_[start][dst][idx_k];
      absolute_poses[graph_to_cc[dst]] = internal_accum;
    }

    computeAbsolutePosesRecursive (mst, dst, start, internal_accum, absolute_poses, cc, graph_to_cc);
  }
}

template <typename PointT>
void
faat_pcl::object_modelling::ggo<PointT>::computeAbsolutePosesFromMST (Graph & mst, boost::graph_traits<Graph>::vertex_descriptor & root,
                                                              std::vector<Eigen::Matrix4f> & absolute_poses,
                                                              std::vector<boost::graph_traits<Graph>::vertex_descriptor> & cc)
{

  absolute_poses.clear ();
  absolute_poses.resize (cc.size ());
  absolute_poses[root] = Eigen::Matrix4f::Identity ();

  for (size_t i = 0; i < absolute_poses.size (); i++)
    absolute_poses[i] = Eigen::Matrix4f::Identity ();

  //std::cout << "root vertex:" << root << std::endl;

  std::vector<boost::graph_traits<Graph>::vertex_descriptor> graph_to_cc;
  graph_to_cc.resize (clouds_.size (), -1);

  for (size_t i = 0; i < cc.size (); i++)
    graph_to_cc[cc[i]] = static_cast<int> (i);

  computeAbsolutePosesRecursive (mst, cc[root], cc[root], absolute_poses[root], absolute_poses, cc, graph_to_cc);
}

template <typename PointT>
void
faat_pcl::object_modelling::ggo<PointT>::refineAbsolutePoses()
{
  //only for the biggest connected component
  std::vector<PointCloudPtr> registered_clouds;
  std::vector<pcl::PointCloud<pcl::Normal>::Ptr> registered_normals;

  boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
  normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
  normal_estimator->setCMR (false);
  normal_estimator->setDoVoxelGrid (true);
  normal_estimator->setRemoveOutliers (false);
  normal_estimator->setMinNRadius (27);
  normal_estimator->setValuesForCMRFalse (0.001f, 0.018f);

  for (size_t i = 0; i < clouds_.size (); i++)
  {

    if (!cloud_registered_[i])
      continue;

    std::stringstream cloud_name;
    cloud_name << "cloud_" << i << ".pcd";

    //compute normals
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    PointCloudPtr vx (new pcl::PointCloud<PointT>);
    normal_estimator->estimate(clouds_[i], vx, normals);

    PointCloudPtr pp (new pcl::PointCloud<PointT>);
    pcl::transformPointCloud (*vx, *pp, absolute_poses_[i]);
    for(size_t kk=0; kk < normals->points.size(); kk++)
    {
      Eigen::Matrix<float, 3, 1> nt (normals->points[kk].normal_x, normals->points[kk].normal_y, normals->points[kk].normal_z);
      normals->points[kk].normal_x = static_cast<float> (absolute_poses_[i] (0, 0) * nt.coeffRef (0) + absolute_poses_[i] (0, 1) * nt.coeffRef (1) + absolute_poses_[i] (0, 2) * nt.coeffRef (2));
      normals->points[kk].normal_y = static_cast<float> (absolute_poses_[i] (1, 0) * nt.coeffRef (0) + absolute_poses_[i] (1, 1) * nt.coeffRef (1) + absolute_poses_[i] (1, 2) * nt.coeffRef (2));
      normals->points[kk].normal_z = static_cast<float> (absolute_poses_[i] (2, 0) * nt.coeffRef (0) + absolute_poses_[i] (2, 1) * nt.coeffRef (1) + absolute_poses_[i] (2, 2) * nt.coeffRef (2));
    }

    registered_clouds.push_back(pp);
    registered_normals.push_back(normals);
  }

  std::vector<std::vector<bool> > A;
  A.resize(registered_clouds.size());
  for(size_t i=0; i < registered_clouds.size(); i++)
  {
    A[i].resize(registered_clouds.size(), true);
  }

  float resolution = 0.002f;
  for (size_t i = 0; i < registered_clouds.size (); i++)
  {
    std::vector<int> indices_p;
    pcl::removeNaNFromPointCloud(*registered_clouds[i], *registered_clouds[i], indices_p);

    pcl::UniformSampling<PointT> keypoint_extractor;
    keypoint_extractor.setRadiusSearch (resolution);
    keypoint_extractor.setInputCloud (registered_clouds[i]);

    pcl::PointCloud<int> keypoints_idxes;
    keypoint_extractor.compute (keypoints_idxes);

    std::vector<int> indices;
    indices.resize (keypoints_idxes.points.size ());
    for (size_t jj = 0; jj < indices.size (); jj++)
      indices[jj] = keypoints_idxes.points[jj];

    PointCloudPtr tmp(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud (*registered_clouds[i], indices, *tmp);
    pcl::copyPointCloud (*tmp, *registered_clouds[i]);
  }

  std::vector<int> pointIdxNKNSearch;
  std::vector<float> pointNKNSquaredDistance;
  float inlier = 0.01f;
  for (size_t i = 0; i < registered_clouds.size (); i++)
  {
    A[i][i] = false;
    pcl::octree::OctreePointCloudSearch<PointT> octree (0.003);
    octree.setInputCloud (registered_clouds[i]);
    octree.addPointsFromInputCloud ();

    for (size_t j = i; j < registered_clouds.size (); j++)
    {
      //compute overlap
      int overlap = 0;
      for (size_t kk = 0; kk < registered_clouds[j]->points.size (); kk++)
      {
        if(pcl_isnan(registered_clouds[j]->points[kk].x))
          continue;

        if (octree.nearestKSearch (registered_clouds[j]->points[kk], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
          float d = sqrt (pointNKNSquaredDistance[0]);
          if (d < inlier)
          {
            overlap++;
          }
        }
      }

      float ov_measure_1 = overlap / static_cast<float>(registered_clouds[j]->points.size());
      float ov_measure_2 = overlap / static_cast<float>(registered_clouds[i]->points.size());
      float ff = 0.3f;
      if(!(ov_measure_1 > ff || ov_measure_2 > ff))
      {
        A[i][j] = false;
        A[j][i] = false;
      }
    }
  }

  for (size_t i = 0; i < registered_clouds.size (); i++)
  {
    for (size_t j = 0; j < registered_clouds.size (); j++)
      std::cout << A[i][j] << " ";

    std::cout << std::endl;
  }

  faat_pcl::registration::MVNonLinearICP<PointT> icp_nl(resolution);
  icp_nl.setClouds(registered_clouds);
  icp_nl.setMaxCorrespondenceDistance(0.01f);
  icp_nl.setInlierThreshold(resolution);
  icp_nl.setVisIntermediate(true);
  icp_nl.setSparseSolver(true);
  //icp_nl.setInputNormals(registered_normals);
  icp_nl.setAdjacencyMatrix(A);
  icp_nl.compute ();
  std::vector<Eigen::Matrix4f> transformations;
  icp_nl.getTransformation(transformations);

  int t=0;
  for (size_t i = 0; i < clouds_.size (); i++)
  {

    if (!cloud_registered_[i])
      continue;

    absolute_poses_[i] = transformations[t] * absolute_poses_[i];
    t++;
  }

  visualizeGlobalAlignment (true, "refined poses (MML)");
}

//compute the absolute poses for each connected component in mst and visualize
template <typename PointT>
void
faat_pcl::object_modelling::ggo<PointT>::visualizeGlobalAlignmentCCAbsolutePoses (Graph & local_registration, Graph & mst)
{
  {
    boost::vector_property_map<int> components (num_vertices (mst));
    int n_cc = static_cast<int> (boost::connected_components (mst, &components[0]));
    std::cout << "Number of connected components in ST..." << n_cc << std::endl;

    std::vector<int> cc_sizes;
    std::vector<std::vector<boost::graph_traits<Graph>::vertex_descriptor> > ccs;
    cc_sizes.resize (n_cc, 0);
    ccs.resize (n_cc);

    for (size_t i = 0; i < (num_vertices (local_registration)); i++)
    {
      cc_sizes[components[i]]++;
      ccs[components[i]].push_back (static_cast<int> (i));
    }

    int max_cc_id, max_size;
    max_size = -1;
    for (size_t i = 0; i < cc_sizes.size (); i++)
    {
      if (cc_sizes[i] > max_size)
      {
        max_cc_id = static_cast<int> (i);
        max_size = cc_sizes[i];
      }
    }

    for (size_t i = 0; i < cc_sizes.size (); i++)
    {
      std::cout << "cc size:" << cc_sizes[i] << std::endl;
    }

    for (size_t k = 0; k < ccs.size (); k++)
    {
      graph_traits<Graph>::vertex_descriptor root = 0;

      std::cout << boost::num_edges(mst) << std::endl;
      std::vector<Eigen::Matrix4f> absolute_poses;
      std::vector<boost::graph_traits<Graph>::vertex_descriptor> cc;
      computeAbsolutePosesFromMST (mst, root, absolute_poses, ccs[k]);

      absolute_poses_.clear ();
      absolute_poses_.resize (clouds_.size ());
      for (size_t i = 0; i < absolute_poses_.size (); i++)
        absolute_poses_[i] = Eigen::Matrix4f::Identity ();

      for (size_t i = 0; i < ccs[k].size (); i++)
        absolute_poses_[ccs[k][i]] = absolute_poses[i];

      cloud_registered_.clear ();
      cloud_registered_.resize (clouds_.size (), false);
      for (size_t i = 0; i < ccs[k].size (); i++)
        cloud_registered_[ccs[k][i]] = true;

      ST_graph_.clear();
      boost::copy_graph (mst, ST_graph_);

      for (size_t i = 0; i < (num_vertices (mst)); i++)
      {
        std::cout << boost::degree(i, mst) << " ";
      }

      std::cout << std::endl;

      for (size_t i = 0; i < (num_vertices (ST_graph_)); i++)
      {
        std::cout << boost::degree(i, ST_graph_) << " ";
      }

      std::cout << std::endl;

      if(ccs[k].size () > 2)
      {
        visualizeGlobalAlignment (true);
      }
    }
  }
}

template <typename PointT>
void
faat_pcl::object_modelling::ggo<PointT>::process ()
{
  computeFSVFraction ();

  const int num_nodes = (static_cast<int> (clouds_.size ()));
  Graph G (num_nodes);

  for (size_t i = 0; i < clouds_.size (); i++)
  {
    for (size_t j = (i + 1); j < clouds_.size (); j++)
    {
      std::cout << "Number of poses..." << pairwise_poses_[i][j].size () << " ... (" << i << "," << j << ")" << std::endl;

      if (pairwise_poses_[i][j].size () == 0)
        continue;

      for(size_t k=0; k < pairwise_poses_[i][j].size (); k++)
      {
        if (pairwise_registration_[i][j][k].fsv_fraction_ < 0.05f)
        {
          EdgeWeightProperty w = std::make_pair(static_cast<int>(k), pairwise_registration_[i][j][k].fsv_fraction_); // + (1.f / (pairwise_registration_[i][j][kk].reg_error_));
          add_edge (static_cast<int> (i), static_cast<int> (j), w, G);
        }
      }
    }
  }

  //visualizePairWiseAlignment();

  std::cout << "Num of edges:" << boost::num_edges(G) << std::endl;
  boost::vector_property_map<int> components (num_vertices (G));
  int n_cc = static_cast<int> (boost::connected_components (G, &components[0]));
  std::cout << "Number of connected components..." << n_cc << std::endl;

  LR_graph_.clear();
  boost::copy_graph (G, LR_graph_);

  /*Graph G_mst_huber (num_nodes);
  computeHuberST (G, G_mst_huber);
  visualizeGlobalAlignmentCCAbsolutePoses (G, G_mst_huber);*/

  //USING THE graph, do a SA optimization to select the best edges that register all views (or as max as possible) consistently...
  SAModel model;
  model.setOptimizer (this);
  model.cost_ = std::numeric_limits<float>::max ();
  std::vector<bool> init_sol;
  init_sol.resize (num_edges (G), false);
  model.setSolution (init_sol);
  model.setLRGraph (G);
  SAModel * best = new SAModel (model);

  mets::noimprove_termination_criteria noimprove (1000);
  mets::exponential_cooling linear_cooling;

  move_manager neigh (LR_graph_);

  mets::best_ever_solution best_recorder (*best);
  /*mets::simulated_annealing<move_manager> sa (model, best_recorder, neigh, noimprove, linear_cooling, 5000, 1e-7, 1);
   sa.setApplyAndEvaluate (true);

   nviews_used_ = 0;
   {
   pcl::ScopeTime t ("SA search...");
   sa.search ();
   }*/

  /*mets::simple_tabu_list tabu_list (init_sol.size () * sqrt (1.0 * init_sol.size ()));
  mets::best_ever_criteria aspiration_criteria;
  mets::tabu_search<move_manager> tabu_search (model, best_recorder, neigh, tabu_list, aspiration_criteria, noimprove);

  {
    pcl::ScopeTime t ("TABU search...");
    try
    {
      tabu_search.search ();
    }
    catch (mets::no_moves_error e)
    {
      //} catch (std::exception e) {

    }
  }*/

  mets::local_search<move_manager> local ( model, best_recorder, neigh, 0, false);
  {
    pcl::ScopeTime t ("local search...");
    local.search ();
  }

  std::vector<boost::graph_traits<Graph>::edge_descriptor> edge_iterators;
  boost::graph_traits<Graph>::edge_iterator ei, ei_end;
  for (boost::tie (ei, ei_end) = boost::edges (LR_graph_); ei != ei_end; ++ei)
  {
    edge_iterators.push_back (*ei);
  }

  boost::property_map<Graph, boost::edge_weight_t>::type EdgeWeightMap = get (boost::edge_weight_t (), LR_graph_);
  Graph ST (num_vertices (LR_graph_));
  const SAModel& best_seen_ = static_cast<const SAModel&> (best_recorder.best_seen ());
  for (size_t i = 0; i < best_seen_.solution_.size (); i++)
  {
    std::cout << best_seen_.solution_[i] << std::endl;
    if (best_seen_.solution_[i])
    {
      EdgeWeightProperty w = (get (EdgeWeightMap, edge_iterators[i]));
      boost::add_edge (static_cast<int> (boost::source (edge_iterators[i], LR_graph_)),
                       static_cast<int> (boost::target (edge_iterators[i], LR_graph_)), w, ST);
    }
  }

  std::cout << "best solution:" << best_seen_.cost_ << std::endl;
  visualizeGlobalAlignmentCCAbsolutePoses(LR_graph_, ST);

  {
    boost::vector_property_map<int>
    components (num_vertices( ST ));
    int n_cc = static_cast<int> (boost::connected_components (ST, &components[0]));
    std::cout << "Number of connected components in ST..." << n_cc << std::endl;

    std::vector<int> cc_sizes;
    std::vector<std::vector<boost::graph_traits<Graph>::vertex_descriptor> > ccs;
    cc_sizes.resize (n_cc, 0);
    ccs.resize (n_cc);

    for (size_t i = 0; i < (num_vertices (LR_graph_)); i++)
    {
      cc_sizes[components[i]]++;
      ccs[components[i]].push_back (static_cast<int> (i));
    }

    int max_cc_id, max_size;
    max_size = -1;
    for (size_t i = 0; i < cc_sizes.size (); i++)
    {
      if (cc_sizes[i] > max_size)
      {
        max_cc_id = static_cast<int> (i);
        max_size = cc_sizes[i];
      }
    }

    for (size_t i = 0; i < cc_sizes.size (); i++)
    {
      std::cout << "cc size:" << cc_sizes[i] << std::endl;
    }

    for (size_t k = 0; k < ccs.size (); k++)
    {
      graph_traits<Graph>::vertex_descriptor root = 0;

      std::cout << boost::num_edges (ST) << std::endl;
      std::vector<Eigen::Matrix4f> absolute_poses;
      std::vector<boost::graph_traits<Graph>::vertex_descriptor> cc;
      computeAbsolutePosesFromMST (ST, root, absolute_poses, ccs[k]);

      absolute_poses_.clear ();
      absolute_poses_.resize (clouds_.size ());
      for (size_t i = 0; i < absolute_poses_.size (); i++)
        absolute_poses_[i] = Eigen::Matrix4f::Identity ();

      for (size_t i = 0; i < ccs[k].size (); i++)
        absolute_poses_[ccs[k][i]] = absolute_poses[i];

      cloud_registered_.clear ();
      cloud_registered_.resize (clouds_.size (), false);
      for (size_t i = 0; i < ccs[k].size (); i++)
        cloud_registered_[ccs[k][i]] = true;

      ST_graph_.clear ();
      boost::copy_graph (ST, ST_graph_);
    }

    //refine poses with multiview LM
    //refineAbsolutePoses();
  }

  visualizePairWiseAlignment ();
  /*{
    boost::vector_property_map<int> components (num_vertices (ST));
    int n_cc = static_cast<int> (boost::connected_components (ST, &components[0]));
    std::cout << "Number of connected components in ST..." << n_cc << std::endl;

    std::vector<int> cc_sizes;
    std::vector<std::vector<boost::graph_traits<Graph>::vertex_descriptor> > ccs;
    cc_sizes.resize (n_cc, 0);
    ccs.resize (n_cc);

    for (size_t i = 0; i < (num_vertices (G)); i++)
    {
      cc_sizes[components[i]]++;
      ccs[components[i]].push_back (static_cast<int> (i));
    }

    int max_cc_id, max_size;
    max_size = -1;
    for (size_t i = 0; i < cc_sizes.size (); i++)
    {
      if (cc_sizes[i] > max_size)
      {
        max_cc_id = static_cast<int> (i);
        max_size = cc_sizes[i];
      }
    }

    for (size_t i = 0; i < cc_sizes.size (); i++)
    {
      std::cout << cc_sizes[i] << std::endl;
    }

    for (size_t k = 0; k < ccs.size (); k++)
    {
      graph_traits<Graph>::vertex_descriptor root = 0;

      std::vector<Eigen::Matrix4f> absolute_poses;
      std::vector<boost::graph_traits<Graph>::vertex_descriptor> cc;
      computeAbsolutePosesFromMST (ST, root, absolute_poses, ccs[k]);

      absolute_poses_.clear ();
      absolute_poses_.resize (clouds_.size ());
      for (size_t i = 0; i < absolute_poses_.size (); i++)
        absolute_poses_[i] = Eigen::Matrix4f::Identity ();

      for (size_t i = 0; i < ccs[k].size (); i++)
        absolute_poses_[ccs[k][i]] = absolute_poses[i];

      cloud_registered_.clear ();
      cloud_registered_.resize (clouds_.size (), false);
      for (size_t i = 0; i < ccs[k].size (); i++)
        cloud_registered_[ccs[k][i]] = true;

      boost::copy_graph (ST, ST_graph_);

      visualizeGlobalAlignment (true);
    }

    visualizePairWiseAlignment ();

  }*/
}

template <typename PointT>
void
faat_pcl::object_modelling::ggo<PointT>::visualizeGlobalAlignment (bool visualize_cameras,
                                                                           std::string name)
{
  pcl::visualization::PCLVisualizer vis (name.c_str());
  pcl::PointCloud<pcl::PointXYZ>::Ptr camera_positions (new pcl::PointCloud<pcl::PointXYZ>);
  int v2, v3;
  vis.createViewPort (0., 0, 0.5, 1, v2);
  vis.createViewPort (0.5, 0, 1, 1, v3);

  typename pcl::PointCloud<PointT>::Ptr big_cloud(new pcl::PointCloud<PointT>);

  for (size_t i = 0; i < clouds_.size (); i++)
  {

    if (!cloud_registered_[i])
      continue;

    std::stringstream cloud_name;
    cloud_name << "cloud_" << i;

    PointCloudPtr pp (new pcl::PointCloud<PointT>);
    pcl::transformPointCloud (*clouds_[i], *pp, absolute_poses_[i]);
    *big_cloud+=*pp;

    float rgb_m;
    bool exists_m;

    typedef pcl::PointCloud<PointT> CloudM;
    typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

    pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (clouds_[i]->points[0], "rgb", exists_m, rgb_m));
    if (exists_m)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::copyPointCloud(*pp,*cloud_rgb);
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (cloud_rgb);
      vis.addPointCloud<pcl::PointXYZRGB> (cloud_rgb, handler_rgb, cloud_name.str ());
    }
    else
    {
      pcl::visualization::PointCloudColorHandlerRandom<PointT> handler_rgb (pp);
      vis.addPointCloud<PointT> (pp, handler_rgb, cloud_name.str ());
    }

    if (visualize_cameras)
    {
      Eigen::Vector4f cam (0, 0, 0, 1);
      cam = absolute_poses_[i] * cam;

      std::stringstream camera_text;
      camera_text << "c" << i << std::endl;
      pcl::PointXYZ ptext;
      ptext.getVector4fMap () = cam;

      std::string text = camera_text.str ();
      vis.addText3D (text, ptext, 0.02, 0.0, 1.0, 0.0, camera_text.str ());

      camera_positions->push_back (ptext);
    }
  }


  for (size_t i = 0; i < (num_vertices (ST_graph_)); i++)
  {
    std::cout << boost::degree(i, ST_graph_) << " ";
  }

  std::cout << std::endl;

  if (visualize_cameras)
  {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_rgb (camera_positions, 255, 0, 0);
    vis.addPointCloud<pcl::PointXYZ> (camera_positions, handler_rgb, "camera positions");

    pcl::ConvexHull<pcl::PointXYZ> convex_hull;
    convex_hull.setInputCloud (camera_positions);
    convex_hull.setDimension (3);
    convex_hull.setComputeAreaVolume (true);
    pcl::PolygonMesh mesh_out;
    convex_hull.reconstruct (mesh_out);
    std::cout << "CHull Volume:" << convex_hull.getTotalVolume () << std::endl;
    vis.addPolygonMesh (mesh_out, "camera hull", v3);

    //visualize graph using connected lines
    int e = 0;
    for (size_t i = 0; i < clouds_.size (); i++)
    {
      for (size_t j = (i + 1); j < clouds_.size (); j++)
      {
        if ((boost::edge (j, i, ST_graph_).second) || (boost::edge (i, j, ST_graph_).second))
        {
          std::stringstream line_name;
          line_name << "line_" << e;
          pcl::PointXYZ p1, p2;
          p1 = camera_positions->points[i];
          p2 = camera_positions->points[j];
          vis.addLine<pcl::PointXYZ, pcl::PointXYZ> (p1, p2, line_name.str (), v2);
          e++;
        }
      }
    }

    //visualize LR graph...
    {
      boost::property_map<Graph, boost::edge_weight_t>::type EdgeWeightMap = get (boost::edge_weight_t (), LR_graph_);
      int e = 0;
      for (size_t i = 0; i < clouds_.size (); i++)
      {
        for (size_t j = (i + 1); j < clouds_.size (); j++)
        {
          if ((boost::edge (j, i, LR_graph_).second) || (boost::edge (i, j, LR_graph_).second))
          {
            std::stringstream line_name;
            line_name << "line_whole_graph_" << e;
            pcl::PointXYZ p1, p2;
            p1 = camera_positions->points[i];
            p2 = camera_positions->points[j];
            //vis.addLine<pcl::PointXYZ, pcl::PointXYZ> (p1, p2, line_name.str (), v3);

            pcl::PointXYZ mid_point;
            mid_point.getVector3fMap () = (p1.getVector3fMap () + p2.getVector3fMap ()) / 2.f;

            {
              std::stringstream camera_name, edge_weight;
              camera_name << "w_" << e << std::endl;
              std::string text = boost::lexical_cast<std::string> ( get (EdgeWeightMap, boost::edge (i, j, LR_graph_).first).second);
              //vis.addText3D (text, mid_point, 0.005, 0.0, 1.0, 0.0, camera_name.str (), v3 + 1);
            }
            e++;
          }
        }
      }
    }
  }

  vis.spin ();

  {
    typename pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setMeanK (50);
    sor.setStddevMulThresh (2);
    sor.setInputCloud (big_cloud);
    sor.filter (*filtered);
    float leaf = 0.001f;
    {
      pcl::VoxelGrid<PointT> sor;
      sor.setLeafSize (leaf,leaf,leaf);
      sor.setInputCloud (filtered);
      sor.setDownsampleAllData(true);
      sor.filter (*big_cloud);
    }

    pcl::visualization::PCLVisualizer vis("filtered");

    float rgb_m;
    bool exists_m;

    typedef pcl::PointCloud<PointT> CloudM;
    typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

    pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (big_cloud->points[0], "rgb", exists_m, rgb_m));
    if (exists_m)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::copyPointCloud(*big_cloud,*cloud_rgb);
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (cloud_rgb);
      vis.addPointCloud<pcl::PointXYZRGB> (cloud_rgb, handler_rgb);
    }
    else
    {
      pcl::visualization::PointCloudColorHandlerRandom<PointT> handler_rgb (big_cloud);
      vis.addPointCloud<PointT> (big_cloud, handler_rgb);
    }

    vis.spin();
  }
}

template <typename PointT>
void
faat_pcl::object_modelling::ggo<PointT>::readGraph (std::string & directory)
{
  bf::path dir = directory;
  if (!bf::exists (dir))
  {
    PCL_ERROR("Path does not exist...\n");
    return;
  }

  std::stringstream graph_file;
  graph_file << directory << "/graph.txt";

  std::ifstream myfile;

  int max_id = -1;
  myfile.open (graph_file.str ().c_str ());
  std::string line1, line2;
  if (myfile.is_open ())
  {
    while (myfile.good ())
    {
      std::getline (myfile, line1);
      std::vector<std::string> strs1;
      boost::split (strs1, line1, boost::is_any_of (" "));
      if (strs1.size () == 3)
      {
        if (atoi (strs1[1].c_str ()) > max_id)
        {
          max_id = atoi (strs1[1].c_str ());
        }
      }
    }
  }

  clouds_.resize (max_id + 1);
  pairwise_poses_.resize (max_id + 1);
  pairwise_registration_.resize (max_id + 1);
  range_images_.clear();

  std::cout << "MAX_ID:" << max_id << std::endl;
  for (int i = 0; i <= max_id; i++)
  {
    clouds_[i].reset (new pcl::PointCloud<PointT>);
    std::stringstream file_name;
    file_name << directory << "/" << i << ".pcd";
    pcl::io::loadPCDFile (file_name.str (), *clouds_[i]);

    std::stringstream range_image_ss;
    range_image_ss << directory << "/range_image_" << i << ".pcd";
    bf::path ri_path = range_image_ss.str();
    if(bf::exists(ri_path))
    {
      std::cout << "Range image found, pushing..." << std::endl;
      PointCloudPtr range_image(new pcl::PointCloud<PointT>);
      pcl::io::loadPCDFile (range_image_ss.str (), *range_image);
      range_images_.push_back(range_image);
    }

    pairwise_poses_[i].resize (max_id + 1);
    pairwise_registration_[i].resize (max_id + 1);
  }

  //range_images_.clear();
  if(range_images_.size() != clouds_.size())
  {
    range_images_.clear();
    float min_fl = std::numeric_limits<float>::max ();
    float max_fl = -1;
    int cx, cy;
    cx = cy = 150;
    for (size_t i = 0; i < clouds_.size (); i++)
    {
      faat_pcl::registration::VisibilityReasoning<PointT> vr (0, 0, 0);
      float fl = vr.computeFocalLength (cx, cy, clouds_[i]);
      std::cout << "Focal length:" << fl << std::endl;
      if (fl > max_fl)
        max_fl = fl;

      if (fl < min_fl)
        min_fl = fl;
    }

    std::cout << "Min Focal length:" << min_fl << std::endl;
    std::cout << "Max Focal length:" << max_fl << std::endl;

    bool kinect_data = false;
    for (size_t i = 0; i < clouds_.size (); i++)
    {
      if(clouds_[i]->isOrganized())
      {
        PointCloudPtr range_image(new pcl::PointCloud<PointT>(*clouds_[i]));
        range_images_.push_back (range_image);
        kinect_data = true;

        //remove Nans
        std::vector<int> indices_p;
        pcl::removeNaNFromPointCloud(*clouds_[i], *clouds_[i], indices_p);
      }
      else
      {
        faat_pcl::registration::VisibilityReasoning<PointT> vr (0, 0, 0);
        PointCloudPtr range_image;
        vr.computeRangeImage (cx, cy, min_fl, clouds_[i], range_image);
        range_images_.push_back (range_image);
      }
    }

    if(kinect_data)
    {
      focal_length_ = 525.f;
      cx_ = 640.f;
      cy_ = 480.f;
    }
    else
    {
      focal_length_ = min_fl;
      cx_ = static_cast<float> (cx);
      cy_ = static_cast<float> (cy);
    }
  }
  else
  {
    //is kinect data => TODO: It wont always be kinect data, fix this (read file with values)
    focal_length_ = 525.f;
    cx_ = 640.f;
    cy_ = 480.f;
    std::cout << "Using saved range images..." << range_images_.size() << std::endl;
  }

  myfile.close ();
  myfile.open (graph_file.str ().c_str ());

  if (myfile.is_open ())
  {
    while (myfile.good ())
    {
      std::vector<std::string> strs1, strs2;
      std::getline (myfile, line1);
      boost::split (strs1, line1, boost::is_any_of (" "));
      if(strs1.size() != 3)
        continue;

      for(int k=0; k < atoi(strs1[2].c_str()); k++)
      {
        std::getline (myfile, line2);
        boost::split (strs2, line2, boost::is_any_of (" "));
        Eigen::Matrix4f matrix;
        for (int i = 0; i < 16; i++)
        {
          matrix (i / 4, i % 4) = static_cast<float> (atof (strs2[i].c_str ()));
        }

        pairwise_poses_[atoi (strs1[0].c_str ())][atoi (strs1[1].c_str ())].push_back (matrix);
      }
    }
    myfile.close ();
  }

  //USE only the best one, there is a problem with multiple
  for(size_t i=0; i < clouds_.size(); i++)
  {
    for(size_t j=i; j < clouds_.size(); j++)
    {
      pairwise_poses_[i][j].resize(std::min(max_poses_, static_cast<int>(pairwise_poses_[i][j].size())));
    }
  }
}

template <typename PointT>
void
faat_pcl::object_modelling::ggo<PointT>::visualizePairWiseAlignment ()
{
  pcl::visualization::PCLVisualizer vis ("");
  int v1,v2,v3;
  vis.createViewPort(0,0,0.33,1,v1);
  vis.createViewPort(0.33,0,0.66,1,v2);
  vis.createViewPort(0.66,0,1,1,v3);

  for (size_t i = 0; i < clouds_.size (); i++)
  {
    std::stringstream cloud_name;
    cloud_name << "cloud_" << i;

    /*pcl::visualization::PointCloudColorHandlerCustom<PointT> handler_rgb (clouds_[i], 0, 0, 255);
    vis.addPointCloud<PointT> (clouds_[i], handler_rgb, cloud_name.str ());*/

    PointCloudPtr pp = clouds_[i];
    float rgb_m;
    bool exists_m;

    typedef pcl::PointCloud<PointT> CloudM;
    typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

    pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (clouds_[i]->points[0], "rgb", exists_m, rgb_m));
    if (exists_m)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::copyPointCloud(*pp,*cloud_rgb);
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (cloud_rgb);
      vis.addPointCloud<pcl::PointXYZRGB> (cloud_rgb, handler_rgb, cloud_name.str (), v1);
      cloud_name << "_comb";
      vis.addPointCloud<pcl::PointXYZRGB> (cloud_rgb, handler_rgb, cloud_name.str (), v3);
    }
    else
    {
      pcl::visualization::PointCloudColorHandlerRandom<PointT> handler_rgb (pp);
      vis.addPointCloud<PointT> (pp, handler_rgb, cloud_name.str (), v1);
      cloud_name << "_comb";
      vis.addPointCloud<PointT> (pp, handler_rgb, cloud_name.str (), v3);
    }

    for (size_t j = (i + 1); j < clouds_.size (); j++)
    {
      std::stringstream cloud_name;
      std::stringstream cloud_name_2;
      cloud_name << "cloud_" << j;
      cloud_name_2 << cloud_name.str();

      std::cout << "Number of transformations for " << i << " to " << j << ": " << pairwise_poses_[i][j].size () << std::endl;
      for (size_t kk = 0; kk < pairwise_poses_[i][j].size (); kk++)
      {
        if (pairwise_poses_[i][j][kk] == Eigen::Matrix4f::Zero ())
          continue;

        pair_wise_registration pw_reg = pairwise_registration_[i][j][kk];
        std::cout << pw_reg.fsv_fraction_ << " " << pw_reg.osv_fraction_ << " " << pw_reg.color_error_ << " " << pw_reg.overlap_factor_ << std::endl;

        //if (pw_reg.fsv_fraction_ > 0.05) // || pw_reg.osv_fraction_ > 0.15)
        //continue;

        PointCloudPtr pp (new pcl::PointCloud<PointT>);
        pcl::transformPointCloud (*clouds_[j], *pp, pairwise_poses_[i][j][kk]);

        float rgb_m;
        bool exists_m;

        typedef pcl::PointCloud<PointT> CloudM;
        typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

        pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (clouds_[i]->points[0], "rgb", exists_m, rgb_m));
        if (exists_m)
        {
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
          pcl::copyPointCloud(*pp,*cloud_rgb);
          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (cloud_rgb);
          vis.addPointCloud<pcl::PointXYZRGB> (cloud_rgb, handler_rgb, cloud_name.str (), v2);
          cloud_name << "_comb";
          vis.addPointCloud<pcl::PointXYZRGB> (cloud_rgb, handler_rgb, cloud_name.str (), v3);
        }
        else
        {
          pcl::visualization::PointCloudColorHandlerRandom<PointT> handler_rgb (pp);
          vis.addPointCloud<PointT> (pp, handler_rgb, cloud_name.str (), v2);
          cloud_name << "_comb";
          vis.addPointCloud<PointT> (pp, handler_rgb, cloud_name.str (), v3);
        }

        //pcl::visualization::PointCloudColorHandlerCustom<PointT> handler_rgb (pp, 255, 0, 0);
        //vis.addPointCloud<PointT> (pp, handler_rgb, cloud_name.str ());

        //pcl::visualization::PointCloudColorHandlerRGBField<PointT> handler_rgb (pp);
        //vis.addPointCloud<PointT> (pp, handler_rgb, cloud_name.str ());
        vis.spin ();
        vis.removePointCloud (cloud_name.str ());
        vis.removeAllPointClouds(v2); // (cloud_name.str ());
      }
    }

    vis.removeAllPointClouds(); // (cloud_name.str ());

  }
}

template <typename PointT>
void
faat_pcl::object_modelling::ggo<PointT>::computeFSVFraction ()
{
  std::vector< boost::shared_ptr<pcl::octree::OctreePointCloudSearch<PointT> > > octrees;
  octrees.resize(clouds_.size());

  std::vector< PointCloudPtr > voxel_grided_clouds_;
  voxel_grided_clouds_.resize(clouds_.size());

  size_t i = 0;
  octrees[i].reset(new pcl::octree::OctreePointCloudSearch<PointT> (0.003));
  octrees[i]->setInputCloud (clouds_[i]);
  octrees[i]->addPointsFromInputCloud ();

  voxel_grided_clouds_[i].reset(new pcl::PointCloud<PointT>);
  float leaf=0.005f;
  Eigen::Vector4f min_p, max_p;
  faat_pcl::registration::UniformSamplingSharedVoxelGrid<PointT> keypoint_extractor;
  keypoint_extractor.setInputCloud (clouds_[i]);
  keypoint_extractor.setRadiusSearch(leaf);
  pcl::PointCloud<int> keypoints;
  keypoint_extractor.compute (keypoints);
  pcl::copyPointCloud(*clouds_[i], keypoints.points, *voxel_grided_clouds_[i]);
  keypoint_extractor.getVoxelGridValues(min_p,max_p);

#pragma omp parallel for num_threads(8)
  for (size_t i = 1; i < clouds_.size (); i++)
  {
    octrees[i].reset(new pcl::octree::OctreePointCloudSearch<PointT> (0.003));
    octrees[i]->setInputCloud (clouds_[i]);
    octrees[i]->addPointsFromInputCloud ();

    voxel_grided_clouds_[i].reset(new pcl::PointCloud<PointT>);
    faat_pcl::registration::UniformSamplingSharedVoxelGrid<PointT> keypoint_extractor;
    keypoint_extractor.setInputCloud (clouds_[i]);
    keypoint_extractor.setVoxelGridValues(min_p, max_p);
    keypoint_extractor.setRadiusSearch(leaf);
    pcl::PointCloud<int> keypoints;
    keypoint_extractor.compute (keypoints);
    pcl::copyPointCloud(*clouds_[i], keypoints.points, *voxel_grided_clouds_[i]);
    /*{
      pcl::VoxelGrid<PointT> sor;
      sor.setInputCloud (clouds_[i]);
      sor.setLeafSize (leaf,leaf,leaf);
      sor.setDownsampleAllData(true);
      sor.filter (*voxel_grided_clouds_[i]);
    }*/
  }

  std::vector<int> pointIdxNKNSearch;
  std::vector<float> pointNKNSquaredDistance;
  float inlier = 0.005f;
  float sigma = 15.f;
  float sigma_y = 128.f;
  sigma_y *= sigma_y;
  sigma *= sigma;

  for (size_t i = 0; i < clouds_.size (); i++)
  {
    for (size_t j = (i + 1); j < clouds_.size (); j++)
    {
      std::cout << "Number of transformations for " << i << " to " << j << ": " << pairwise_poses_[i][j].size () << std::endl;
      pairwise_registration_[i][j].resize(pairwise_poses_[i][j].size ());
//#pragma omp parallel for num_threads(8)
      for (size_t kk = 0; kk < pairwise_poses_[i][j].size (); kk++)
      {
        if (pairwise_poses_[i][j][kk] == Eigen::Matrix4f::Zero ())
          continue;

        faat_pcl::registration::VisibilityReasoning<PointT> vr (focal_length_, cx_, cy_);
        vr.setThresholdTSS(0.01f);
        float fsv_ij = vr.computeFSV (range_images_[i], clouds_[j], pairwise_poses_[i][j][kk]);
        float fsv_ji = vr.computeFSV (range_images_[j], clouds_[i], pairwise_poses_[i][j][kk].inverse ());

        float osv_ij = vr.computeOSV (range_images_[i], clouds_[j], pairwise_poses_[i][j][kk]);
        float osv_ji = vr.computeOSV (range_images_[j], clouds_[i], pairwise_poses_[i][j][kk].inverse ());

        //compute overlap
        int overlap_i_j = 0;
        int overlap_j_i = 0;
        float color_weight_i_j = 0.f;
        float color_weight_j_i = 0.f;
        float rgb_m, rgb_s;
        bool exists_m;
        bool exists_s;

        typedef pcl::PointCloud<PointT> CloudM;
        typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

        {
          PointCloudPtr trans_cloud(new PointCloud);
          pcl::transformPointCloud(*voxel_grided_clouds_[j], *trans_cloud, pairwise_poses_[i][j][kk]);
          std::cout << pairwise_poses_[i][j][kk] << std::endl;

          for (size_t kkk = 0; kkk < trans_cloud->points.size (); kkk++)
          {
            if (octrees[i]->nearestKSearch (trans_cloud->points[kkk], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
              float d = sqrt (pointNKNSquaredDistance[0]);
              if (d < inlier)
              {
                overlap_i_j++;

                pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (
                    trans_cloud->points[kkk],"rgb", exists_m, rgb_m));

                pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (
                    clouds_[i]->points[pointIdxNKNSearch[0]],"rgb", exists_s, rgb_s));

                if (exists_m && exists_s)
                {
                  uint32_t rgb = *reinterpret_cast<int*> (&rgb_m);
                  uint8_t rm = (rgb >> 16) & 0x0000ff;
                  uint8_t gm = (rgb >> 8) & 0x0000ff;
                  uint8_t bm = (rgb) & 0x0000ff;

                  rgb = *reinterpret_cast<int*> (&rgb_s);
                  uint8_t rs = (rgb >> 16) & 0x0000ff;
                  uint8_t gs = (rgb >> 8) & 0x0000ff;
                  uint8_t bs = (rgb) & 0x0000ff;

                  float ym = 0.257f * rm + 0.504f * gm + 0.098f * bm + 16; //between 16 and 235
                  float um = -(0.148f * rm) - (0.291f * gm) + (0.439f * bm) + 128;
                  float vm = (0.439f * rm) - (0.368f * gm) - (0.071f * bm) + 128;

                  float ys = 0.257f * rs + 0.504f * gs + 0.098f * bs + 16;
                  float us = -(0.148f * rs) - (0.291f * gs) + (0.439f * bs) + 128;
                  float vs = (0.439f * rs) - (0.368f * gs) - (0.071f * bs) + 128;

                  float color_weight = std::exp ((-0.5f * (ym - ys) * (ym - ys)) / (sigma_y));
                  color_weight *= std::exp ((-0.5f * (um - us) * (um - us)) / (sigma));
                  color_weight *= std::exp ((-0.5f * (vm - vs) * (vm - vs)) / (sigma));
                  color_weight_i_j += color_weight;
                }
              }
            }
          }
        }

        {
          PointCloudPtr trans_cloud(new PointCloud);
          Eigen::Matrix4f inv = pairwise_poses_[i][j][kk].inverse();
          pcl::transformPointCloud(*voxel_grided_clouds_[i], *trans_cloud, inv);
          for (size_t kkk = 0; kkk < trans_cloud->points.size (); kkk++)
          {
            if (octrees[j]->nearestKSearch (trans_cloud->points[kkk], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
              float d = sqrt (pointNKNSquaredDistance[0]);
              if (d < inlier)
              {
                overlap_j_i++;

                pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (
                                    trans_cloud->points[kkk],"rgb", exists_m, rgb_m));

                pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (
                    clouds_[j]->points[pointIdxNKNSearch[0]],"rgb", exists_s, rgb_s));

                if (exists_m && exists_s)
                {
                  uint32_t rgb = *reinterpret_cast<int*> (&rgb_m);
                  uint8_t rm = (rgb >> 16) & 0x0000ff;
                  uint8_t gm = (rgb >> 8) & 0x0000ff;
                  uint8_t bm = (rgb) & 0x0000ff;

                  rgb = *reinterpret_cast<int*> (&rgb_s);
                  uint8_t rs = (rgb >> 16) & 0x0000ff;
                  uint8_t gs = (rgb >> 8) & 0x0000ff;
                  uint8_t bs = (rgb) & 0x0000ff;

                  float ym = 0.257f * rm + 0.504f * gm + 0.098f * bm + 16; //between 16 and 235
                  float um = -(0.148f * rm) - (0.291f * gm) + (0.439f * bm) + 128;
                  float vm = (0.439f * rm) - (0.368f * gm) - (0.071f * bm) + 128;

                  float ys = 0.257f * rs + 0.504f * gs + 0.098f * bs + 16;
                  float us = -(0.148f * rs) - (0.291f * gs) + (0.439f * bs) + 128;
                  float vs = (0.439f * rs) - (0.368f * gs) - (0.071f * bs) + 128;

                  float color_weight = std::exp ((-0.5f * (ym - ys) * (ym - ys)) / (sigma_y));
                  color_weight *= std::exp ((-0.5f * (um - us) * (um - us)) / (sigma));
                  color_weight *= std::exp ((-0.5f * (vm - vs) * (vm - vs)) / (sigma));
                  color_weight_j_i += color_weight;
                }
              }
            }
          }
        }

        float ov_measure_1 = overlap_i_j / static_cast<float>(voxel_grided_clouds_[j]->points.size());
        float ov_measure_2 = overlap_j_i / static_cast<float>(voxel_grided_clouds_[i]->points.size());
        float cw_1 = color_weight_i_j / static_cast<float>(overlap_i_j);
        float cw_2 = color_weight_j_i / static_cast<float>(overlap_j_i);
        pair_wise_registration pw_reg;
        pw_reg.fsv_fraction_ = std::max (fsv_ij, fsv_ji); //this will be minimized, take worse
        pw_reg.osv_fraction_ = std::max (osv_ij, osv_ji); //this will be minimized, take worse
        pw_reg.overlap_factor_ = std::max(ov_measure_1, ov_measure_2); //this will be maximized, take best
        pw_reg.color_error_ = std::max(cw_1,cw_2); //this will be maximized, take best
        pairwise_registration_[i][j][kk] =  (pw_reg);
      }
    }
  }
}

template class PCL_EXPORTS faat_pcl::object_modelling::ggo<pcl::PointXYZ>;
template class PCL_EXPORTS faat_pcl::object_modelling::ggo<pcl::PointXYZRGB>;
