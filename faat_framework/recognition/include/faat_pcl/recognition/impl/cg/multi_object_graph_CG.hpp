/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *  
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef FAAT_PCL_RECOGNITION_MULTI_OBJECT_GRAPH_CG_HPP
#define FAAT_PCL_RECOGNITION_MULTI_OBJECT_GRAPH_CG_HPP

#include <faat_pcl/recognition/cg/multi_object_graph_CG.h>
#include <pcl/point_cloud.h>
#include <pcl/common/angles.h>
#include <pcl/common/centroid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/biconnected_components.hpp>
#include <boost/graph/connected_components.hpp>

bool
mo_gcGraphCorrespSorter (pcl::Correspondence i, pcl::Correspondence j)
{
  return (i.distance < j.distance);
}

template<typename PointModelT, typename PointSceneT>
void
faat_pcl::MultiObjectGraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::visualizeGraph(GraphGGCG & g, std::string title)
{
    pcl::visualization::PCLVisualizer vis(title);
    int v1, v2, v3;
    vis.createViewPort(0,0,0.33,1.0,v1);
    vis.createViewPort(0.33,0,0.66,1.0,v2);
    vis.createViewPort(0.66,0,1.0,1.0,v3);
    vis.setBackgroundColor(255,255,255);

    int max_label = model_scene_corrs_.size();
    std::vector<uint32_t> label_colors;
    label_colors.reserve (max_label);
    srand (static_cast<unsigned int> (time (0)));
    while (label_colors.size () <= max_label )
    {
        uint8_t r = static_cast<uint8_t>( (rand () % 256));
        uint8_t g = static_cast<uint8_t>( (rand () % 256));
        uint8_t b = static_cast<uint8_t>( (rand () % 256));
        label_colors.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
    }

    typename boost::graph_traits<GraphGGCG>::vertex_iterator vertexIt, vertexEnd;
    std::vector<typename boost::graph_traits<GraphGGCG>::vertex_descriptor> to_be_removed;
    boost::tie (vertexIt, vertexEnd) = boost::vertices (g);
    int s=0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud, full_scene, scene_cloud, scene_cloud_non_zero;

    model_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    scene_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    scene_cloud_non_zero.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (; vertexIt != vertexEnd; ++vertexIt, s++)
    {

      int model_id = vertex_idx_to_model_correspondence_[s].first;
      int correspondence_id = vertex_idx_to_model_correspondence_[s].second;

      pcl::Correspondence corresp = model_scene_corrs_[model_id]->at(correspondence_id);



      uint32_t rgb = *reinterpret_cast<int*> (&label_colors[model_id]);
      unsigned char rs = (rgb >> 16) & 0x0000ff;
      unsigned char gs = (rgb >> 8) & 0x0000ff;
      unsigned char bs = (rgb) & 0x0000ff;


      float r=rs; float green=gs; float b=bs;

      /*float radius = 0.005f;

      if(boost::out_degree(*vertexIt, g) < (gc_threshold_ - 1))
      {
        r = 0;
        green = 125;
        radius = 0.0025f;
      }*/

      int model_index_k = corresp.index_query;
      int scene_index_k = corresp.index_match;
      pcl::PointXYZ mp, sp;
      mp.getVector3fMap() = model_clouds_[model_id]->at (model_index_k).getVector3fMap();
      sp.getVector3fMap()  = scene_->at (scene_index_k).getVector3fMap();

      pcl::PointXYZRGB mpp, spp;
      mpp.getVector3fMap() = mp.getVector3fMap();
      spp.getVector3fMap() = sp.getVector3fMap();
      mpp.r = r; mpp.g = green; mpp.b = b;
      spp.r = r; spp.g = green; spp.b = b;
      model_cloud->points.push_back(mpp);
      scene_cloud->points.push_back(spp);
      scene_cloud_non_zero->points.push_back(spp);
    }

    Eigen::Vector4f centroid_model, centroid_scene;

    pcl::compute3DCentroid(*model_cloud, centroid_model);
    pcl::compute3DCentroid(*scene_cloud, centroid_scene);
    Eigen::Vector4f demean = centroid_model + (centroid_scene * -1.f);
    pcl::demeanPointCloud(*model_cloud, demean, *model_cloud);

    vis.addPointCloud(full_scene_, "full_scene", v3);

    vis.addPointCloud(model_cloud, "model", v1);
    vis.addPointCloud(scene_cloud, "scene", v2);
    vis.addPointCloud(scene_cloud, "scene_v3", v3);

    vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 9, "scene_v3");
    vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 9, "scene");

    //iterate over edges and draw lines
    typedef typename GraphGGCG::edge_iterator EdgeIterator;
    std::pair<EdgeIterator, EdgeIterator> edges = boost::edges(g);
    EdgeIterator edge;
    int edge_skip=5;
    int n_edge = 0;

    edge_skip = (boost::num_edges(g) / 2000);

    n_edge = 0;

    std::cout << boost::num_edges(g) << " " << edge_skip << std::endl;

    for (edge = edges.first; edge != edges.second; edge++, n_edge++) {
      if(edge_skip != 0 && (n_edge % edge_skip) != 0)
        continue;

      typename boost::graph_traits<GraphGGCG>::vertex_descriptor s, t;
      s = boost::source(*edge, g);
      t = boost::target(*edge, g);

      int model_id_source = vertex_idx_to_model_correspondence_[s].first;
      int correspondence_id_source = vertex_idx_to_model_correspondence_[s].second;

      int model_id_target = vertex_idx_to_model_correspondence_[t].first;
      int correspondence_id_target = vertex_idx_to_model_correspondence_[t].second;

      pcl::Correspondence corresp_source = model_scene_corrs_[model_id_source]->at(correspondence_id_source);
      pcl::Correspondence corresp_target = model_scene_corrs_[model_id_target]->at(correspondence_id_target);

      int scene_index_k = corresp_source.index_match;
      int scene_index_j = corresp_target.index_match;

      pcl::PointXYZ p1,p2;
      p1.getVector3fMap() = scene_->at (scene_index_j).getVector3fMap();
      p2.getVector3fMap()  = scene_->at (scene_index_k).getVector3fMap();

      std::stringstream name;
      name << "line_" << n_edge;
      vis.addLine(p1,p2, name.str(), v2);
    }

    vis.spin();
}

template<typename PointModelT, typename PointSceneT>
void
faat_pcl::MultiObjectGraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::cluster(std::vector<pcl::Correspondences> &model_instances)
{

    model_instances.clear ();
    found_transformations_.clear ();

    //temp copy of scene cloud with the type cast to ModelT in order to use Ransac
    PointCloudPtr temp_scene_cloud_ptr (new PointCloud ());
    pcl::copyPointCloud<PointSceneT, PointModelT> (*scene_, *temp_scene_cloud_ptr);

    size_t graph_size = 0;
    for(size_t i=0; i < model_scene_corrs_.size(); i++)
    {
        graph_size += model_scene_corrs_[i]->size();
    }

    dist_for_cluster_factor_ = 0; //ATTENTION
    float min_dist_for_cluster = gc_size_ * dist_for_cluster_factor_;

    GraphGGCG correspondence_graph (graph_size);
    vertex_idx_to_model_correspondence_.resize(graph_size);

    assert(model_scene_corrs_.size() == models_normals_.size());
    assert(model_scene_corrs_.size() == model_clouds_.size());

    int graph_start = 0;
    for(size_t i=0; i < model_scene_corrs_.size(); i++)
    {

        std::cout << "Processing model" << i << std::endl;

        pcl::CorrespondencesPtr sorted_corrs (new pcl::Correspondences (*model_scene_corrs_[i]));
        //std::sort (sorted_corrs->begin (), sorted_corrs->end (), mo_gcGraphCorrespSorter);

        for (size_t k = 0; k < sorted_corrs->size (); ++k)
        {

            int scene_index_k = sorted_corrs->at (k).index_match;
            int model_index_k = sorted_corrs->at (k).index_query;

            //std::cout << "i:" << i << " " << models_normals_.size() << std::endl;
            //std::cout << models_normals_[i]->points.size() << " " << model_index_k << std::endl;

            assert(model_index_k < model_clouds_[i]->points.size());
            assert(scene_index_k < scene_->points.size());

            const Eigen::Vector3f& scene_point_k = scene_->at (scene_index_k).getVector3fMap ();
            const Eigen::Vector3f& model_point_k = model_clouds_[i]->at (model_index_k).getVector3fMap ();

            const Eigen::Vector3f& scene_normal_k = scene_normals_->at (scene_index_k).getNormalVector3fMap ();
            const Eigen::Vector3f& model_normal_k = models_normals_[i]->at (model_index_k).getNormalVector3fMap ();

            vertex_idx_to_model_correspondence_[graph_start + k] = std::make_pair(i, k);

            for (size_t j = (k + 1); j < sorted_corrs->size (); ++j)
            {
                int scene_index_j = sorted_corrs->at (j).index_match;
                int model_index_j = sorted_corrs->at (j).index_query;

                //same scene or model point constraint
                if(scene_index_j == scene_index_k || model_index_j == model_index_k)
                    continue;

                assert(model_index_j < model_clouds_[i]->points.size());
                assert(scene_index_j < scene_->points.size());

                const Eigen::Vector3f& scene_point_j = scene_->at (scene_index_j).getVector3fMap ();
                const Eigen::Vector3f& model_point_j = model_clouds_[i]->at (model_index_j).getVector3fMap ();

                const Eigen::Vector3f& scene_normal_j = scene_normals_->at (scene_index_j).getNormalVector3fMap ();
                const Eigen::Vector3f& model_normal_j = models_normals_[i]->at (model_index_j).getNormalVector3fMap ();

                Eigen::Vector3f dist_trg = model_point_k - model_point_j;
                Eigen::Vector3f dist_ref = scene_point_k - scene_point_j;

                //minimum distance constraint
                if ((dist_trg.norm () < min_dist_for_cluster) || (dist_ref.norm () < min_dist_for_cluster))
                    continue;

                assert((dist_trg.norm () >= min_dist_for_cluster) && (dist_ref.norm () >= min_dist_for_cluster));

                double distance = fabs (dist_trg.norm () - dist_ref.norm());

                double dot_distance = 0;
                if (pcl_isnan(scene_normal_k.dot (scene_normal_j)) || pcl_isnan(model_normal_k.dot (model_normal_j)))
                    dot_distance = 0.f;
                else
                {
                    float dot_model = model_normal_k.dot (model_normal_j);
                    dot_distance = std::abs (scene_normal_k.dot (scene_normal_j) - dot_model);
                }

                //Model normals should be consistently oriented! otherwise reject!
                float dot_distance_model = model_normal_k.dot (model_normal_j);
                if(dot_distance_model < -0.1f)
                    continue;

                //gc constraint and dot_product constraint!
                if ((distance < gc_size_) && (dot_distance <= thres_dot_distance_))
                {
                    boost::add_edge (graph_start + k, graph_start + j, correspondence_graph);
                }
            }
        }

        graph_start += static_cast<int>(sorted_corrs->size());
    }

    std::cout << "edges:" << num_edges (correspondence_graph) << std::endl;
    std::cout << "number of nodes:" << boost::num_vertices(correspondence_graph) << std::endl;
    visualizeGraph(correspondence_graph);

    /*{
        double ms = time_watch.getTime();
        std::cout << "Built graph:" << ms << std::endl;
    }*/

    //std::cout << "edges before cleaning:" << num_edges (correspondence_graph) << std::endl;
    /*cleanGraph (correspondence_graph, gc_threshold_);
    std::cout << "edges:" << num_edges (correspondence_graph) << std::endl;*/

    /*int num_edges_start;
    do
    {
      num_edges_start = num_edges (correspondence_graph);
      //pcl::ScopeTime t ("checking for additional edges...\n");
      //std::cout << "edges:" << num_edges (correspondence_graph) << std::endl;
      for (size_t k = 0; k < model_scene_corrs_->size (); ++k)
      {
        for (size_t j = (k + 1); j < model_scene_corrs_->size (); ++j)
        {
          if (boost::edge (k, j, correspondence_graph).second)
          {
            //check if there is another correspondence different than k and j, that fullfills the gc constraint with both k and j
            bool edge_available = false;
            for(size_t c=0; c < model_scene_corrs_->size(); c++)
            {
              if (c == j || c == k)
                continue;

              if (boost::edge (k, c, correspondence_graph).second && boost::edge (j, c, correspondence_graph).second)
              {
                edge_available = true;
                break;
              }
            }

            if (!edge_available)
              boost::remove_edge(k, j, correspondence_graph);
          }
        }
      }
      //std::cout << "edges:" << num_edges (correspondence_graph) << std::endl;
      cleanGraph(correspondence_graph, gc_threshold_);
      //std::cout << "edges:" << num_edges (correspondence_graph) << std::endl;
    } while (num_edges_start != num_edges (correspondence_graph));*/

    /*{
        double ms = time_watch.getTime();
        std::cout << "Built graph + clean graph:" << ms << std::endl;
    }*/

    /*boost::vector_property_map<int> components (boost::num_vertices (correspondence_graph));
    int n_cc = static_cast<int> (boost::connected_components (correspondence_graph, &components[0]));
    std::cout << "Number of connected components..." << n_cc << std::endl;*/

    typename boost::property_map < GraphGGCG, edge_component_t>::type components = get(edge_component, correspondence_graph);
    int n_cc = static_cast<int>(biconnected_components(correspondence_graph, components));

    std::cout << "Number of connected components:" << n_cc << std::endl;

    if(n_cc < 1)
      return;

    std::vector<int> model_instances_kept_indices;

    std::vector< std::set<int> > unique_vertices_per_cc;
    std::vector<int> cc_sizes;
    cc_sizes.resize (n_cc, 0);
    unique_vertices_per_cc.resize (n_cc);

    typename boost::graph_traits<GraphGGCG>::edge_iterator edgeIt, edgeEnd;
    boost::tie (edgeIt, edgeEnd) = edges (correspondence_graph);
    for (; edgeIt != edgeEnd; ++edgeIt)
    {
      int c = components[*edgeIt];
      unique_vertices_per_cc[c].insert(boost::source(*edgeIt, correspondence_graph));
      unique_vertices_per_cc[c].insert(boost::target(*edgeIt, correspondence_graph));
    }

    for(size_t i=0; i < unique_vertices_per_cc.size(); i++)
      cc_sizes[i] = unique_vertices_per_cc[i].size();

    int analyzed_ccs = 0;
    for (int c = 0; c < n_cc; c++)
    {
      //ignore if not enough vertices...
      int num_v_in_cc = cc_sizes[c];
      if (num_v_in_cc < gc_threshold_)
      {
        continue;
      }

      std::cout << "Component: " << c << std::endl;
      analyzed_ccs++;

      GraphGGCG connected_graph(correspondence_graph);

      typename boost::graph_traits<GraphGGCG>::edge_iterator edgeIt, edgeEnd;
      boost::tie (edgeIt, edgeEnd) = edges (connected_graph);
      for (; edgeIt != edgeEnd; ++edgeIt)
      {
        if (components[*edgeIt] != c)
        {
          boost::remove_edge(*edgeIt, connected_graph);
        }
      }

      visualizeGraph(connected_graph, "connected component");
    }
}

template<typename PointModelT, typename PointSceneT>
bool
faat_pcl::MultiObjectGraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::poseExists(Eigen::Matrix4f corr_rej_trans)
{
    bool found = false;
    Eigen::Vector3f trans = corr_rej_trans.block<3,1>(0,3);
    Eigen::Quaternionf quat(corr_rej_trans.block<3,3>(0,0));
    quat.normalize();
    Eigen::Quaternionf quat_conj = quat.conjugate();

    for(size_t t=0; t < found_transformations_.size(); t++)
    {
      Eigen::Vector3f trans_found = found_transformations_[t].block<3,1>(0,3);
      if((trans - trans_found).norm() < gc_size_)
      {
        found = true;
        break;

        Eigen::Quaternionf quat_found(found_transformations_[t].block<3,3>(0,0));
        quat_found.normalize();
        Eigen::Quaternionf quat_prod = quat_found * quat_conj;
        double angle = acos(quat_prod.z());
        if(std::abs(pcl::rad2deg(angle)) < 5.0)
        {
        }
      }
    }

    return found;
}

#endif // FAAT_PCL_RECOGNITION_MULTI_OBJECT_GRAPH_CG_HPP
