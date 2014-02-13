/*
 * GO3D.cpp
 *
 *  Created on: Oct 24, 2013
 *      Author: aitor
 */

#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <pcl/common/transforms.h>
#include <fstream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/common/angles.h>
#include <pcl/registration/icp.h>

#include <pcl/filters/passthrough.h>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/biconnected_components.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>

#include <faat_pcl/object_modelling/merge_sequence.h>
#include <faat_pcl/object_modelling/get_nice_model.h>
#include <limits>

struct IndexPoint
{
    int idx;
};

namespace bf = boost::filesystem;

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

class myEdge
{
public:

    Eigen::Matrix4f transformation;
    float edge_weight;
    int source_id, target_id;

    myEdge(float w) :
        edge_weight(w)
    {

    }

    myEdge() : edge_weight(0.f)
    {

    }

    bool operator<(const myEdge& e) const {
        if(edge_weight < e.edge_weight)
            return true;

        return false;
    }
};

namespace std {
template<>
struct numeric_limits<myEdge> {
    static myEdge max() { return myEdge(numeric_limits<float>::max()); }
};
}


typedef boost::property<boost::edge_weight_t, myEdge> EdgeWeightProperty;
typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, int, EdgeWeightProperty> Graph;
typedef boost::graph_traits < Graph >::vertex_descriptor Vertex;
typedef boost::graph_traits < Graph >::edge_descriptor Edge;
typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;

template<typename PointT>
inline void
createBigPointCloudRecursive (Graph & grph_final, typename pcl::PointCloud<PointT>::Ptr & big_cloud, Vertex start, Vertex coming_from,
                              Eigen::Matrix4f accum,
                              std::vector<typename pcl::PointCloud<PointT>::Ptr> & partial_models,
                              std::vector<Eigen::Matrix4f> & absolute_poses)
{
  if (boost::degree (start, grph_final) == 1)
  {
    //check if target is like coming_from
    boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
    for (boost::tie (ei, ei_end) = boost::out_edges (start, grph_final); ei != ei_end; ++ei)
    {
      if (target (*ei, grph_final) == coming_from)
        return;
    }
  }

  boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
  std::vector < boost::graph_traits<Graph>::out_edge_iterator > edges;
  for (boost::tie (ei, ei_end) = boost::out_edges (start, grph_final); ei != ei_end; ++ei)
  {

    if (target (*ei, grph_final) == coming_from)
    {
      continue;
    }

    edges.push_back (ei);
  }

  boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, grph_final);

  for (size_t i = 0; i < edges.size (); i++)
  {
    Eigen::Matrix4f internal_accum;
    Edge e = *edges[i];
    Vertex src = boost::source (e, grph_final);
    Vertex targ = boost::target (e, grph_final);
    Eigen::Matrix4f transform;
    myEdge my_e = weightmap[e];

    if ((int)src > (int)targ)
    {
      PCL_WARN("inverse\n");
      transform = my_e.transformation.inverse ();
    }
    else
    {
      PCL_WARN("normal\n");
      transform = my_e.transformation;
    }

    internal_accum = accum * transform;
    std::cout << internal_accum << std::endl;
    typename pcl::PointCloud<PointT>::Ptr trans (new pcl::PointCloud<PointT>);
    pcl::transformPointCloudWithNormals (*partial_models[targ], *trans, internal_accum);
    *big_cloud += *trans;
    absolute_poses[targ] = internal_accum;
    createBigPointCloudRecursive<PointT> (grph_final, big_cloud, targ, start, internal_accum, partial_models, absolute_poses);
  }
}

template<typename PointT>
void
createBigPointCloud (Graph & grph_final,
                     typename pcl::PointCloud<PointT>::Ptr & big_cloud,
                     std::vector<typename pcl::PointCloud<PointT>::Ptr> & partial_models,
                     std::vector<Eigen::Matrix4f> & absolute_poses)
{
  std::pair<vertex_iter, vertex_iter> vp;
  vp = vertices (grph_final);
  Eigen::Matrix4f accum;
  accum.setIdentity ();
  //*big_cloud += *grph_final[*vp.first].pScenePCl;
  absolute_poses[*vp.first] = accum;
  *big_cloud += *(partial_models[*vp.first]);
  createBigPointCloudRecursive<PointT> (grph_final, big_cloud, *vp.first, *vp.first, accum, partial_models, absolute_poses);
}

//./bin/merge_multiple -sequences /media/DATA/models/asus_box/seq1_aligned/,/media/DATA/models/asus_box/seq2_aligned/ -overlap 0.5 -inliers_threshold 0.01 -output_path /media/DATA/models/asus_box/merged_sequences -use_color 1

int
main (int argc, char ** argv)
{
    std::string output_path = "";
    float overlap = 0.6f;
    float inliers_threshold = 0.01f;
    bool use_color = false;
    bool visualize = false;

    std::string sequences;
    pcl::console::parse_argument (argc, argv, "-sequences", sequences);
    pcl::console::parse_argument (argc, argv, "-overlap", overlap);
    pcl::console::parse_argument (argc, argv, "-inliers_threshold", inliers_threshold);
    pcl::console::parse_argument (argc, argv, "-output_path", output_path);
    pcl::console::parse_argument (argc, argv, "-use_color", use_color);
    pcl::console::parse_argument (argc, argv, "-visualize", visualize);

    std::vector<std::string> sequence_paths;
    boost::split (sequence_paths, sequences, boost::is_any_of (","));

    std::cout << sequence_paths.size() << std::endl;

    //for each sequence, get nice models
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> partial_models_;
    std::vector<faat_pcl::modelling::Sequence<pcl::PointXYZRGB> > original_sequences;
    original_sequences.resize(sequence_paths.size());

    for(size_t i=0; i < sequence_paths.size(); i++)
    {
        faat_pcl::modelling::NiceModelFromSequence<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> nice_model_;
        nice_model_.setInputDir(sequence_paths[i]);
        nice_model_.readSequence();
        nice_model_.setVisualize(false);
        nice_model_.setLateralSigma(0.001f);
        nice_model_.compute();
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model = nice_model_.getModelCloud();
        partial_models_.push_back(model);
        nice_model_.getSequence(original_sequences[i]);
    }

    typedef pcl::PointXYZRGBNormal ModelPointT;

    pcl::visualization::PCLVisualizer vis ("registered cloud");
    int v1, v2, v3, v4;
    vis.createViewPort (0, 0, 0.25, 1, v1);
    vis.createViewPort (0.25, 0, 0.5, 1, v4);
    vis.createViewPort (0.5, 0, 0.75, 1, v2);
    vis.createViewPort (0.75, 0, 1, 1, v3);
    vis.setBackgroundColor(1,1,1);
    vis.addCoordinateSystem(0.1f);

    //create graph (nodes are the partial models, edges are 1 - ICP_weight and transformations going from model to target

    Graph G;
    for(size_t i=0; i < partial_models_.size(); i++)
    {
        boost::add_vertex((int)i, G);
    }

    for(size_t i=0; i < partial_models_.size(); i++)
    {
        pcl::PointCloud<ModelPointT>::Ptr target_model(new pcl::PointCloud<ModelPointT>(*partial_models_[i]));

        Eigen::Matrix4f center_transform;
        center_transform.setIdentity();
        center_transform(2,3) = 0.01f;

        /*Eigen::Vector4f centroid;
        centroid.setZero();
        centroid[2] = -0.01f;
        pcl::demeanPointCloud(*target_model, centroid, *target_model);*/

        pcl::transformPointCloudWithNormals(*target_model, *target_model, center_transform);

        for(size_t j=(i+1); j < partial_models_.size(); j++)
        {
            pcl::PointCloud<ModelPointT>::Ptr model_cloud(new pcl::PointCloud<ModelPointT>(*partial_models_[j]));

            pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler (model_cloud);
            vis.addPointCloud<ModelPointT> (model_cloud, handler, "big", v1);

            pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler_v4 (target_model);
            vis.addPointCloud<ModelPointT> (target_model, handler_v4, "big_target", v4);

            //vis.addPointCloudNormals<ModelPointT,ModelPointT> (model_cloud, model_cloud, 10, 0.01, "normals_big", v1);

            pcl::ConvexHull<ModelPointT> convex_hull;
            convex_hull.setInputCloud (model_cloud);
            convex_hull.setDimension (3);
            convex_hull.setComputeAreaVolume (false);

            pcl::PolygonMeshPtr mesh_out(new pcl::PolygonMesh);
            convex_hull.reconstruct (*mesh_out);

            vis.addPolygonMesh(*mesh_out, "hull", v2);

            {
                pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler (target_model);
                vis.addPointCloud<ModelPointT> (target_model, handler, "target", v3);
            }

            std::vector<std::pair<float, Eigen::Matrix4f> > res;
            faat_pcl::modelling::MergeSequences<pcl::PointXYZRGBNormal> merge_;
            merge_.setInlierThreshold(inliers_threshold);
            merge_.setOverlap(overlap);
            merge_.setInputSource(model_cloud);
            merge_.setInputTarget(target_model);
            merge_.setMaxIterations(50);
            merge_.setUseColor(use_color);
            merge_.compute(res);

            for(size_t k=0; k < std::min((int)res.size(), 1); k++)
            {
                vis.removePointCloud("model_cloud_trans");

                std::cout << k << " " << res[k].first << std::endl;
                pcl::PointCloud<ModelPointT>::Ptr Final(new pcl::PointCloud<ModelPointT>);
                pcl::transformPointCloudWithNormals(*model_cloud, *Final, res[k].second);

                pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler (Final);
                vis.addPointCloud<ModelPointT> (Final, handler, "model_cloud_trans", v3);
                if(visualize)
                    vis.spin();
                else
                    vis.spinOnce();
            }

            pcl::PointCloud<ModelPointT>::Ptr merged_cloud(new pcl::PointCloud<ModelPointT>(*target_model));
            pcl::PointCloud<ModelPointT>::Ptr Final(new pcl::PointCloud<ModelPointT>);

            std::cout << res[0].second << std::endl;

            pcl::transformPointCloudWithNormals(*model_cloud, *Final, res[0].second);
            *merged_cloud += *Final;
            pcl::io::savePCDFileBinary("merged_cloud.pcd", *merged_cloud);

            myEdge e;
            e.edge_weight = overlap - res[0].first;
            e.transformation = center_transform.inverse() * res[0].second;
            boost::add_edge ((int)j, (int)i, e, G);

            vis.removeAllPointClouds();
            vis.removeAllShapes();
        }
    }

    //minimum spanning tree
    boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, G);
    std::vector < boost::graph_traits<Graph>::vertex_descriptor > p (boost::num_vertices (G));
    boost::prim_minimum_spanning_tree (G, &p[0]);

    for (std::size_t i = 0; i != p.size (); ++i)
    {
        if (p[i] != i)
            std::cout << "parent[" << i << "] = " << p[i] << std::endl;
        else
            std::cout << "parent[" << i << "] = no parent" << std::endl;
    }

    Graph MST;
    for(size_t i=0; i < partial_models_.size(); i++)
        boost::add_vertex((int)i, MST);

    typedef typename Graph::edge_iterator EdgeIterator;
    std::pair<EdgeIterator, EdgeIterator> edges = boost::edges(G);
    EdgeIterator edge;

    for (edge = edges.first; edge != edges.second; edge++)
    {
      typename boost::graph_traits<Graph>::vertex_descriptor s, t;
      s = boost::source(*edge, G);
      t = boost::target(*edge, G);

      if(p[s] == t || p[t] == s)
      {
          //edge in prim
          boost::add_edge ((int)s, (int)t, weightmap[*edge], MST);
      }
    }

    std::cout << boost::num_edges(MST) << std::endl;
    std::vector<Eigen::Matrix4f> absolute_poses;
    absolute_poses.resize(boost::num_vertices(MST));

    pcl::PointCloud<ModelPointT>::Ptr merged_cloud(new pcl::PointCloud<ModelPointT>());
    createBigPointCloud<ModelPointT>(MST, merged_cloud, partial_models_, absolute_poses);

    {
        pcl::visualization::PCLVisualizer vis ("registered cloud");
        pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler (merged_cloud);
        vis.addPointCloud<ModelPointT> (merged_cloud, handler, "model_cloud_trans");
        if(visualize)
            vis.spin();
        else
            vis.spinOnce();
    }

    {

        bf::path aligned_output = output_path;
        if(!bf::exists(aligned_output))
        {
            bf::create_directory(aligned_output);
        }

        pcl::visualization::PCLVisualizer vis ("all togethr!");

        int k=0;
        for(size_t i=0; i < absolute_poses.size(); i++)
        {
            for(size_t j=0; j < original_sequences[i].original_clouds_.size(); j++, k++)
            {
                std::stringstream name;
                name << "cloud_" << i << "_" << j;

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*original_sequences[i].original_clouds_[j]));

                pcl::PassThrough<pcl::PointXYZRGB> pass_;
                pass_.setFilterLimits (0.f, 1);
                pass_.setFilterFieldName ("z");
                pass_.setInputCloud (cloud);
                pass_.setKeepOrganized (false);
                pass_.filter (*cloud);

                Eigen::Matrix4f trans = absolute_poses[i] * original_sequences[i].transforms_to_global_[j];
                pcl::transformPointCloud(*cloud, *cloud, trans);
                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (cloud);
                vis.addPointCloud<pcl::PointXYZRGB> (cloud, handler, name.str());

                {
                    std::stringstream temp;
                    temp << output_path << "/cloud_";
                    temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".pcd";
                    std::string scene_name;
                    temp >> scene_name;
                    std::cout << scene_name << std::endl;
                    pcl::io::savePCDFileBinary(scene_name, *original_sequences[i].original_clouds_[j]);
                }

                //write pose
                {
                    std::stringstream temp;
                    temp << output_path << "/pose_";
                    temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".txt";
                    std::string scene_name;
                    temp >> scene_name;
                    std::cout << scene_name << std::endl;
                    faat_pcl::utils::writeMatrixToFile(scene_name, trans);
                }

                std::vector<int> obj_indices_original = original_sequences[i].original_indices_[j];std::stringstream temp;
                temp << output_path << "/object_indices_";
                temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".pcd";
                std::string scene_name;
                temp >> scene_name;
                std::cout << scene_name << std::endl;
                pcl::PointCloud<IndexPoint> obj_indices_cloud;
                obj_indices_cloud.width = obj_indices_original.size();
                obj_indices_cloud.height = 1;
                obj_indices_cloud.points.resize(obj_indices_cloud.width);
                for(size_t kk=0; kk < obj_indices_original.size(); kk++)
                    obj_indices_cloud.points[kk].idx = obj_indices_original[kk];

                pcl::io::savePCDFileBinary(scene_name, obj_indices_cloud);

            }
        }

        vis.addCoordinateSystem(0.15f);
        if(visualize)
            vis.spin();
        else
            vis.spinOnce();
    }
    //merge all sequences into a single directory using a sequence as reference...
    //from niceMOdelFromSeq... i will need more information than the merged model in order to save the necessary stuff into the common directory

    return 0;
}

POINT_CLOUD_REGISTER_POINT_STRUCT (IndexPoint,
                                   (int, idx, idx)
                                   )
