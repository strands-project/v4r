#include <pcl/common/common.h>
#ifndef V4R_REGISTRATION_MSM
#define V4R_REGISTRATION_MSM

#include "PartialModelRegistrationBase.h"

#include <boost/graph/copy.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>

#include <v4r/core/macros.h>

#include <limits>

//MST stuff
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

    bool operator<=(const myEdge& e) const {
        if(edge_weight <= e.edge_weight)
            return true;

        return false;
    }

    bool operator>(const myEdge& e) const {
        if(edge_weight > e.edge_weight)
            return true;

        return false;
    }

    bool operator>=(const myEdge& e) const {
        if(edge_weight >= e.edge_weight)
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

namespace v4r
{
    namespace Registration
    {
        class EdgeBetweenPartialModels
        {
            public:
                Eigen::Matrix4f transformation_;
                size_t i_, j_;
                float cost_;
        };

        template<class PointT>
        class V4R_EXPORTS MultiSessionModelling
        {
            private:

                typedef typename pcl::PointCloud<PointT>::Ptr PointCloudTPtr;

                std::vector<PointCloudTPtr> clouds_;

                //initial poses bringing clouds_ into alignment
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses_;

                //indices to clouds indicating where the object is
                std::vector<std::vector<int> > object_indices_;

                //normal clouds...
                std::vector< pcl::PointCloud<pcl::Normal>::Ptr > normals_;

                //first and final cloud for each session
                std::vector<std::pair<int,int> > session_ranges_;

                //for each session, pose aligning all poses to the CS of the first
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > output_session_poses_;

                //poses aligning all clouds to the CS of the first session
                //basically, a concatenation of output_session_poses_ and poses_
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > output_cloud_poses_;

                std::vector< boost::shared_ptr< PartialModelRegistrationBase<PointT> > > reg_algos_;

                void computeCost(EdgeBetweenPartialModels & edge);

                float computeFSV (PointCloudTPtr & cloud,
                                  pcl::PointCloud<pcl::Normal>::Ptr & normals,
                                  std::vector<int> & indices,
                                  Eigen::Matrix4f & pose,
                                  PointCloudTPtr & range_image);


                //MST stuff...
                typedef boost::property<boost::edge_weight_t, myEdge> EdgeWeightProperty;
                typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, int, EdgeWeightProperty> Graph;
                typedef boost::graph_traits < Graph >::vertex_descriptor Vertex;
                typedef boost::graph_traits < Graph >::edge_descriptor Edge;
                typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
                typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;

                inline void
                computeAbsolutePosesRecursive (Graph & grph_final,
                                              Vertex start, Vertex coming_from,
                                              Eigen::Matrix4f accum,
                                              std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & absolute_poses)
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
                      transform = my_e.transformation.inverse ();
                    }
                    else
                    {
                      transform = my_e.transformation;
                    }

                    internal_accum = accum * transform;
                    absolute_poses[targ] = internal_accum;
                    computeAbsolutePosesRecursive (grph_final, targ, start, internal_accum, absolute_poses);
                  }
                }

                void
                computeAbsolutePoses (Graph & grph_final,
                                     std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & absolute_poses)
                {
                  std::pair<vertex_iter, vertex_iter> vp;
                  vp = vertices (grph_final);
                  Eigen::Matrix4f accum;
                  accum.setIdentity ();
                  absolute_poses[*vp.first] = accum;
                  computeAbsolutePosesRecursive (grph_final, *vp.first, *vp.first, accum, absolute_poses);
                }

            public:

                MultiSessionModelling();

                void setInputData(std::vector<PointCloudTPtr> & clouds,
                                  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & poses,
                                  std::vector<std::vector<int> > & object_indices,
                                  std::vector<std::pair<int,int> > & session_ranges)
                {
                    clouds_ = clouds;
                    poses_ = poses;
                    object_indices_ = object_indices;
                    session_ranges_ = session_ranges;
                }

                void setInputNormals(std::vector< pcl::PointCloud<pcl::Normal>::Ptr > & normals)
                {
                    normals_ = normals;
                }

                void addRegAlgorithm(typename boost::shared_ptr< PartialModelRegistrationBase<PointT> > & alg)
                {
                    reg_algos_.push_back(alg);
                }

                //access functions...
                size_t getTotalNumberOfClouds()
                {
                    return clouds_.size();
                }

                PointCloudTPtr getCloud(size_t i)
                {
                    return clouds_[i];
                }

                pcl::PointCloud<pcl::Normal>::Ptr getNormal(size_t i)
                {
                    return normals_[i];
                }

                std::vector<int> & getIndices(size_t i)
                {
                    return object_indices_[i];
                }

                Eigen::Matrix4f getPose(size_t i)
                {
                    return poses_[i];
                }

                void getOutputPoses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & out)
                {
                    out = output_cloud_poses_;
                }

                void compute();
        };
    }
}

#endif
