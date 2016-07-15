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

#ifndef FAAT_PCL_RECOGNITION_GRAPH_GEOMETRIC_CONSISTENCY_IMPL_H_
#define FAAT_PCL_RECOGNITION_GRAPH_GEOMETRIC_CONSISTENCY_IMPL_H_

#include <v4r/common/graph_geometric_consistency.h>
#include <pcl/registration/correspondence_types.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/common/io.h>
#include <pcl/common/time.h>
#include <boost/unordered_map.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/biconnected_components.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <exception>

struct V4R_EXPORTS ExtendedClique
{
    std::vector<size_t> * correspondences_;
    float avg_descriptor_distance_;
    float avg_pair_3D_distance_;
    float normalized_clique_size_;
    float avg_pair_3D_distance_unnormalized_;
    float far_away_correspondences_weight_;
};

bool less_clique_vectors (const std::vector<size_t> * a, const std::vector<size_t> * b);
bool best_clique_vectors (const std::pair<float, std::vector<size_t> *> a, const std::pair<float, std::vector<size_t> *> b);
bool best_extended_cliques (const ExtendedClique & a, const ExtendedClique & b);
bool gcGraphCorrespSorter (pcl::Correspondence i, pcl::Correspondence j);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
gcGraphCorrespSorter (pcl::Correspondence i, pcl::Correspondence j)
{
    return i.distance < j.distance;
}

struct V4R_EXPORTS ViewD
{
    size_t idx_;
    size_t degree_;

    bool
    operator< (const ViewD & j) const
    {
        if (degree_ == j.degree_)
            return (idx_ < j.idx_);

        return degree_ > j.degree_;
    }
};

struct V4R_EXPORTS vertexDegreeSorter
{
    bool
    operator() (const ViewD & i, const ViewD & j) const
    {
        return i < j;
    }
};

template<typename Graph>
class V4R_EXPORTS save_cliques
{

public:
    /*save_cliques (std::size_t& max, std::size_t& maximum_clique, std::size_t& n_cliques, std::vector<std::vector<void *> *> & cliquess) :
     min_size (max), maximum (maximum_clique), n_cliques (n_cliques), cliques (cliquess)
     {
     }*/

    save_cliques (size_t max, size_t maximum_clique, size_t n_cliques, std::vector<std::vector<size_t> *> & cliques) :
        min_size_ (max), maximum_ (maximum_clique), n_cliques_ (n_cliques), cliques_ (cliques)
    {
    }

    template<typename Clique, typename Graph2>
    inline void
    clique (const Clique& c, Graph2& g)
    {

        if (c.size () >= min_size_)
        {
            BOOST_USING_STD_MAX();
            maximum_ = std::max BOOST_PREVENT_MACRO_SUBSTITUTION (maximum_, c.size());

            //save clique...
            typename Clique::const_iterator i, end = c.end ();
            std::vector<size_t> * cc = new std::vector<size_t> (c.size ());
            cliques_.push_back (cc);
            size_t p;
            for (i = c.begin (); i != end; ++i, ++p)
                cc->at (p) = (*i);

            n_cliques_++;
        }
        else
            return;
    }

    size_t min_size_;
    size_t maximum_;
    size_t n_cliques_;
    std::vector<std::vector<size_t> *> cliques_;
};

class FAATPCL_CliquesException: public std::exception
{
    virtual const char* what() const throw()
    {
        return "My exception happened";
    }
} myex;

template<typename Graph>
class V4R_EXPORTS Tomita
{
    typedef std::set<typename boost::graph_traits<Graph>::vertex_descriptor> SetType;
    typedef std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> VectorType;
    std::vector<VectorType *> cliques_found_;
    size_t min_clique_size_;
    typedef boost::unordered_map<typename boost::graph_traits<Graph>::vertex_descriptor, size_t> MapType;
    MapType used_ntimes_in_cliques_;
    std::vector<SetType> nnbrs;
    float max_time_allowed_;
    pcl::StopWatch time_elapsed_;
    bool max_time_reached_;

    void
    addClique (const VectorType & clique)
    {
        if (clique.size () >= min_clique_size_)
        {
            VectorType * vt = new VectorType (clique);
            cliques_found_.push_back (vt);
            for (size_t i = 0; i < clique.size (); i++)
            {
                used_ntimes_in_cliques_[clique[i]]++;
            }

        }
    }

    void
    printSet (const SetType & s)
    {
        typename SetType::const_iterator vertexIt, vertexEnd;

        vertexIt = s.begin ();
        vertexEnd = s.end ();

        for (; vertexIt != vertexEnd; ++vertexIt)
            std::cout << *vertexIt + 1 << " ";

        std::cout << std::endl;
    }
    //_extend(nnbrs,cand,done,clique_so_far,cliques);
    void
    extend (SetType & cand, SetType & done, VectorType & clique_so_far)
    {
        SetType small_cand, pivot_nbrs;
        int maxconn = -1;
        size_t num_cand = cand.size ();

        //iterate over done and compute maximum intersection between candidates and the adjacents of done (nnbrs)
        typename SetType::iterator vertexIt, vertexEnd;
        SetType tmp;

        vertexIt = done.begin ();
        vertexEnd = done.end ();

        for (; vertexIt != vertexEnd; ++vertexIt)
        {
            std::set_intersection (cand.begin (), cand.end (), nnbrs[*vertexIt].begin (), nnbrs[*vertexIt].end (), std::inserter (tmp, tmp.begin ()));

            if (static_cast<int> (tmp.size ()) > maxconn)
            {
                maxconn = static_cast<int> (tmp.size ());
                pivot_nbrs = tmp;
                if (maxconn == (int)num_cand)
                {
                    //All possible cliques already found
                    return;
                }
            }

            tmp.clear ();
        }

        //same for candidates
        vertexIt = cand.begin ();
        vertexEnd = cand.end ();

        for (; vertexIt != vertexEnd; ++vertexIt)
        {
            std::set_intersection (cand.begin (), cand.end (), nnbrs[*vertexIt].begin (), nnbrs[*vertexIt].end (), std::inserter (tmp, tmp.begin ()));

            if (static_cast<int> (tmp.size ()) > maxconn)
            {
                maxconn = static_cast<int> (tmp.size ());
                pivot_nbrs = tmp;
            }
            tmp.clear ();
        }

        std::set_difference (cand.begin (), cand.end (), pivot_nbrs.begin (), pivot_nbrs.end (), std::inserter (small_cand, small_cand.begin ()));
        vertexIt = small_cand.begin ();
        vertexEnd = small_cand.end ();

        for (; vertexIt != vertexEnd; ++vertexIt)
        {
            cand.erase (*vertexIt);
            clique_so_far.push_back (*vertexIt);
            SetType new_cand, new_done;
            std::set_intersection (cand.begin (), cand.end (), nnbrs[*vertexIt].begin (), nnbrs[*vertexIt].end (),
                    std::inserter (new_cand, new_cand.begin ()));

            std::set_intersection (done.begin (), done.end (), nnbrs[*vertexIt].begin (), nnbrs[*vertexIt].end (),
                    std::inserter (new_done, new_done.begin ()));

            if (new_done.size () == 0 && new_cand.size () == 0)
                addClique (clique_so_far);
            else if (new_done.size () == 0 && (new_cand.size () == 1))
            {
                if ((clique_so_far.size () + 1) >= min_clique_size_)
                {
                    VectorType tt = clique_so_far;
                    tt.push_back (*(new_cand.begin ()));
                    addClique (tt);
                }
            }
            else
            {
                float t_elapsed = static_cast<float>(time_elapsed_.getTime());
                if(t_elapsed > max_time_allowed_)
                {
                    max_time_reached_ = true;
                    return;
                }

                extend (new_cand, new_done, clique_so_far);
            }

            clique_so_far.erase (clique_so_far.begin () + (clique_so_far.size () - 1));
            done.insert (*vertexIt);
        }
    }

public:

    Tomita (size_t mins = 3)
    {
        min_clique_size_ = mins;
        max_time_allowed_ = std::numeric_limits<float>::infinity();
        max_time_reached_ = false;
    }

    bool
    getMaxTimeReached()
    {
        return max_time_reached_;
    }

    void
    setMaxTimeAllowed(float t)
    {
        max_time_allowed_ = t;
    }

    void
    find_cliques (const Graph & G, size_t num_v)
    {
        SetType cand, done;
        VectorType clique_so_far;
        nnbrs.clear ();
        used_ntimes_in_cliques_.clear ();
        cliques_found_.clear ();
        time_elapsed_.reset();
        max_time_reached_ = false;

        typename boost::graph_traits<Graph>::vertex_iterator vertexIt, vertexEnd;
        boost::tie (vertexIt, vertexEnd) = vertices (G);
        nnbrs.resize (num_v);

        size_t i = 0;
        for (; vertexIt != vertexEnd; ++vertexIt, ++i)
        {
            typename boost::graph_traits<Graph>::adjacency_iterator vi, vi_end;
            size_t k = 0;
            for (boost::tie (vi, vi_end) = boost::adjacent_vertices (*vertexIt, G); vi != vi_end; ++vi, ++k)
            {
                nnbrs[i].insert (*vi);
                cand.insert (*vi);
            }

            used_ntimes_in_cliques_[*vertexIt] = 0;
        }

        extend (cand, done, clique_so_far);
    }

    size_t
    getNumCliquesFound () const
    {
        return cliques_found_.size ();
    }

    void
    getCliques (std::vector<VectorType *> & cliques) const
    {
        cliques = cliques_found_;
    }
};


bool
less_clique_vectors (const std::vector<size_t> * a, const std::vector<size_t> * b)
{
    return a->size () < b->size ();
}

bool
best_clique_vectors (const std::pair<float, std::vector<size_t> *> a,
                     const std::pair<float, std::vector<size_t> *> b)
{
    if(a.second->size() == b.second->size())
    {
        return a.first > b.first;
    }

    return a.second->size () > b.second->size ();
}

bool
best_extended_cliques (const ExtendedClique & a,
                       const ExtendedClique & b)
{
    /*float a_value = static_cast<float>(a.correspondences_->size()) * 0.5f + a.avg_descriptor_distance_ * 0.25f + a.avg_pair_3D_distance_ * 0.25f;
    float b_value = static_cast<float>(b.correspondences_->size()) * 0.5f + b.avg_descriptor_distance_ * 0.25f + b.avg_pair_3D_distance_ * 0.25f;*/

    /*float a_value = a.avg_pair_3D_distance_ * 0.5f + a.avg_descriptor_distance_ * 0.5f;
    float b_value = b.avg_pair_3D_distance_ * 0.5f + b.avg_descriptor_distance_ * 0.5f;*/

    /*float a_value = a.avg_pair_3D_distance_ * 0.5f;
    float b_value = b.avg_pair_3D_distance_ * 0.5f;*/

    /*float a_value = static_cast<float>(a.normalized_clique_size_) * 0.5f + a.avg_pair_3D_distance_ * 0.5f;
    float b_value = static_cast<float>(b.normalized_clique_size_) * 0.5f + b.avg_pair_3D_distance_ * 0.5f;*/

    /*float a_value = static_cast<float>(a.normalized_clique_size_) * 0.25f + a.avg_descriptor_distance_ * 0.25f + a.avg_pair_3D_distance_ * 0.25f + a.far_away_correspondences_weight_ * 0.25f;
    float b_value = static_cast<float>(b.normalized_clique_size_) * 0.25f + b.avg_descriptor_distance_ * 0.25f + b.avg_pair_3D_distance_ * 0.25f + b.far_away_correspondences_weight_ * 0.25f;*/

    float a_value = static_cast<float>(a.normalized_clique_size_) * 0.25f + a.avg_descriptor_distance_ * 0.25f; //+ a.far_away_correspondences_weight_ * 0.1f;
    float b_value = static_cast<float>(b.normalized_clique_size_) * 0.25f + b.avg_descriptor_distance_ * 0.25f; //+ b.far_away_correspondences_weight_ * 0.1f;

    return a_value > b_value;
}

template<typename PointModelT, typename PointSceneT>
void
v4r::GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::cleanGraph2(GraphGGCG & g, size_t gc_thres)
{
    typename boost::graph_traits<GraphGGCG>::vertex_iterator vertexIt, vertexEnd;
    std::vector<typename boost::graph_traits<GraphGGCG>::vertex_descriptor> to_be_removed;

    do
    {
        to_be_removed.clear();
        boost::tie (vertexIt, vertexEnd) = vertices (g);
        for (; vertexIt != vertexEnd; ++vertexIt)
        {
            size_t deg = boost::out_degree (*vertexIt, g);

            if ((deg > 0) && (deg < (gc_thres - 1)))
                to_be_removed.push_back (*vertexIt);
        }

        for (size_t i = 0; i < to_be_removed.size (); i++)
            clear_vertex (to_be_removed[i], g);

    } while(!to_be_removed.empty());
}

template<typename PointModelT, typename PointSceneT>
void
v4r::GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::cleanGraph(GraphGGCG & g, size_t gc_thres)
{
    cleanGraph2(g, gc_thres);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointModelT, typename PointSceneT>
void
v4r::GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::clusterCorrespondences (std::vector<pcl::Correspondences> &model_instances)
{
  try
  {
    model_instances.clear ();
    found_transformations_.clear ();

    //for the old gc...
    std::sort (model_scene_corrs_.begin (), model_scene_corrs_.end (), gcGraphCorrespSorter);

    if (model_scene_corrs_.empty())
        throw std::runtime_error("[pcl::GeometricConsistencyGrouping::clusterCorrespondences()] Error! Correspondences not set, please set them before calling again this function.\n");

    //temp copy of scene cloud with the type cast to ModelT in order to use Ransac
    PointCloudPtr temp_scene_cloud_ptr (new PointCloud ());
    pcl::copyPointCloud<PointSceneT, PointModelT> (*scene_, *temp_scene_cloud_ptr);

    GraphGGCG correspondence_graph (model_scene_corrs_.size ());
    float min_dist_for_cluster = param_.gc_size_ * param_.dist_for_cluster_factor_;

    for (size_t k = 0; k < model_scene_corrs_.size (); ++k)
    {
        int scene_index_k = model_scene_corrs_[k].index_match;
        int model_index_k = model_scene_corrs_[k].index_query;
        const Eigen::Vector3f& scene_point_k = scene_->at (scene_index_k).getVector3fMap ();
        const Eigen::Vector3f& model_point_k = input_->at (model_index_k).getVector3fMap ();
        const Eigen::Vector3f& scene_normal_k = scene_normals_->at (scene_index_k).getNormalVector3fMap ();
        const Eigen::Vector3f& model_normal_k = input_normals_->at (model_index_k).getNormalVector3fMap ();

        for (size_t j = (k + 1); j < model_scene_corrs_.size (); ++j)
        {
            int scene_index_j = model_scene_corrs_[j].index_match;
            int model_index_j = model_scene_corrs_[j].index_query;

            //same scene or model point constraint
            if(scene_index_j == scene_index_k || model_index_j == model_index_k)
                continue;

            const Eigen::Vector3f& scene_point_j = scene_->at (scene_index_j).getVector3fMap ();
            const Eigen::Vector3f& model_point_j = input_->at (model_index_j).getVector3fMap ();

            const Eigen::Vector3f& scene_normal_j = scene_normals_->at (scene_index_j).getNormalVector3fMap ();
            const Eigen::Vector3f& model_normal_j = input_normals_->at (model_index_j).getNormalVector3fMap ();

            float dist_model_pts = (model_point_k - model_point_j).norm();
            float dist_scene_pts = (scene_point_k - scene_point_j).norm();

            //minimum distance constraint
            if ((dist_model_pts < min_dist_for_cluster) || (dist_scene_pts < min_dist_for_cluster))
                continue;

            double distance = fabs (dist_model_pts - dist_scene_pts);

            double dot_scene_pts = scene_normal_k.dot (scene_normal_j);
            double dot_model_pts = model_normal_k.dot (model_normal_j);

            //Model normals should be consistently oriented! otherwise reject!
            if(dot_model_pts < -0.1f)
                continue;

            double dot_distance = 0;
            if ( !pcl_isnan(dot_scene_pts)  && !pcl_isnan(dot_model_pts))
                dot_distance = fabs(dot_scene_pts - dot_model_pts);

            //gc constraint and dot_product constraint!
            if ((distance < param_.gc_size_) && (dot_distance <= param_.thres_dot_distance_))
                boost::add_edge (k, j, correspondence_graph);
        }
    }


    typename boost::property_map < GraphGGCG, edge_component_t>::type components = get(edge_component, correspondence_graph);
    size_t n_cc = biconnected_components(correspondence_graph, components);

    if(n_cc < 1)
        return;

    std::vector<size_t> model_instances_kept_indices;

    std::vector< std::set<size_t> > unique_vertices_per_cc (n_cc);
    std::vector<size_t> cc_sizes (n_cc, 0);

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

    pcl::registration::CorrespondenceRejectorSampleConsensus<PointModelT> corr_rejector;
    corr_rejector.setMaximumIterations (10000);
    corr_rejector.setInlierThreshold (param_.ransac_threshold_);
    corr_rejector.setInputSource (input_);
    corr_rejector.setInputTarget (temp_scene_cloud_ptr);
    corr_rejector.setSaveInliers(true);

    //Go through the connected components and decide whether to use CliqueGC or usualGC or ignore (cc_sizes[i] < gc_threshold_)
    //Decision based on the number of vertices in the connected component and graph arbocity...

    //std::cout << "Number of connected components over threshold..." << over_gc << std::endl;
    //    std::cout << "Number of connected components..." << n_cc << std::endl;
    size_t analyzed_ccs = 0;
    std::vector<bool> cliques_computation_possible_;
    cliques_computation_possible_.resize(n_cc, param_.use_graph_);
    for (size_t c = 0; c < n_cc; c++)
    {
        //ignore if not enough vertices...
        size_t num_v_in_cc = cc_sizes[c];
        if (num_v_in_cc < param_.gc_threshold_)
            continue;

        analyzed_ccs++;

        GraphGGCG connected_graph(correspondence_graph);

        //iterate over edges and remove those not belonging to this biconnected component
        boost::tie (edgeIt, edgeEnd) = edges (connected_graph);
        for (; edgeIt != edgeEnd; ++edgeIt)
        {
            if (components[*edgeIt] != c)
                boost::remove_edge(*edgeIt, connected_graph);
        }

        //std::cout << "Num edges connnected component:" << boost::num_edges(connected_graph) << std::endl;
//        visualizeGraph(connected_graph, "connected component");

        float arboricity = num_edges (connected_graph) / static_cast<float>(num_v_in_cc - 1);
        //std::cout << "arboricity:" << arboricity << " num_v:" << num_v_in_cc << " edges:" << num_edges (connected_graph) << std::endl;
        //std::vector<std::pair<int, int> > edges_used;
        std::set<size_t> correspondences_used;

        std::vector< std::vector<size_t> > correspondence_to_instance;
        if(param_.prune_by_CC_)
            correspondence_to_instance.resize(model_scene_corrs_.size());

        if (cliques_computation_possible_[c] && arboricity < 25 /*&& (num_v_in_cc < 400) && (num_edges (connected_graph) < 8000) && arboricity < 10*/)
        {
            //std::cout << "Using cliques" << std::endl;
            //std::cout << "N edges: " << num_edges (connected_graph) << " vertices:" << num_v_in_cc << " arboricity:" << arboricity <<  std::endl;

            std::vector<std::vector<size_t> *> cliques;
            {
                //pcl::ScopeTime t ("tomita cliques...");
                Tomita<GraphGGCG> tom (param_.gc_threshold_);
                tom.setMaxTimeAllowed(param_.max_time_allowed_cliques_comptutation_);
                tom.find_cliques (connected_graph, model_scene_corrs_.size ());
                if(tom.getMaxTimeReached())
                {
                    std::cout << "Max time ( " << std::setprecision(2) << param_.max_time_allowed_cliques_comptutation_ << " ms) reached during clique computation" << std::endl;
                    cliques_computation_possible_[c] = false;
                    c--;
                    analyzed_ccs--;

                    //free memory for cliques
                    tom.getCliques (cliques);
                    for (size_t p = 0; p < cliques.size (); p++)
                        delete cliques[p];

                    continue;
                }
                //std::cout << "Number of cliques found by tomita..." << tom.getNumCliquesFound () << std::endl;
                tom.getCliques (cliques);
            }

            std::vector< ExtendedClique > extended_cliques;
            std::vector<std::pair<float, std::vector<size_t> * > > cliques_with_average_weight;
            for(size_t k = 0; k < cliques.size(); k++)
            {
                float avg_dist = 0.f;
                float max_dist_ = 0.03f; //3 centimeters
                float far_away_average_weight_ = 0.f;

                for(size_t jj=0; jj < cliques[k]->size(); jj++)
                    avg_dist += model_scene_corrs_[ cliques[k]->at(jj) ].distance;

                avg_dist /= static_cast<float>(cliques[k]->size());
                cliques_with_average_weight.push_back(std::make_pair(avg_dist, cliques[k]));

                float avg_3D_dist = 0.f;

                for(size_t jj=0; jj < cliques[k]->size(); jj++)
                {

                    int scene_index_j = model_scene_corrs_[ cliques[k]->at(jj) ].index_match;
                    int model_index_j = model_scene_corrs_[ cliques[k]->at(jj) ].index_query;
                    const Eigen::Vector3f& scene_point_j = scene_->at (scene_index_j).getVector3fMap ();
                    const Eigen::Vector3f& model_point_j = input_->at (model_index_j).getVector3fMap ();

                    for(size_t kk=(jj+1); kk < cliques[k]->size(); kk++)
                    {
                        //for each pair, average 3D distance

                        int scene_index_k = model_scene_corrs_[ cliques[k]->at(kk) ].index_match;
                        int model_index_k = model_scene_corrs_[ cliques[k]->at(kk) ].index_query;

                        const Eigen::Vector3f& scene_point_k = scene_->at (scene_index_k).getVector3fMap ();
                        const Eigen::Vector3f& model_point_k = input_->at (model_index_k).getVector3fMap ();

                        Eigen::Vector3f dist_trg = model_point_k - model_point_j;
                        Eigen::Vector3f dist_ref = scene_point_k - scene_point_j;

                        float distance_ref_norm = dist_ref.norm();
                        float distance = fabs (dist_trg.norm () - dist_ref.norm());
                        avg_3D_dist += distance;

                        far_away_average_weight_ += std::min((distance_ref_norm / max_dist_), 1.f);
                    }
                }

                avg_3D_dist /= (static_cast<float>(cliques[k]->size()) * static_cast<float>(cliques[k]->size() - 1)) / 2.f;
                far_away_average_weight_ /= (static_cast<float>(cliques[k]->size()) * static_cast<float>(cliques[k]->size() - 1)) / 2.f;

                ExtendedClique ec;
                ec.correspondences_ = cliques[k];
                ec.avg_pair_3D_distance_ = avg_3D_dist;
                ec.avg_descriptor_distance_ = avg_dist;
                ec.avg_pair_3D_distance_unnormalized_ = avg_3D_dist;
                ec.far_away_correspondences_weight_ = far_away_average_weight_;
                extended_cliques.push_back(ec);
            }

            float max_avg_3D_dist = 0;
            float max_avg_descriptor_dist = 0;
            size_t max_clique_size = 0;

            for(size_t k = 0; k < cliques.size(); k++)
            {
                if(extended_cliques[k].avg_pair_3D_distance_ > max_avg_3D_dist)
                    max_avg_3D_dist = extended_cliques[k].avg_pair_3D_distance_;

                if(extended_cliques[k].correspondences_->size() > max_clique_size)
                    max_clique_size = extended_cliques[k].correspondences_->size();

                if(extended_cliques[k].avg_descriptor_distance_ > max_avg_descriptor_dist)
                    max_avg_descriptor_dist = extended_cliques[k].avg_descriptor_distance_;
            }

            for(size_t k = 0; k < cliques.size(); k++)
            {
                extended_cliques[k].avg_pair_3D_distance_ = 1.f - (extended_cliques[k].avg_pair_3D_distance_ / max_avg_3D_dist);
                extended_cliques[k].avg_descriptor_distance_ = 1.f - (extended_cliques[k].avg_descriptor_distance_ / max_avg_descriptor_dist);
                extended_cliques[k].normalized_clique_size_ = static_cast<float>(extended_cliques[k].correspondences_->size()) / static_cast<float>(max_clique_size);
            }

            //process cliques to remove similar ones...
            //sort (cliques.begin (), cliques.end (), less_clique_vectors); //cliques are sorted in increasing order (smaller cliques first)

            /*sort (cliques_with_average_weight.begin (), cliques_with_average_weight.end (), best_clique_vectors);
        for(size_t k = 0; k < cliques.size(); k++)
        {
            cliques[k] = cliques_with_average_weight[k].second;
        }*/

            sort (extended_cliques.begin (), extended_cliques.end (), best_extended_cliques);

            std::vector<std::vector<size_t> *>::iterator it;
            std::vector<size_t> taken_corresps (model_scene_corrs_.size (), 0);
            int max_taken = param_.max_taken_correspondence_;

            if(!param_.cliques_big_to_small_)
                std::reverse (cliques.begin (), cliques.end ());

            for (it = cliques.begin (); it != cliques.end (); it++)
            {
                //std::cout << "clique size:" << (*it)->size () << std::endl;
                //create a new clique based on how many time the correspondences in *it clique were used
                std::vector<size_t> * new_clique = new std::vector<size_t>;
                new_clique->reserve ((*it)->size ());
                size_t used = 0;
                for (size_t i = 0; i < (*it)->size (); i++)
                {
                    if (taken_corresps[(**it)[i]] < max_taken)
                    {
                        new_clique->push_back ((**it)[i]); //(**it)
                        used++;
                    }
                }

                if (used >= param_.gc_threshold_)
                {
                    new_clique->resize (used);

                    //do ransac with these correspondences...
                    pcl::Correspondences temp_corrs, filtered_corrs;
                    temp_corrs.reserve (used);
                    for (size_t j = 0; j < new_clique->size (); j++)
                    {
                        assert(new_clique->at (j) < model_scene_corrs_.size());
                        temp_corrs.push_back (model_scene_corrs_[ new_clique->at (j) ]);
                    }

                    corr_rejector.getRemainingCorrespondences (temp_corrs, filtered_corrs);

                    std::vector<int> inlier_indices;
                    corr_rejector.getInliersIndices (inlier_indices);

                    //check if corr_rejector.getBestTransformation () was not found already
                    bool found = poseExists (corr_rejector.getBestTransformation ());

                    if ((filtered_corrs.size () >= param_.gc_threshold_) && !found && (inlier_indices.size() != 0))
                    {
                        Eigen::Matrix4f trans = corr_rejector.getBestTransformation ();

                        //check if the normals are ok after applying the transformation
                        bool all_wrong = param_.check_normals_orientation_;

                        if( param_.check_normals_orientation_ )
                        {
                            for(size_t j=0; j < filtered_corrs.size(); j++)
                            {
                                //transform normal
                                const Eigen::Vector3f& model_normal = input_normals_->at (filtered_corrs[j].index_query).getNormalVector3fMap ();
                                const Eigen::Vector3f& scene_normal = scene_normals_->at (filtered_corrs[j].index_match).getNormalVector3fMap ();
                                if(!pcl_isfinite(model_normal[0]) || !pcl_isfinite(scene_normal[0]) ||
                                        !pcl_isfinite(model_normal[1]) || !pcl_isfinite(scene_normal[1]) ||
                                        !pcl_isfinite(model_normal[2]) || !pcl_isfinite(scene_normal[2]))
                                {
                                    continue;
                                }

                                Eigen::Vector3f nt;
                                nt[0] = static_cast<float> (trans (0, 0) * model_normal[0] + trans (0, 1) * model_normal[1] + trans (0, 2) * model_normal[2]);
                                nt[1] = static_cast<float> (trans (1, 0) * model_normal[0] + trans (1, 1) * model_normal[1] + trans (1, 2) * model_normal[2]);
                                nt[2] = static_cast<float> (trans (2, 0) * model_normal[0] + trans (2, 1) * model_normal[1] + trans (2, 2) * model_normal[2]);
                                if(nt.dot(scene_normal) >= (1.f - param_.thres_dot_distance_))
                                    all_wrong = false;

                            }
                        }

                        if(!all_wrong)
                        {
                            //PCL_INFO("Normals are consistent %d!!\n", static_cast<int>(all_nans));
                            found_transformations_.push_back (trans);
                            model_instances.push_back (filtered_corrs);

                            //mark all inliers
                            for (size_t j = 0; j < inlier_indices.size (); j++)
                            {
                                /*std::cout << "1" << inlier_indices[j] << std::endl;
                                std::cout << "2" << new_clique->size() << std::endl;
                                std::cout << "3" << new_clique->at (inlier_indices[j]) << std::endl;
                                std::cout << "4" << taken_corresps.size() << std::endl;*/

                                taken_corresps[new_clique->at (inlier_indices[j])]++;

                                if( param_.prune_by_CC_ )
                                    correspondence_to_instance[new_clique->at (inlier_indices[j])].push_back(model_instances.size() - 1);
                            }

                            if( param_.prune_by_CC_ )
                            {
                                for (size_t j = 0; j < inlier_indices.size (); j++)
                                    correspondences_used.insert(new_clique->at (inlier_indices[j]));
                            }
                        }
                        delete new_clique;
                    }
                    else
                        delete new_clique;
                }
                else
                    delete new_clique;
            }

            for (size_t p = 0; p < cliques.size (); p++)
                delete cliques[p];
        }
        else
        {
            //use iterative gc for simple cases with lots of correspondences...
            std::cout << "Correspondence grouping is too hard to solve it using cliques..." << std::endl;
//            std::cout << "N edges: " << num_edges (connected_graph) << " vertices:" << num_v_in_cc << " arboricity:" << arboricity <<  std::endl;

            std::vector<size_t> consensus_set;
            consensus_set.resize(model_scene_corrs_.size ());
            std::vector<bool> taken_corresps (model_scene_corrs_.size (), false);

            GraphGGCG connected_graph2(correspondence_graph);
            //iterate over edges and remove those not belonging to this biconnected component
            boost::tie (edgeIt, edgeEnd) = edges (connected_graph2);
            for (; edgeIt != edgeEnd; ++edgeIt)
            {
                if (components[*edgeIt] != c)
                {
                    boost::remove_edge(*edgeIt, connected_graph2);
                }
            }

            typename boost::graph_traits<GraphGGCG>::vertex_iterator vertexIt, vertexEnd;
            boost::tie (vertexIt, vertexEnd) = vertices (connected_graph2);
            for (; vertexIt != vertexEnd; ++vertexIt)
            {
                if ( boost::out_degree(*vertexIt, connected_graph2) < (param_.gc_threshold_ - 1))
                    taken_corresps[*vertexIt] = true;
            }

            for (size_t i = 0; i < model_scene_corrs_.size (); ++i)
            {
                if (taken_corresps[i])
                    continue;

                int consensus_size = 0;
                consensus_set[consensus_size++] = i;

                for (size_t j = 0; j < model_scene_corrs_.size (); ++j)
                {
                    if (j != i && !taken_corresps[j])
                    {
                        //Let's check if j fits into the current consensus set
                        bool is_a_good_candidate = true;

                        for (int k = 0; k < consensus_size; ++k)
                        {
                            //check if edge (j, consensus_set[k] exists in the graph, if it does not, is_a_good_candidate = false!...
                            if (!(boost::edge (j, consensus_set[k], connected_graph2).second))
                            {
                                is_a_good_candidate = false;
                                break;
                            }
                        }

                        if (is_a_good_candidate)
                            consensus_set[consensus_size++] = j;
                    }
                }

                if (consensus_size >= param_.gc_threshold_)
                {
                    pcl::Correspondences temp_corrs, filtered_corrs;
                    temp_corrs.reserve (consensus_size);

                    for (size_t j = 0; j < consensus_size; j++)
                        temp_corrs.push_back (model_scene_corrs_[ consensus_set[j] ]);

                    if ( param_.ransac_threshold_ > 0)
                    {
                        //pcl::ScopeTime tt("ransac filtering");
                        //ransac filtering
                        corr_rejector.getRemainingCorrespondences (temp_corrs, filtered_corrs);
                        //check if corr_rejector.getBestTransformation () was not found already
                        bool found = poseExists (corr_rejector.getBestTransformation ());

                        std::vector<int> inlier_indices;
                        corr_rejector.getInliersIndices (inlier_indices);

                        //save transformations for recognize
                        if ((filtered_corrs.size () >= param_.gc_threshold_) && !found && (inlier_indices.size() != 0))
                        {
                            Eigen::Matrix4f trans = corr_rejector.getBestTransformation ();

                            //check if the normals are ok after applying the transformation
                            bool all_wrong = param_.check_normals_orientation_;

                            if( param_.check_normals_orientation_ )
                            {
                                for(size_t j=0; j < filtered_corrs.size(); j++)
                                {
                                    //transform normal
                                    const Eigen::Vector3f& model_normal = input_normals_->at (filtered_corrs[j].index_query).getNormalVector3fMap ();
                                    const Eigen::Vector3f& scene_normal = scene_normals_->at (filtered_corrs[j].index_match).getNormalVector3fMap ();
                                    if(!pcl_isfinite(model_normal[0]) || !pcl_isfinite(scene_normal[0]) ||
                                            !pcl_isfinite(model_normal[1]) || !pcl_isfinite(scene_normal[1]) ||
                                            !pcl_isfinite(model_normal[2]) || !pcl_isfinite(scene_normal[2]))
                                    {
                                        continue;
                                    }

                                    Eigen::Vector3f nt;
                                    nt[0] = static_cast<float> (trans (0, 0) * model_normal[0] + trans (0, 1) * model_normal[1] + trans (0, 2) * model_normal[2]);
                                    nt[1] = static_cast<float> (trans (1, 0) * model_normal[0] + trans (1, 1) * model_normal[1] + trans (1, 2) * model_normal[2]);
                                    nt[2] = static_cast<float> (trans (2, 0) * model_normal[0] + trans (2, 1) * model_normal[1] + trans (2, 2) * model_normal[2]);
                                    if(nt.dot(scene_normal) >= (1.f - param_.thres_dot_distance_))
                                        all_wrong = false;

                                }
                            }

                            if(all_wrong)
                            {
                                //PCL_ERROR("Normals are not consistent %d %d!!\n", static_cast<int>(all_wrong), static_cast<int>(all_nans));
                                for (size_t j = 0; j < consensus_size; j++)
                                    taken_corresps[consensus_set[j]] = false;
                            }
                            else
                            {
                                //PCL_INFO("Normals are consistent, pushing filtered_corrs %d!!\n", static_cast<int>(filtered_corrs.size()));
                                found_transformations_.push_back (trans);
                                model_instances.push_back (filtered_corrs);

                                //mark all inliers
                                for (size_t j = 0; j < inlier_indices.size (); j++)
                                {
                                    taken_corresps[consensus_set[inlier_indices[j]]] = true;

                                    if(param_.prune_by_CC_)
                                        correspondence_to_instance[consensus_set[inlier_indices[j]]].push_back(model_instances.size() - 1);
                                }

                                if(param_.prune_by_CC_)
                                {
                                    for (size_t j = 0; j < inlier_indices.size (); j++)
                                        correspondences_used.insert(consensus_set[inlier_indices[j]]);
                                }
                            }
                        }
                        else
                        {
                            //Free the correspondences so they can be used in another set...
                            //PCL_ERROR("Freeing %d correspondences from invalid set...\n", consensus_set.size ());
                            for (size_t j = 0; j < consensus_size; j++)
                                taken_corresps[consensus_set[j]] = false;
                        }
                    }
                }
            }
        }

        if( param_.prune_by_CC_ )
        {
            //pcl::ScopeTime t("final post-processing...");
            GraphGGCG connected_graph_used_edges(connected_graph);
            typename boost::graph_traits<GraphGGCG>::vertex_iterator vertexIt, vertexEnd;
            std::vector<typename boost::graph_traits<GraphGGCG>::vertex_descriptor> to_be_removed;
            boost::tie (vertexIt, vertexEnd) = vertices (connected_graph_used_edges);
            for (; vertexIt != vertexEnd; ++vertexIt)
            {
                std::set<size_t>::const_iterator it;
                it = correspondences_used.find(*vertexIt);
                if (it == correspondences_used.end())
                    to_be_removed.push_back (*vertexIt);
            }

            for (size_t i = 0; i < to_be_removed.size (); i++)
                clear_vertex (to_be_removed[i], connected_graph_used_edges);

            boost::vector_property_map<size_t> components2 (boost::num_vertices (connected_graph_used_edges));
            size_t n_cc2 = boost::connected_components (connected_graph_used_edges, &components2[0]);

            std::vector<size_t> cc_sizes2  (n_cc2, 0);
            for (size_t i = 0; i < model_scene_corrs_.size (); i++)
                cc_sizes2[components2[i]]++;

            size_t ncc_overthres = 0;
            for (size_t i = 0; i < n_cc2; i++)
            {
                if(cc_sizes2[i] >= param_.gc_threshold_)
                    ncc_overthres++;
            }

            //std::cout << "Number of connected components over threshold: " << ncc_overthres << std::endl;

            //somehow now i need to do a Nonmax supression of the model_instances that are in the same CC
            //gather instances that were generated with correspondences found in a specific CC
            //correspondence_to_instance maps correspondences (vertices) to instance, we can use that i guess

            for (size_t internal_c = 0; internal_c < n_cc2; internal_c++)
            {
                //ignore if not enough vertices...
                size_t num_v_in_cc_tmp = cc_sizes2[internal_c];
                if (num_v_in_cc_tmp < param_.gc_threshold_)
                    continue;

                std::set<size_t> instances_for_this_cc;
                {
                    boost::tie (vertexIt, vertexEnd) = vertices (connected_graph_used_edges);

                    for (; vertexIt != vertexEnd; ++vertexIt)
                    {
                        if (components2[*vertexIt] == internal_c)
                        {
                            for(size_t k=0; k < correspondence_to_instance[*vertexIt].size(); k++)
                            {
                                instances_for_this_cc.insert(correspondence_to_instance[*vertexIt][k]);
                            }
                        }
                    }
                }

                //std::cout << "instances in this cc:" << instances_for_this_cc.size() << std::endl;
                std::set<size_t>::const_iterator it;
                size_t max_size = 0;
                for(it = instances_for_this_cc.begin(); it != instances_for_this_cc.end(); it++)
                {
                    //std::cout << *it << " " << model_instances[*it].size() << std::endl;
                    if(max_size <= model_instances[*it].size())
                    {
                        max_size = model_instances[*it].size();
                        //max_idx = *it;
                    }
                }

                //std::cout << std::endl;

                float thres = 0.5f;
                for(it = instances_for_this_cc.begin(); it != instances_for_this_cc.end(); it++)
                {
                    if( model_instances[*it].size() > (max_size * thres))
                        model_instances_kept_indices.push_back(*it);
                }
            }

            /*if(visualize_graph_ && correspondences_used.size() > 0)
            visualizeGraph(connected_graph_used_edges, "used edges");*/
        }
    }

    if(param_.prune_by_CC_)
    {
        for(size_t i=0; i < model_instances_kept_indices.size(); i++)
        {
            model_instances[i] = model_instances[model_instances_kept_indices[i]];
            found_transformations_[i] = found_transformations_[model_instances_kept_indices[i]];
        }

        model_instances.resize(model_instances_kept_indices.size());
        found_transformations_.resize(model_instances_kept_indices.size());
    }

    //visualizeCorrespondences(*model_scene_corrs_);

  }
  // HACK: Michael Zillich: added this try/catch, as somewhere above somtimes a range
  // check exception is thrown. This will have to be handled properly eventually
  catch(std::exception e)
  {
      std::cerr << "v4r::GraphGeometricConsistencyGroupingi: caught exception: "
        << e.what() << "\n";
  }
  // HACK END
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointModelT, typename PointSceneT>
bool
v4r::GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::recognize (
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<
        Eigen::Matrix4f> > &transformations)
{
    std::vector<pcl::Correspondences> model_instances;
    return (this->recognize (transformations, model_instances));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointModelT, typename PointSceneT>
bool
v4r::GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::recognize (
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<
        Eigen::Matrix4f> > &transformations,
        std::vector<pcl::Correspondences> &clustered_corrs)
{
    transformations.clear ();
    if (!this->initCompute ())
    {
        PCL_ERROR("[v4r::GraphGeometricConsistencyGrouping::recognize()] Error! Model cloud or Scene cloud not set, please set them before calling again this function.\n");
        return (false);
    }

    clusterCorrespondences (clustered_corrs);

    transformations = found_transformations_;

    this->deinitCompute ();
    return true;
}

#endif // FAAT_PCL_RECOGNITION_SI_GEOMETRIC_CONSISTENCY_IMPL_H_
