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

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/angles.h>
#include <pcl/common/centroid.h>
#include <pcl/recognition/cg/correspondence_grouping.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <v4r/core/macros.h>

#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

namespace v4r
{
class V4R_EXPORTS GraphGeometricConsistencyGroupingParameter
{
public:
    size_t gc_threshold_;  ///< Minimum cluster size. At least 3 correspondences are needed to compute the 6DOF pose
    float gc_size_; ///< Resolution of the consensus set used to cluster correspondences together. If the difference in distance between model keypoints and scene keypoints of a pair of correspondence is greater than this threshold, the correspondence pair will not be connected.
    float thres_dot_distance_;
    bool use_graph_;
    float dist_for_cluster_factor_; ///< this value times gc_size is the minimum distance between two model as well as scene points to allow them to be clustered together
    size_t max_taken_correspondence_;
    bool cliques_big_to_small_;
    bool check_normals_orientation_;
    double max_time_allowed_cliques_comptutation_;    ///< if grouping correspondences takes more processing time in milli seconds than this defined value, correspondences will be no longer computed by this graph based approach but by the simpler greedy correspondence grouping algorithm
    float ransac_threshold_;
    bool prune_;
    bool prune_by_CC_;

    GraphGeometricConsistencyGroupingParameter( ) :
        gc_threshold_ (5),
        gc_size_ ( 0.015f ),
        thres_dot_distance_ ( 0.2f ),// 0.05f
        use_graph_ ( true ),
        dist_for_cluster_factor_ ( 0.f ), //3.f
        max_taken_correspondence_ ( 5 ),
        cliques_big_to_small_ ( false ),
        check_normals_orientation_ ( true ),
        max_time_allowed_cliques_comptutation_ ( 100. ),
        ransac_threshold_ ( 0.015f ),
        prune_ ( false ),
        prune_by_CC_ ( false )
    {}

    /**
         * @brief init parameters
         * @param command_line_arguments (according to Boost program options library)
         * @return unused parameters (given parameters that were not used in this initialization call)
         */
    std::vector<std::string>
    init(int argc, char **argv)
    {
        std::vector<std::string> arguments(argv + 1, argv + argc);
        return init(arguments);
    }

    /**
         * @brief init parameters
         * @param command_line_arguments (according to Boost program options library)
         * @return unused parameters (given parameters that were not used in this initialization call)
         */
    std::vector<std::string>
    init(const std::vector<std::string> &command_line_arguments)
    {
        po::options_description desc("Graph Geometric Consistency Grouping Parameters\n=====================");
        desc.add_options()
                ("help,h", "produce help message")
                ("cg_size_thresh,c", po::value<size_t>(&gc_threshold_)->default_value(gc_threshold_), "Minimum cluster size. At least 3 correspondences are needed to compute the 6DOF pose ")
                ("cg_size", po::value<float>(&gc_size_)->default_value(gc_size_, boost::str(boost::format("%.2e") % gc_size_) ), "Resolution of the consensus set used to cluster correspondences together ")
                ("cg_thres_dot_distance", po::value<float>(&thres_dot_distance_)->default_value(thres_dot_distance_, boost::str(boost::format("%.2e") % thres_dot_distance_) ), " ")
                ("cg_use_graph", po::value<bool>(&use_graph_)->default_value(use_graph_), " ")
                ("cg_dist_for_clutter_factor", po::value<float>(&dist_for_cluster_factor_)->default_value(dist_for_cluster_factor_, boost::str(boost::format("%.2e") % dist_for_cluster_factor_) ), " ")
                ("cg_max_taken_correspondences", po::value<size_t>(&max_taken_correspondence_)->default_value(max_taken_correspondence_), " ")
                ("cg_cliques_big_to_small", po::value<bool>(&cliques_big_to_small_)->default_value(cliques_big_to_small_), " ")
                ("cg_check_normals_orientation", po::value<bool>(&check_normals_orientation_)->default_value(check_normals_orientation_), " ")
                ("cg_max_time_for_cliques_computation", po::value<double>(&max_time_allowed_cliques_comptutation_)->default_value(max_time_allowed_cliques_comptutation_, "100.0"), " if grouping correspondences takes more processing time in milliseconds than this defined value, correspondences will be no longer computed by this graph based approach but by the simpler greedy correspondence grouping algorithm")
                ("cg_ransac_threshold", po::value<float>(&ransac_threshold_)->default_value(ransac_threshold_, boost::str(boost::format("%.2e") % ransac_threshold_) ), " ")
                ("cg_prune", po::value<bool>(&prune_)->default_value(prune_), " ")
                ("cg_prune_by_CC", po::value<bool>(&prune_by_CC_)->default_value(prune_by_CC_), " ")
                ;
        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(command_line_arguments).options(desc).allow_unregistered().run();
        std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
        po::store(parsed, vm);
        if (vm.count("help")) { std::cout << desc << std::endl; to_pass_further.push_back("-h"); }
        try { po::notify(vm); }
        catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }
        return to_pass_further;
    }
};

template<typename PointModelT, typename PointSceneT>
class V4R_EXPORTS GraphGeometricConsistencyGrouping : public pcl::CorrespondenceGrouping<PointModelT, PointSceneT>
{
private:
    using pcl::CorrespondenceGrouping<PointModelT, PointSceneT>::input_;
    using pcl::CorrespondenceGrouping<PointModelT, PointSceneT>::scene_;
    using pcl::CorrespondenceGrouping<PointModelT, PointSceneT>::model_scene_corrs_;

    pcl::PointCloud<pcl::Normal>::ConstPtr scene_normals_;
    pcl::PointCloud<pcl::Normal>::ConstPtr input_normals_;

    struct edge_component_t
    {
        enum
        { num = 555 };
        typedef boost::edge_property_tag kind;
    }
    edge_component;

    typedef boost::adjacency_matrix<boost::undirectedS, size_t, boost::property<edge_component_t, std::size_t> > GraphGGCG;
    void cleanGraph(GraphGGCG & g, size_t gc_thres);

    /** \brief Transformations found by clusterCorrespondences method. */
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > found_transformations_;

    //    bool visualize_graph_;


    /** \brief Cluster the input correspondences in order to distinguish between different instances of the model into the scene.
        *
        * \param[out] model_instances a vector containing the clustered correspondences for each model found on the scene.
        * \return true if the clustering had been successful or false if errors have occurred.
        */
    void
    clusterCorrespondences (std::vector<pcl::Correspondences> &model_instances);

    /*void visualizeCorrespondences(const pcl::Correspondences & correspondences);

      void visualizeGraph(GraphGGCG & g, std::string title="Correspondence Graph");*/

public:
    GraphGeometricConsistencyGroupingParameter param_;

    typedef pcl::PointCloud<PointModelT> PointCloud;
    typedef typename PointCloud::Ptr PointCloudPtr;
    typedef typename PointCloud::ConstPtr PointCloudConstPtr;

    /** \brief Constructor */
    GraphGeometricConsistencyGrouping (const GraphGeometricConsistencyGroupingParameter &p = GraphGeometricConsistencyGroupingParameter()) : pcl::CorrespondenceGrouping<PointModelT, PointSceneT>()
    {
        param_ = p;
//        visualize_graph_ = false;
    }

    inline
    bool poseExists(const Eigen::Matrix4f &corr_rej_trans)
    {
        if(!param_.prune_)
            return false;

        const Eigen::Vector3f &trans = corr_rej_trans.block<3,1>(0,3);

        for(size_t t=0; t < found_transformations_.size(); t++)
        {
            const Eigen::Matrix4f &transf_tmp = found_transformations_[t];
            const Eigen::Vector3f &trans_found = transf_tmp.block<3,1>(0,3);
            if((trans - trans_found).norm() < param_.gc_size_)
                return true;
        }

        return false;
    }

    /** \brief The main function, recognizes instances of the model into the scene set by the user.
        *
        * \param[out] transformations a vector containing one transformation matrix for each instance of the model recognized into the scene.
        *
        * \return true if the recognition had been successful or false if errors have occurred.
        */
    bool
    recognize (std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transformations);

    /** \brief The main function, recognizes instances of the model into the scene set by the user.
        *
        * \param[out] transformations a vector containing one transformation matrix for each instance of the model recognized into the scene.
        * \param[out] clustered_corrs a vector containing the correspondences for each instance of the model found within the input data (the same output of clusterCorrespondences).
        *
        * \return true if the recognition had been successful or false if errors have occurred.
        */
    bool
    recognize (std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transformations, std::vector<pcl::Correspondences> &clustered_corrs);

    /**
     * @brief setInputAndSceneNormals
     * @param input_n input normals
     * @param scene_n scene normals
     */
    void
    setInputAndSceneNormals(const pcl::PointCloud<pcl::Normal>::ConstPtr & input_n, const pcl::PointCloud<pcl::Normal>::ConstPtr & scene_n)
    {
        input_normals_ = input_n;
        scene_normals_ = scene_n;
    }

//    void
//    setVisualizeGraph(bool vis)
//    {
//        visualize_graph_ = vis;
//    }

public:
    typedef boost::shared_ptr<GraphGeometricConsistencyGrouping<PointModelT, PointSceneT> > Ptr;
    typedef boost::shared_ptr<const GraphGeometricConsistencyGrouping<PointModelT, PointSceneT> > ConstPtr;
};
}
