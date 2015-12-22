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

#ifndef FAAT_PCL_RECOGNITION_GRAPH_GEOMETRIC_CONSISTENCY_H_
#define FAAT_PCL_RECOGNITION_GRAPH_GEOMETRIC_CONSISTENCY_H_

#include "correspondence_grouping.h"
#include <pcl/point_cloud.h>
#include <pcl/common/angles.h>
#include <pcl/common/centroid.h>
//#include <pcl/visualization/pcl_visualizer.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <v4r/core/macros.h>

namespace v4r
{
  template<typename PointModelT, typename PointSceneT>
  class V4R_EXPORTS GraphGeometricConsistencyGrouping : public CorrespondenceGrouping<PointModelT, PointSceneT>
  {

    struct edge_component_t
    {
      enum
      { num = 555 };
      typedef boost::edge_property_tag kind;
    }
    edge_component;

    typedef boost::adjacency_matrix<boost::undirectedS, size_t, boost::property<edge_component_t, std::size_t> > GraphGGCG;
    void cleanGraph2(GraphGGCG & g, size_t gc_thres);
    void cleanGraph(GraphGGCG & g, size_t gc_thres);

    public:
      class V4R_EXPORTS Parameter
      {
      public:
          size_t gc_threshold_;  /// @brief Minimum cluster size. At least 3 correspondences are needed to compute the 6DOF pose
          double gc_size_; /// @brief Resolution of the consensus set used to cluster correspondences together
          double thres_dot_distance_;
          bool use_graph_;
          double dist_for_cluster_factor_;
          size_t max_taken_correspondence_;
          bool cliques_big_to_small_;
          bool check_normals_orientation_;
          double max_time_allowed_cliques_comptutation_;    /// @brief if grouping correspondences takes more processing time in milli seconds than this defined value, correspondences will be no longer computed by this graph based approach but by the simpler greedy correspondence grouping algorithm
          double ransac_threshold_;
          bool prune_;
          bool prune_by_CC_;

          Parameter(
                  size_t gc_threshold = 5,
                  double gc_size = 0.015,
                  double thres_dot_distance = 0.2f, // 0.05f
                  bool use_graph = true,
                  double dist_for_cluster_factor = 0., //3.f
                  size_t max_taken_correspondence = 3, //5
                  bool cliques_big_to_small = false,
                  bool check_normals_orientation = true,
                  double max_time_allowed_cliques_comptutation = 100, //std::numeric_limits<double>::infinity()
                  double ransac_threshold = 0.015f,
                  bool prune = false,
                  bool prune_by_CC = false
                  )
          : gc_threshold_ (gc_threshold),
            gc_size_ ( gc_size ),
            thres_dot_distance_ ( thres_dot_distance ),
            use_graph_ ( use_graph ),
            dist_for_cluster_factor_ ( dist_for_cluster_factor ),
            max_taken_correspondence_ ( max_taken_correspondence ),
            cliques_big_to_small_ ( cliques_big_to_small ),
            check_normals_orientation_ ( check_normals_orientation ),
            max_time_allowed_cliques_comptutation_ ( max_time_allowed_cliques_comptutation ),
            ransac_threshold_ ( ransac_threshold ),
            prune_ ( prune ),
            prune_by_CC_ ( prune_by_CC )
      {}
      }param_;

      typedef pcl::PointCloud<PointModelT> PointCloud;
      typedef typename PointCloud::Ptr PointCloudPtr;
      typedef typename PointCloud::ConstPtr PointCloudConstPtr;

      typedef typename CorrespondenceGrouping<PointModelT, PointSceneT>::SceneCloudConstPtr SceneCloudConstPtr;

      /** \brief Constructor */
      GraphGeometricConsistencyGrouping (const Parameter &p = Parameter()) : CorrespondenceGrouping<PointModelT, PointSceneT>()
      {
        param_ = p;
        require_normals_ = true;
        visualize_graph_ = false;
      }

      inline
      void setMaxTimeForCliquesComputation(float t)
      {
          param_.max_time_allowed_cliques_comptutation_ = t;
      }

      inline
      void setCheckNormalsOrientation(bool b)
      {
        param_.check_normals_orientation_ = b;
      }

      inline
      void setSortCliques(bool b)
      {
        param_.cliques_big_to_small_ = b;
      }

      inline
      void setMaxTaken(size_t t)
      {
        param_.max_taken_correspondence_ = t;
      }

      inline
      void pruneByCC(bool b)
      {
        param_.prune_by_CC_ = b;
      }

      inline
      void setDistForClusterFactor(float f)
      {
        param_.dist_for_cluster_factor_ = f;
      }

      inline
      bool poseExists(Eigen::Matrix4f corr_rej_trans)
      {
        if(!param_.prune_)
          return false;

        bool found = false;
        Eigen::Vector3f trans = corr_rej_trans.block<3,1>(0,3);
        Eigen::Quaternionf quat(corr_rej_trans.block<3,3>(0,0));
        quat.normalize();
        Eigen::Quaternionf quat_conj = quat.conjugate();

        for(size_t t=0; t < found_transformations_.size(); t++)
        {
          Eigen::Matrix4f transf_tmp = found_transformations_[t];
          Eigen::Vector3f trans_found = transf_tmp.block<3,1>(0,3);
          if((trans - trans_found).norm() < param_.gc_size_)
          {
            found = true;
            break;

            Eigen::Matrix4f trans_tmp = found_transformations_[t];
            Eigen::Quaternionf quat_found(trans_tmp.block<3,3>(0,0));
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

      inline
      void setPrune(bool b)
      {
        param_.prune_ = b;
      }

      inline void
      setUseGraph (bool b)
      {
        param_.use_graph_ = b;
      }

      inline
      void setDotDistance(double t)
      {
        param_.thres_dot_distance_ = t;
      }

      /** \brief Sets the minimum cluster size
        * \param[in] threshold the minimum cluster size 
        */
      inline void
      setGCThreshold (size_t threshold)
      {
        param_.gc_threshold_ = threshold;
      }

      /** \brief Gets the minimum cluster size.
        * 
        * \return the minimum cluster size used by GC.
        */
      inline size_t
      getGCThreshold () const
      {
        return param_.gc_threshold_;
      }

      /** \brief Sets the consensus set resolution. This should be in metric units.
        * 
        * \param[in] gc_size consensus set resolution.
        */
      inline void
      setGCSize (double gc_size)
      {
        param_.gc_size_ = gc_size;
      }

      /** \brief Gets the consensus set resolution.
        * 
        * \return the consensus set resolution.
        */
      inline double
      getGCSize () const
      {
        return param_.gc_size_;
      }

      /** \brief Sets the minimum cluster size
        * \param[in] threshold the minimum cluster size
        */
      inline void
      setRansacThreshold (double threshold)
      {
        param_.ransac_threshold_ = threshold;
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

      void setInputAndSceneNormals(const pcl::PointCloud<pcl::Normal>::Ptr & input_n, const pcl::PointCloud<pcl::Normal>::Ptr & scene_n)
      {
        input_normals_ = input_n;
        scene_normals_ = scene_n;
      }

      void
      setVisualizeGraph(bool vis)
      {
        visualize_graph_ = vis;
      }

    protected:
      using CorrespondenceGrouping<PointModelT, PointSceneT>::input_;
      using CorrespondenceGrouping<PointModelT, PointSceneT>::scene_;
      using CorrespondenceGrouping<PointModelT, PointSceneT>::model_scene_corrs_;
      using CorrespondenceGrouping<PointModelT, PointSceneT>::require_normals_;

      pcl::PointCloud<pcl::Normal>::Ptr scene_normals_;

      pcl::PointCloud<pcl::Normal>::Ptr input_normals_;

      bool visualize_graph_;


      /** \brief Transformations found by clusterCorrespondences method. */
      std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > found_transformations_;

      /** \brief Cluster the input correspondences in order to distinguish between different instances of the model into the scene.
        * 
        * \param[out] model_instances a vector containing the clustered correspondences for each model found on the scene.
        * \return true if the clustering had been successful or false if errors have occurred.
        */ 
      void
      clusterCorrespondences (std::vector<pcl::Correspondences> &model_instances);

      /*void visualizeCorrespondences(const pcl::Correspondences & correspondences);

      void visualizeGraph(GraphGGCG & g, std::string title="Correspondence Graph");*/
  };
}

#endif // FAAT_PCL_RECOGNITION_SI_GEOMETRIC_CONSISTENCY_H_
