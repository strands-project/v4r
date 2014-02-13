/*
 * object_modeller.h
 *
 *  Created on: Mar 15, 2013
 *      Author: aitor
 */

#ifndef FAAT_PCL_OBJECT_MODELLER_H_
#define FAAT_PCL_OBJECT_MODELLER_H_

#define VISUALIZATION

#include <pcl/common/common.h>

#ifdef VISUALIZATION
  #include <pcl/visualization/pcl_visualizer.h>
#endif

#include <flann/flann.h>
#include <pcl/octree/octree.h>
#include <boost/graph/adjacency_list.hpp>

namespace faat_pcl
{
  namespace object_modelling
  {
    template<template<class > class Distance, typename PointT, typename PointTNormal>
    class ObjectModeller
    {
      private:
        typedef Distance<float> DistT;
        typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
        typedef typename pcl::PointCloud<pcl::Normal>::Ptr PointCloudNormalPtr;
        typedef boost::property<boost::edge_weight_t, float> EdgeWeightProperty;
        typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::directedS, int, EdgeWeightProperty > DirectedGraph;
        typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, int, EdgeWeightProperty > Graph;

        std::vector<PointCloudPtr> clouds_;
        std::vector<PointCloudPtr> processed_clouds_;
        std::vector<PointCloudPtr> range_images_;
        float focal_length_; float cx_; float cy_;

        std::vector< typename pcl::PointCloud<PointTNormal>::Ptr > processed_xyz_normals_clouds_;
        std::vector<PointCloudPtr> keypoint_clouds_;
        std::vector<PointCloudNormalPtr> clouds_normals_;
        std::vector<PointCloudNormalPtr> clouds_normals_at_keypoints_;
        std::vector<pcl::PointCloud<pcl::Histogram<352> >::Ptr > clouds_signatures_;

        Graph LR_graph_;
        DirectedGraph ST_graph_;

        //Matchin stuff
        int kdtree_splits_;
        bool refine_features_poses_;
        bool use_max_cluster_only_;
        bool use_gc_icp_;
        bool bf_pairwise_;
        bool icp_with_gc_survival_of_the_fittest_;
        bool VIS_FINAL_GC_ICP_;
        float dt_vx_size_;

        class flann_model
        {
        public:
          int keypoint_id;
          std::vector<float> descr;
        };

        class ourcvfh_flann_model
        {
        public:
          int view_id;
          int descriptor_id;
          std::vector<float> descr;
        };

        class codebook_model
        {
          public:
            int cluster_idx_;
            std::vector<float> descr;
            std::vector<int> clustered_indices_to_flann_models_;
        };

        struct index_score
        {
          int idx_models_;
          int idx_input_;
          double score_;
        };

        struct sortIndexScores
        {
          bool
          operator() (const index_score& d1, const index_score& d2)
          {
            return d1.score_ < d2.score_;
          }
        } sortIndexScoresOp;

        std::vector<flann::Matrix<float> > signatures_flann_data_;
        std::vector<flann::Index<DistT> * > flann_index_;
        std::vector<std::vector<flann_model> > clouds_flann_signatures_;
        std::vector<std::vector<codebook_model> > clouds_codebooks_;
        std::vector<std::vector<bool> > is_candidate_;

        template <typename Type>
        inline void
        convertToFLANN (const std::vector<Type> &models, flann::Matrix<float> &data)
        {
          data.rows = models.size ();
          data.cols = models[0].descr.size (); // number of histogram bins

          flann::Matrix<float> flann_data (new float[models.size () * models[0].descr.size ()], models.size (), models[0].descr.size ());

          for (size_t i = 0; i < data.rows; ++i)
            for (size_t j = 0; j < data.cols; ++j)
            {
              flann_data.ptr ()[i * data.cols + j] = models[i].descr[j];
            }

          data = flann_data;
        }

        void
        nearestKSearch (flann::Index<DistT> * index, float * descr, int descr_size, int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
        {
          flann::Matrix<float> p = flann::Matrix<float> (new float[descr_size], 1, descr_size);
          memcpy (&p.ptr ()[0], &descr[0], p.cols * p.rows * sizeof(float));

          std::cout << indices.rows << " " << p.rows << std::endl;
          index->knnSearch (p, indices, distances, k, flann::SearchParams (kdtree_splits_));
          delete[] p.ptr ();
        }

        inline void
        getIndicesFromCloud(PointCloudPtr & processed, PointCloudPtr & keypoints_pointcloud, std::vector<int> & indices)
        {
          pcl::octree::OctreePointCloudSearch<PointT> octree (0.003);
          octree.setInputCloud (processed);
          octree.addPointsFromInputCloud ();

          std::vector<int> pointIdxNKNSearch;
          std::vector<float> pointNKNSquaredDistance;

          for(size_t j=0; j < keypoints_pointcloud->points.size(); j++)
          {
           if (octree.nearestKSearch (keypoints_pointcloud->points[j], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
           {
             indices.push_back(pointIdxNKNSearch[0]);
           }
          }
        }

        void
        getCorrespondences(PointCloudPtr i, PointCloudPtr j, pcl::CorrespondencesPtr & corresp_i_j);

        struct
        pair_wise_registration
        {
          float reg_error_;
          int overlap_;
          float fsv_fraction_;
          float fsv_local_;
          float loop_closure_probability_;
        };

        std::vector< std::vector<pcl::CorrespondencesPtr> > pairwise_correspondences_;
        std::vector< std::vector< std::vector< Eigen::Matrix4f> > >  pairwise_poses_;
        std::vector< std::vector< std::vector<pcl::Correspondences> > > pairwise_corresp_clusters_;
        std::vector< std::vector< std::vector< pair_wise_registration > > > pairwise_registration_;

        std::vector< Eigen::Matrix4f > absolute_poses_;

#ifdef VISUALIZATION
        void
        drawCorrespondences (PointCloudPtr & scene_cloud, PointCloudPtr & model_cloud, PointCloudPtr & keypoints_pointcloud, PointCloudPtr & keypoints_model, pcl::Correspondences & correspondences)
        {
          pcl::visualization::PCLVisualizer vis_corresp_;
          vis_corresp_.setWindowName("correspondences...");
          pcl::visualization::PointCloudColorHandlerCustom<PointT> random_handler (scene_cloud, 255, 0, 0);
          vis_corresp_.addPointCloud<PointT> (scene_cloud, random_handler, "points");

          pcl::visualization::PointCloudColorHandlerCustom<PointT> random_handler_sampled (model_cloud, 0, 0, 255);
          vis_corresp_.addPointCloud<PointT> (model_cloud, random_handler_sampled, "sampled");

          for (size_t kk = 0; kk < correspondences.size (); kk++)
          {
            pcl::PointXYZ p;
            p.getVector4fMap () = keypoints_model->points[correspondences[kk].index_query].getVector4fMap ();
            pcl::PointXYZ p_scene;
            p_scene.getVector4fMap () = keypoints_pointcloud->points[correspondences[kk].index_match].getVector4fMap ();

            std::stringstream line_name;
            line_name << "line_" << kk;

            vis_corresp_.addLine<pcl::PointXYZ, pcl::PointXYZ> (p_scene, p, line_name.str ());
          }

          vis_corresp_.spin ();
          vis_corresp_.removeAllPointClouds();
          vis_corresp_.removeAllShapes();
          vis_corresp_.close();
        }
#endif

        //void selectBestRelativePose(int i, int j);

        void refineRelativePoses(int i, int j);

        void refineRelativePosesICPWithGC(int i, int j);

        /*void computeAbsolutePosesFromMST(DirectedGraph & mst,
                                             boost::graph_traits < DirectedGraph >::vertex_descriptor & root);

        void computeAbsolutePosesRecursive(DirectedGraph & mst,
                                               boost::graph_traits < DirectedGraph >::vertex_descriptor & start,
                                               Eigen::Matrix4f accum);

        bool graphGloballyConsistent();*/

        //void computePairWisePosesBruteFroce();
        //void computePairWisePosesSelective();

        float ov_percentage_;

      public:

        ObjectModeller()
        {
          kdtree_splits_ = 128;
          refine_features_poses_ = true;
          use_max_cluster_only_ = true;
          use_gc_icp_ = false;
          bf_pairwise_ = true;
          icp_with_gc_survival_of_the_fittest_ = false;
          ov_percentage_ = 0.5f;
          VIS_FINAL_GC_ICP_ = false;
          dt_vx_size_ = 0.003f;
        }

        inline void
        setDtVxSize(float f)
        {
          dt_vx_size_ = f;
        }

        inline void
        setVisFinal(bool b)
        {
          VIS_FINAL_GC_ICP_ = b;
        }

        inline void
        setOverlapPercentage(float f)
        {
          ov_percentage_ = f;
        }

        void saveGraph(std::string & directory);

        void setICPGCSurvivalOfTheFittest(bool b)
        {
          icp_with_gc_survival_of_the_fittest_ = b;
        }

        void
        setBFPairwise(bool b)
        {
          bf_pairwise_ = b;
        }

        void
        setUseGCICP(bool b)
        {
          use_gc_icp_ = b;
        }

        void
        setUseMaxClusterOnly(bool b)
        {
          use_max_cluster_only_ = b;
        }

        void addInputCloud(PointCloudPtr & cloud)
        {
          clouds_.push_back(cloud);
        }

        void setRefineFeaturesPoses(bool b)
        {
          refine_features_poses_ = b;
        }

        void setRangeImages(std::vector<PointCloudPtr> & range_im,
                              std::vector<std::vector<int> > & obj_indices,
                              float focal_length, float cx, float cy)
        {
          for(size_t i=0; i < range_im.size(); i++)
          {
            //set the range_im to NaN at those positions where
            PointCloudPtr obj_range_im_(new pcl::PointCloud<PointT>);
            obj_range_im_->width = range_im[i]->width;
            obj_range_im_->height = range_im[i]->height;
            obj_range_im_->points.resize(range_im[i]->points.size());
            obj_range_im_->is_dense = true;
            for(size_t j=0; j < obj_range_im_->points.size(); j++)
            {
              obj_range_im_->points[j].x = std::numeric_limits<float>::quiet_NaN();
              obj_range_im_->points[j].y = std::numeric_limits<float>::quiet_NaN();
              obj_range_im_->points[j].z = std::numeric_limits<float>::quiet_NaN();
            }

            for(size_t j=0; j < obj_indices[i].size(); j++)
            {
              obj_range_im_->points[obj_indices[i][j]] = range_im[i]->points[obj_indices[i][j]];
            }
            //pcl::copyPointCloud(*range_im[i], obj_indices[i], *obj_range_im_);

            std::cout << obj_range_im_->width << " " << obj_range_im_->height << std::endl;
            std::cout << range_im[i]->width << " " << range_im[i]->height << std::endl;

            range_images_.push_back(obj_range_im_);
          }

          focal_length_ = focal_length;
          cx_ = cx;
          cy_ = cy;
        }

        void processClouds();
        //void computeFeatures();
        //void computePairWiseRelativePosesWithFeatures();
        void computeRelativePosesWithICP();

        void refinePosesGlobally(DirectedGraph & mst,
                                    boost::graph_traits < DirectedGraph >::vertex_descriptor & root);

        void visualizeProcessed();

        void visualizePairWiseAlignment();

        void visualizeGlobalAlignment(bool visualize_cameras=false);
        /*
         * This function builds a graph from pair-wise alignment.
         * A node in the graph represents a view. An edge between two nodes
         * represents the relative pose between nodes and the cost of traversing that edge
         * The edge weights that can be used are:
         *    -> #correspondences
         *    -> %overlap
         *    -> weighted overlap
         */
        //void globalSelectionWithMST(bool refine_global_pose = false);
    };

  }
}

#endif /* FAAT_PCL_OBJECT_MODELLER_H_ */
