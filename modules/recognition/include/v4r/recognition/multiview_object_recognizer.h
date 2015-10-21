/*
* multiview object instance recognizer
*
*  Created on: March, 2015
*      Author: Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      Reference: Faeulhammer et al, ICRA 2015
*                 Faeulhammer et al, MVA 2015
*/

#ifndef MYGRAPHCLASSES_H
#define MYGRAPHCLASSES_H

#include <vector>
#include <iostream>
#include <string>
#include <sstream>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/flann_search.hpp>

#include <v4r/common/noise_models.h>

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include "boost_graph_extension.h"
#include <v4r/recognition/multi_pipeline_recognizer.h>

#ifdef USE_SIFT_GPU
#include <v4r/features/sift_local_estimator.h>
#else
#include <v4r/features/opencv_sift_local_estimator.h>
#endif

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS MultiviewRecognizer : public Recognizer<PointT>
{
protected:
    typedef Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;
    typedef flann::L1<float> DistT;

    using Recognizer<PointT>::scene_;
    using Recognizer<PointT>::scene_normals_;
    using Recognizer<PointT>::models_;
    using Recognizer<PointT>::transforms_;
    using Recognizer<PointT>::hv_algorithm_;

    using Recognizer<PointT>::poseRefinement;
    using Recognizer<PointT>::hypothesisVerification;
    using Recognizer<PointT>::icp_scene_indices_;

    boost::shared_ptr<MultiRecognitionPipeline<PointT> > rr_;

    typedef boost::property<boost::edge_weight_t, CamConnect> EdgeWeightProperty;
    typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, size_t, EdgeWeightProperty> Graph;
    typedef boost::graph_traits < Graph >::vertex_descriptor ViewD;
    typedef boost::graph_traits < Graph >::edge_descriptor EdgeD;
    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
    typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
    Graph gs_;


    typedef typename std::map<std::string, ObjectHypothesis<PointT> > symHyp;

    size_t id_;

    typename std::map<size_t, View<PointT> > views_;

    std::string scene_name_;

    /** \brief stores keypoint correspondences */
    symHyp obj_hypotheses_;

    /** \brief Point-to-point correspondence grouping algorithm */
    typename boost::shared_ptr<v4r::CorrespondenceGrouping<PointT, PointT> > cg_algorithm_;

    Eigen::Matrix4f pose_;

    cv::Ptr<SiftGPU> sift_;
//    int vp_go3d_1, vp_go3d_2;

    Eigen::Matrix4f current_global_transform_;

    pcl::visualization::PCLVisualizer::Ptr go3d_vis_;
    std::vector<int> go_3d_viewports_;

    bool computeAbsolutePose(CamConnect & e, bool &is_first_edge = false);

    void pruneGraph();

    void correspondenceGrouping();

public:
    class Parameter : public Recognizer<PointT>::Parameter
    {
    public:
        using Recognizer<PointT>::Parameter::icp_iterations_;
        using Recognizer<PointT>::Parameter::icp_type_;
        using Recognizer<PointT>::Parameter::normal_computation_method_;
        using Recognizer<PointT>::Parameter::voxel_size_icp_;

        bool scene_to_scene_;
        bool use_robot_pose_;
        bool hyp_to_hyp_;   // if true adds edges for common object hypotheses (not implemented atm)
        bool use_gc_s2s_;
        bool go3d_;
        double distance_same_keypoint_;
        float same_keypoint_dot_product_;
        int extension_mode_; // defines method used to extend information from other views (0 = keypoint correspondences (ICRA2015 paper); 1 = full hypotheses only (MVA2015 paper))
        int max_vertices_in_graph_;
        float resolution_;
        float chop_z_;
        bool do_noise_modelling_;
        bool compute_mst_; // if true, does point cloud registration by SIFT background matching (given scene_to_scene_ == true),
                           // by using given pose (if use_robot_pose_ == true) and by common object hypotheses (if hyp_to_hyp_ == true)
                           // from all the possible connection a Mimimum Spanning Tree is computed.
                           // if false, it only uses the given pose for each point cloud

        Parameter (
                bool scene_to_scene = true,
                bool use_robot_pose = true,
                bool hyp_to_hyp = false,
                bool use_gc_s2s = true,
                bool go3d = true,
                double distance_same_keypoint = 0.005f*0.005f,
                float same_keypoint_dot_product = 0.8f,
                int extension_mode = 0,
                int max_vertices_in_graph = 3,
                float resolution = 0.005f,
                float chop_z = std::numeric_limits<float>::max(),
                bool do_noise_modelling = true,
                bool compute_mst = true) :
            Recognizer<PointT>::Parameter(),
            scene_to_scene_ (scene_to_scene),
            use_robot_pose_ (use_robot_pose),
            hyp_to_hyp_ (hyp_to_hyp),
            use_gc_s2s_ (use_gc_s2s),
            go3d_ (go3d),
            distance_same_keypoint_ (distance_same_keypoint),
            same_keypoint_dot_product_ (same_keypoint_dot_product),
            extension_mode_ (extension_mode),
            max_vertices_in_graph_ (max_vertices_in_graph),
            resolution_ (resolution),
            chop_z_ (chop_z),
            do_noise_modelling_ (do_noise_modelling),
            compute_mst_ (compute_mst)
        {}
    }param_;

    MultiviewRecognizer(const Parameter &p = Parameter()) : Recognizer<PointT>(p){
        param_ = p;
        id_ = 0;
        pose_ = Eigen::Matrix4f::Identity();
    }

    void setSingleViewRecognizer(const typename boost::shared_ptr<MultiRecognitionPipeline<PointT> > & rec)
    {
        rr_ = rec;
    }

    typename noise_models::NguyenNoiseModel<PointT>::Parameter nm_param_;

    bool calcSiftFeatures (const typename pcl::PointCloud<PointT>::Ptr &cloud_src,
                           typename pcl::PointCloud<PointT>::Ptr &sift_keypoints,
                           std::vector< int > &sift_keypoint_indices,
                           pcl::PointCloud<FeatureT>::Ptr &sift_signatures,
                           std::vector<float> &sift_keypoint_scales);

    void estimateViewTransformationBySIFT(const pcl::PointCloud<PointT> &src_cloud,
                                          const pcl::PointCloud<PointT> &dst_cloud,
                                          const std::vector<int> &src_sift_keypoint_indices,
                                          const std::vector<int> &dst_sift_keypoint_indices,
                                          const pcl::PointCloud<FeatureT> &src_sift_signatures,
                                          boost::shared_ptr< flann::Index<DistT> > &dst_flann_index,
                                          std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transformations,
                                          bool use_gc = false );


    //    void calcMST ( const std::vector<Edge> &edges, const Graph &grph, std::vector<Edge> &edges_final );
    //    void createEdgesFromHypothesisMatch ( Graph &grph, std::vector<Edge> &edges );
    //    void selectLowestWeightEdgesFromParallelEdges ( const std::vector<Edge> &parallel_edges, const Graph &grph, std::vector<Edge> &single_edges );

    float calcEdgeWeightAndRefineTf (const typename pcl::PointCloud<PointT>::ConstPtr &cloud_src,
                                    const typename pcl::PointCloud<PointT>::ConstPtr &cloud_dst,
                                    Eigen::Matrix4f &refined_transform,
                                    const Eigen::Matrix4f &transform = Eigen::Matrix4f::Identity());


    std::string get_scene_name() const
    {
        return scene_name_;
    }

    void set_sift(cv::Ptr<SiftGPU> &sift)
    {
        sift_ = sift;
    }


    /** \brief Sets the algorithm for Correspondence Grouping (Hypotheses generation from keypoint correspondences) */
    void
    setCGAlgorithm (const typename boost::shared_ptr<v4r::CorrespondenceGrouping<PointT, PointT> > & alg)
    {
      cg_algorithm_ = alg;
    }

    /** \brief sets the camera pose of the input cloud */
    void setCameraPose(const Eigen::Matrix4f &tf)
    {
        pose_ = tf;
    }

    typename boost::shared_ptr<Source<PointT> >
    getDataSource () const
    {
        return rr_->getDataSource();
    }

    void recognize();

};
}

#endif
