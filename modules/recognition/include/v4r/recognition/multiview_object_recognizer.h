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
#include "boost_graph_visualization_extension.h"
#include "singleview_object_recognizer.h"

namespace v4r
{
class V4R_EXPORTS MultiviewRecognizer : public SingleViewRecognizer
{
private:
    MVGraph grph_, grph_final_;
    static size_t ID;
    size_t current_id_;
    std::string scene_name_;
    boost::shared_ptr< pcl::PointCloud<PointT> > pAccumulatedKeypoints_;
    boost::shared_ptr< pcl::PointCloud<pcl::Normal> > pAccumulatedKeypointNormals_;
    std::map<std::string, v4r::ObjectHypothesis<PointT> > accumulatedHypotheses_;
    pcl::visualization::PCLVisualizer::Ptr vis_;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;
//    int vp_go3d_1, vp_go3d_2;

    Eigen::Matrix4f current_global_transform_;

    BoostGraphVisualizer bgvis_;
    pcl::visualization::PCLVisualizer::Ptr go3d_vis_;
    std::vector<int> go_3d_viewports_;

protected:
    bool visualize_output_;
    void savePCDwithPose();

public:
    class MVParameter
    {
    public:
        bool scene_to_scene_;
        bool use_robot_pose_;
        bool use_gc_s2s_;
        bool go3d_;
        bool hyp_to_hyp_;   // if true adds edges for common object hypotheses
        double distance_keypoints_get_discarded_;
        int extension_mode_; // defines method used to extend information from other views (0 = keypoint correspondences (ICRA2015 paper); 1 = full hypotheses only (MVA2015 paper))
        int max_vertices_in_graph_;

        MVParameter (
                bool scene_to_scene = true,
                bool use_robot_pose = true,
                bool use_gc_s2s = true,
                bool go3d = true,
                bool hyp_to_hyp = false,
                double distance_keypoints_get_discarded = 0.005*0.005,
                int extension_mode = 0,
                int max_vertices_in_graph = 3 ) :
            scene_to_scene_ (scene_to_scene),
            use_robot_pose_ (use_robot_pose),
            use_gc_s2s_ (use_gc_s2s),
            go3d_ (go3d),
            hyp_to_hyp_ (hyp_to_hyp),
            distance_keypoints_get_discarded_ (distance_keypoints_get_discarded),
            extension_mode_ (extension_mode),
            max_vertices_in_graph_ (max_vertices_in_graph)
        {}
    };

    MVParameter mv_params_;

    MultiviewRecognizer(const MVParameter &param = MVParameter()){
        mv_params_ = param;
        visualize_output_ = false;
        pAccumulatedKeypoints_.reset (new pcl::PointCloud<PointT>);
        pAccumulatedKeypointNormals_.reset (new pcl::PointCloud<pcl::Normal>);
    }

    v4r::noise_models::NguyenNoiseModel<PointT>::Parameter nm_param_;

    bool calcSiftFeatures(ViewD &src, MVGraph &grph);

    void estimateViewTransformationBySIFT ( const ViewD &src, const ViewD &trgt, MVGraph &grph,
                                            boost::shared_ptr<flann::Index<DistT> > flann_index,
                                            Eigen::Matrix4f &transformation,
                                            std::vector<EdgeD> & edges, bool use_gc=false );

    /**
     * @brief  If a global_transform for the point cloud is provided as an argument for the recognition
     * service and use_robot_pose is set to true, additional edges between each views (vertices) are created.
     * @param src
     * @param trgt
     * @param grph
     * @param edge
     */
    void estimateViewTransformationByRobotPose ( const ViewD &src, const ViewD &trgt, MVGraph &grph, EdgeD &edge );


    /**
     * @brief Extends hypotheses construced from other views in graph by following "calling_out_edge" and recursively the other views
     */
    void extendHypothesisRecursive ( MVGraph &grph, ViewD &vrtx_start, std::vector<Hypothesis<PointT> > &hyp_vec, bool use_unverified_hypotheses = false);

    /**
     * @brief  * Transfers keypoints and hypotheses from other views (vertices) in the graph and
     * adds/merges them to the current keyppoints (existing) ones.
     * Correspondences with keypoints close to existing ones (distance, normal and model id) are
     * not transferred (to avoid redundant correspondences)
     * @param grph
     * @param vrtx_start
     * @param hypotheses
     * @param keypoints
     * @param keypointNormals
     */
    void extendFeatureMatchesRecursive ( MVGraph &grph,
                                         ViewD &vrtx_start,
                                         std::map < std::string,v4r::ObjectHypothesis<PointT> > &hypotheses,
                                         pcl::PointCloud<PointT>::Ptr keypoints,
                                         pcl::PointCloud<pcl::Normal>::Ptr keypointNormals);
    //    void calcMST ( const std::vector<Edge> &edges, const Graph &grph, std::vector<Edge> &edges_final );
    //    void createEdgesFromHypothesisMatch ( Graph &grph, std::vector<Edge> &edges );
    //    void selectLowestWeightEdgesFromParallelEdges ( const std::vector<Edge> &parallel_edges, const Graph &grph, std::vector<Edge> &single_edges );


    /**
     * @brief  Connects a new view to the graph by edges sharing a common object hypothesis between single-view
     * hypotheses in new observation and verified multi-view hypotheses in previous views
     * @param new_vertex
     * @param grph
     * @param edges
     */
    void createEdgesFromHypothesisMatchOnline ( const ViewD new_vertex, MVGraph &grph, std::vector<EdgeD> &edges );

    void calcEdgeWeight (MVGraph &grph, std::vector<EdgeD> &edges);


    std::string get_scene_name() const
    {
        return scene_name_;
    }

    bool recognize (const pcl::PointCloud<PointT>::ConstPtr cloud,
                    const Eigen::Matrix4f &global_transform = Eigen::Matrix4f::Identity());

    bool getVerifiedHypotheses(std::vector<ModelTPtr> &models,
                               std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms) const
    {
        models.clear();
        transforms.clear();

        if(num_vertices(grph_)) {
            std::pair<vertex_iter, vertex_iter> vp;
            for ( vp = vertices ( grph_ ); vp.first != vp.second; ++vp.first ) {
                if (grph_[*vp.first].id_ == ID-1) {
                    for(size_t i=0; i < grph_[*vp.first].hypothesis_mv_.size(); i++) {
                        if(grph_[*vp.first].hypothesis_mv_[i].verified_) {
                            models.push_back(grph_[*vp.first].hypothesis_mv_[i].model_);
                            transforms.push_back(grph_[*vp.first].hypothesis_mv_[i].transform_);
                        }
                    }
                    return true;
                }
            }
        }
        PCL_ERROR("There is no most current vertex in the graph.");
        return false;
    }


    bool getVerifiedHypothesesSingleView(std::vector<ModelTPtr> &models,
                                         std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms) const
    {
        models.clear();
        transforms.clear();

        if(num_vertices(grph_)) {
            std::pair<vertex_iter, vertex_iter> vp;
            for ( vp = vertices ( grph_ ); vp.first != vp.second; ++vp.first ) {
                if (grph_[*vp.first].id_ == ID-1) {
                    for(size_t i=0; i < grph_[*vp.first].hypothesis_sv_.size(); i++) {
                        if(grph_[*vp.first].hypothesis_sv_[i].verified_) {
                            models.push_back(grph_[*vp.first].hypothesis_sv_[i].model_);
                            transforms.push_back(grph_[*vp.first].hypothesis_sv_[i].transform_);
                        }
                    }
                    return true;
                }
            }
        }
        PCL_ERROR("There is no most current vertex in the graph.");
        return false;
    }

    void printParams(std::ostream &ostr = std::cout);

    /**
     * @brief saves the current (full) graph structure into a file. If filename ends with .dot, it can be opened with xdot.
     * @param filename
     */
    void printFullGraph(const std::string &filename)
    {
        outputgraph (grph_, filename.c_str() );
    }

    /**
     * @brief saves the current (reduced/final) graph structure into a file. If filename ends with .dot, it can be opened with xdot.
     * @param filename
     */
    void printFinalGraph(const std::string &filename)
    {
        outputgraph (grph_, filename.c_str() );
    }

    void visualizeOutput(bool vis)
    {
        visualize_output_ = vis;
    }
};
}

#endif
