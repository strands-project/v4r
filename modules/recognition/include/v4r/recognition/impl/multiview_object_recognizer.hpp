/******************************************************************************
 * Copyright (c) 2015 Thomas Faeulhammer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/

/**
*
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date August, 2015
*      @brief multiview object instance recognizer
*      Reference(s): Faeulhammer et al, ICRA 2015
*                    Faeulhammer et al, MVA 2015
*/

#include <v4r/recognition/multiview_object_recognizer.h>

#include <math.h>       // atan2
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>
#include <omp.h>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

#include <v4r/common/normals.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/registration/FeatureBasedRegistration.h>
#include <v4r/registration/metrics.h>
#include <v4r/recognition/segmenter.h>
#include <v4r/segmentation/multiplane_segmentation.h>
#include <v4r/segmentation/segmentation_utils.h>

#ifdef HAVE_SIFTGPU
#include <v4r/features/sift_local_estimator.h>
#else
#include <v4r/features/opencv_sift_local_estimator.h>
#endif

namespace v4r
{

template<typename PointT>
bool
MultiviewRecognizer<PointT>::calcSiftFeatures (const typename pcl::PointCloud<PointT>::Ptr &cloud_src,
                                               std::vector< int > &sift_keypoint_indices,
                                               std::vector<std::vector<float> > &sift_signatures)
{
    SIFTLocalEstimation<PointT> estimator(sift_);
    estimator.setInputCloud(cloud_src);
    estimator.compute (sift_signatures);
    sift_keypoint_indices = estimator.getKeypointIndices(  );
    return true;
}

template<typename PointT>
void
MultiviewRecognizer<PointT>::pruneGraph ()
{
    if( views_.size() > param_.max_vertices_in_graph_ ) {
        size_t lowest_vertex_id = std::numeric_limits<size_t>::max();

        typename std::map<size_t, View<PointT> >::const_iterator view_it;
        for (view_it = views_.begin(); view_it != views_.end(); ++view_it) {
             if( view_it->first < lowest_vertex_id)
                 lowest_vertex_id = view_it->first;
        }

        views_.erase(lowest_vertex_id);

        if (param_.compute_mst_) {
            std::pair<vertex_iter, vertex_iter> vp;
            for (vp = vertices(gs_); vp.first != vp.second; ++vp.first) {
                if (gs_[*vp.first] == lowest_vertex_id) {
                    clear_vertex(*vp.first, gs_);
                    remove_vertex(*vp.first, gs_); // iterator might be invalidated but we stop anyway
                    break;
                }
            }
        }
    }
}

template<typename PointT>
bool
MultiviewRecognizer<PointT>::computeAbsolutePose(CamConnect &e, bool is_first_edge)
{
    size_t src = e.source_id_;
    size_t trgt = e.target_id_;
    View<PointT> &src_tmp = views_[src];
    View<PointT> &trgt_tmp = views_[trgt];

    std::cout << "[" << src << "->" << trgt << "] with weight " << e.edge_weight_ << " by " << e.model_name_ << std::endl;

    if (is_first_edge) {
        src_tmp.has_been_hopped_ = true;
        src_tmp.absolute_pose_ = Eigen::Matrix4f::Identity();
        src_tmp.cumulative_weight_to_new_vrtx_ = 0.f;
        is_first_edge = false;
    }

    if(src_tmp.has_been_hopped_) {
        trgt_tmp.has_been_hopped_ = true;
        trgt_tmp.absolute_pose_ = src_tmp.absolute_pose_ * e.transformation_;
        trgt_tmp.cumulative_weight_to_new_vrtx_ = src_tmp.cumulative_weight_to_new_vrtx_ + e.edge_weight_;
    }
    else if (trgt_tmp.has_been_hopped_) {
        src_tmp.has_been_hopped_ = true;
        src_tmp.absolute_pose_ = trgt_tmp.absolute_pose_ * e.transformation_.inverse();
        src_tmp.cumulative_weight_to_new_vrtx_ = trgt_tmp.cumulative_weight_to_new_vrtx_ + e.edge_weight_;
    }
    else {
        std::cerr << "None of the vertices has been hopped yet!";
        return false;
    }

    return true;
}

template<typename PointT>
void
MultiviewRecognizer<PointT>::recognize ()
{
    if(!rr_)
        throw std::runtime_error("Single-View recognizer is not set. Please provide a recognizer to the multi-view recognition system!");

    std::cout << "=================================================================" << std::endl <<
                 "Started recognition for view " << id_ << " in scene " << scene_name_ <<
                 "=========================================================" << std::endl << std::endl;

    boost::shared_ptr< pcl::PointCloud<pcl::Normal> > scene_normals_f (new pcl::PointCloud<pcl::Normal> );

    if (!scene_ || scene_->width != 640 || scene_->height != 480)
        throw std::runtime_error("Size of input cloud is not 640x480, which is the only resolution currently supported by the verification framework.");

    View<PointT> vv;
    views_[id_] = vv;
    View<PointT> &v = views_[id_];

    v.id_ = id_;
    v.scene_ = scene_;
    v.transform_to_world_co_system_ = pose_;
    v.absolute_pose_ = pose_;

    computeNormals<PointT>(v.scene_, v.scene_normals_, param_.normal_computation_method_);

    if( param_.chop_z_ > 0 && std::isfinite(param_.chop_z_)) {
        for(size_t i=0; i <v.scene_->points.size(); i++) {
            PointT &pt = v.scene_->points[i];
            if( pt.z > param_.chop_z_) {
                pt.z = pt.x = pt.y = std::numeric_limits<float>::quiet_NaN();// keep it organized
                pt.r = pt.g = pt.b = 0.f;
            }
        }
    }
    else {
//        v.scene_f_ = v.scene_;
        scene_normals_f = v.scene_normals_;
    }


    if (param_.compute_mst_) {
        if( param_.scene_to_scene_) {   // compute SIFT keypoints for the scene (since neighborhood of keypoint
                                        // matters for their SIFT descriptors, the descriptors are computed on the
                                        // original rather than on the filtered point cloud. Keypoints at infinity
                                        // are removed.
            std::vector<int> sift_kp_indices;
            std::vector<std::vector<float> > sift_signatures;
            calcSiftFeatures( v.scene_, sift_kp_indices, sift_signatures);

            v.sift_kp_indices_.reserve( sift_kp_indices.size() );
            v.sift_signatures_.reserve( sift_signatures.size() );
//            v.sift_keypoints_scales_.reserve( sift_keypoints_scales.size() );
            size_t kept=0;
            for (size_t i=0; i<sift_kp_indices.size(); i++) {   // remove infinte keypoints
                if ( pcl::isFinite( v.scene_->points[sift_kp_indices[i]] ) ) {
                    v.sift_kp_indices_.push_back( sift_kp_indices[i] );
                    v.sift_signatures_.push_back( sift_signatures[i] );
//                    v.sift_keypoints_scales_.push_back( sift_keypoints_scales[i] );
                    kept++;
                }
            }
            v.sift_kp_indices_.shrink_to_fit();
            v.sift_signatures_.shrink_to_fit();
//            v.sift_keypoints_scales_.shrink_to_fit();
            std::cout << "keypoints: " << v.sift_kp_indices_.size() << std::endl;

            // In addition to matching views, we can use the computed SIFT features for recognition
            rr_->setFeatAndKeypoints(v.sift_signatures_, v.sift_kp_indices_, SIFT_GPU);
            rr_->setFeatAndKeypoints(v.sift_signatures_, v.sift_kp_indices_, SIFT_OPENCV);
        }


        //=====================Pose Estimation=======================
        typename std::map<size_t, View<PointT> >::iterator v_it;
        for (v_it = views_.begin(); v_it != views_.end(); ++v_it) {
            const View<PointT> &w = v_it->second;
            if( w.id_ ==  v.id_ )
                continue;

            std::vector<CamConnect> transforms;
            CamConnect edge;
            edge.source_id_ = v.id_;
            edge.target_id_ = w.id_;

            if(param_.scene_to_scene_) {
                edge.model_name_ = "sift_background_matching";

                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > sift_transforms =
                        Registration::FeatureBasedRegistration<PointT>::estimateViewTransformationBySIFT(
                            *w.scene_, *v.scene_, w.sift_kp_indices_, v.sift_kp_indices_, w.sift_signatures_, v.sift_signatures_, param_.use_gc_s2s_);

                for(size_t sift_tf_id = 0; sift_tf_id < sift_transforms.size(); sift_tf_id++) {
                    edge.transformation_ = sift_transforms[sift_tf_id];
                    transforms.push_back(edge);
                }
            }

            if (param_.use_robot_pose_) {
                edge.model_name_ = "given_pose";
                Eigen::Matrix4f tf2wco_src = w.transform_to_world_co_system_;
                Eigen::Matrix4f tf2wco_trgt = v.transform_to_world_co_system_;
                edge.transformation_ = tf2wco_trgt.inverse() * tf2wco_src;
                transforms.push_back ( edge );
            }

            if( !transforms.empty() ) {
                size_t best_transform_id = 0;
                float lowest_edge_weight = std::numeric_limits<float>::max();

                for ( size_t trans_id = 0; trans_id < transforms.size(); trans_id++ ) {
                    CamConnect &e_tmp = transforms[trans_id];

                    try {
                        Eigen::Matrix4f icp_refined_trans;
                        calcEdgeWeightAndRefineTf<PointT>( w.scene_, v.scene_, e_tmp.transformation_, e_tmp.edge_weight_, icp_refined_trans);

                        e_tmp.transformation_ = icp_refined_trans;
                        std::cout << "Edge weight is " << e_tmp.edge_weight_ << " for edge connecting vertex " <<
                                     e_tmp.source_id_ << " and " << e_tmp.target_id_ << " by " <<  e_tmp.model_name_ ;

                        if(e_tmp.edge_weight_ < lowest_edge_weight) {
                            lowest_edge_weight = e_tmp.edge_weight_;
                            best_transform_id = trans_id;
                        }
                    }
                    catch (int e) {
                        e_tmp.edge_weight_ = std::numeric_limits<float>::max();
                        std::cerr << "Something is wrong with the SIFT based camera pose estimation. Turning it off and using the given camera poses only." << std::endl;
                        continue;
                    }
                }


                bool found = false;
                ViewD target_d;
                std::pair<vertex_iter, vertex_iter> vp;
                for (vp = vertices(gs_); vp.first != vp.second; ++vp.first) {
                    if (gs_[*vp.first] == transforms[best_transform_id].target_id_) {
                        target_d = *vp.first;
                        found = true;
                        break;
                    }
                }

                if(!found) {
                    target_d = boost::add_vertex (gs_);
                    gs_[target_d] = transforms[best_transform_id].target_id_;
                }

                ViewD src_D = boost::add_vertex (gs_);
                gs_[src_D] = transforms[best_transform_id].source_id_;
                boost::add_edge ( src_D, target_d, transforms[best_transform_id], gs_);
            }
        }

        boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, gs_);
        std::vector < EdgeD > spanning_tree;
        boost::kruskal_minimum_spanning_tree(gs_, std::back_inserter(spanning_tree));

        std::cout << "Print the edges in the MST:" << std::endl;

        for (auto & view : views_)
            view.second.has_been_hopped_ = false;

        bool is_first_edge = true;
        std::vector<CamConnect> loose_edges;

        for (std::vector < EdgeD >::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei) {
            CamConnect e = weightmap[*ei];
            if ( !computeAbsolutePose(e, is_first_edge) )
                loose_edges.push_back(e);

            is_first_edge = false;
        }

        while(loose_edges.size()) {
            for (size_t i=0; i <loose_edges.size(); i++) {
                if ( computeAbsolutePose( loose_edges[i], is_first_edge ) )
                    loose_edges.erase(loose_edges.begin() + i);
            }
        }
    }

//    if(views_.size()>1) {
//        typename pcl::PointCloud<PointT>::Ptr registration_check (new pcl::PointCloud<PointT>);
//        typename std::map<size_t, View<PointT> >::const_iterator view_it;
//        for (view_it = views_.begin(); view_it != views_.end(); ++view_it) {   // merge feature correspondences
//            const View<PointT> &w = view_it->second;
//            typename pcl::PointCloud<PointT>::Ptr cloud_tmp (new pcl::PointCloud<PointT>);
//            pcl::transformPointCloud(*w.scene_, *cloud_tmp, w.absolute_pose_);
//            *registration_check += *cloud_tmp;
//        }
//        pcl::visualization::PCLVisualizer registration_vis("registration_check");
//        registration_vis.addPointCloud(registration_check);
//        registration_vis.spin();
//    }

    rr_->setInputCloud(v.scene_);
    rr_->setSceneNormals(v.scene_normals_);
    rr_->recognize();

    if(true) {  // we have to do the correspondence grouping ourselve [Faeulhammer et al 2015, ICRA paper]
        rr_->getSavedHypotheses(v.hypotheses_);
        rr_->getKeypointCloud(v.scene_kp_);
        rr_->getKeyPointNormals(v.scene_kp_normals_);

        scene_keypoints_ = *v.scene_kp_;
        scene_kp_normals_ = *v.scene_kp_normals_;
        local_obj_hypotheses_ = v.hypotheses_;

        for (const auto &w_m : views_) {   // merge feature correspondences from other views
            const View<PointT> &w = w_m.second;
            if (w.id_ == v.id_)
                continue;

            //------ Transform keypoints and rotate normals----------
            Eigen::Matrix4f w_tf  = v.absolute_pose_.inverse() * w.absolute_pose_;
            typename pcl::PointCloud<PointT> scene_kp_aligned_;
            pcl::PointCloud<pcl::Normal> scene_kp_normals;
            pcl::transformPointCloud(*w.scene_kp_, scene_kp_aligned_, w_tf);
            transformNormals(*w.scene_kp_normals_, scene_kp_normals, w_tf);

            for (const auto &oh_remote_m : w.hypotheses_) {
                LocalObjectHypothesis<PointT> oh_remote = oh_remote_m.second; // copy because we need to update correspondences indices and don't change the original information

                // check if correspondences for model already exist
                const std::string &model_name = oh_remote.model_->id_;
                typename symHyp::iterator oh_local_m = local_obj_hypotheses_.find( model_name );
                if( oh_local_m == local_obj_hypotheses_.end() )   // no feature correspondences exist yet
                    local_obj_hypotheses_.insert(std::pair<std::string, LocalObjectHypothesis<PointT> >(model_name, oh_remote));

                else {  // merge with existing object hypotheses
                    LocalObjectHypothesis<PointT> &oh_local = oh_local_m->second;

                    // check each new correspondence for duplicate in existing database. If there is a sufficient close keypoint (Euclidean distance and normal dot product), do not add another one
                    size_t new_corr = oh_remote.model_scene_corresp_.size();
                    std::vector<bool> is_kept(new_corr, true);

                    #pragma omp parallel for
                    for (size_t c_id=0; c_id<new_corr; c_id++)  {
                        const pcl::Correspondence &c_new = oh_remote.model_scene_corresp_[c_id];
                        const PointT &m_kp_new = oh_remote.model_->keypoints_->points[ c_new.index_query ];
                        const PointT &s_kp_new = scene_kp_aligned_.points[ c_new.index_match ];
                        const pcl::Normal &s_kp_normal_new = scene_kp_normals.points[ c_new.index_match ];

                        for(const pcl::Correspondence &c_existing : oh_local.model_scene_corresp_) {
                            const PointT &m_kp_existing = oh_local.model_->keypoints_->points[ c_existing.index_query ];
                            const PointT &s_kp_existing = scene_kp_aligned_.points[ c_existing.index_match ];
                            const pcl::Normal &s_kp_normal_existing = scene_kp_normals.points[ c_existing.index_match ];

                            float squaredDistModelKeypoints = pcl::squaredEuclideanDistance(m_kp_new, m_kp_existing);
                            float squaredDistSceneKeypoints = pcl::squaredEuclideanDistance(s_kp_new, s_kp_existing);

                            if( (squaredDistModelKeypoints < param_.distance_same_keypoint_) &&
                                (squaredDistSceneKeypoints < param_.distance_same_keypoint_) &&
                                (s_kp_normal_new.getNormalVector3fMap().dot(s_kp_normal_existing.getNormalVector3fMap()) > param_.same_keypoint_dot_product_) ) {

                                is_kept[c_id] = false;
                                break;
                            }
                        }
                    }

                    size_t kept = 0;
                    for(size_t c_id=0; c_id<oh_remote.model_scene_corresp_.size(); c_id++) {
                        if(is_kept[c_id])
                            oh_remote.model_scene_corresp_[kept++] = oh_remote.model_scene_corresp_[c_id];
                    }
                    oh_remote.model_scene_corresp_.resize(kept);

                    for (auto &corr : oh_remote.model_scene_corresp_)  // add appropriate offset to correspondence index of the scene cloud
                            corr.index_match += scene_keypoints_.points.size();
                }
            }
            scene_keypoints_ += scene_kp_aligned_;
            scene_kp_normals_ += scene_kp_normals;
        }

//        pcl::visualization::PCLVisualizer vis_tmp;
//        pcl::visualization::PointCloudColorHandlerRGBField<PointT> handler (accum_scene);
//        vis_tmp.addPointCloud<PointT> (accum_scene, handler, "cloud wo normals");
//        vis_tmp.addPointCloudNormals<PointT,pcl::Normal>(accum_scene, accum_normals, 10);
//        vis_tmp.spin();

        if(cg_algorithm_) {
            verified_hypotheses_.clear();
            correspondenceGrouping();
            v.models_ = models_;
            v.transforms_ = transforms_;
            v.origin_view_id_.resize(models_.size());
            std::fill(v.origin_view_id_.begin(), v.origin_view_id_.end(), v.id_);
            v.model_or_plane_is_verified_.resize(models_.size());
            std::fill(v.model_or_plane_is_verified_.begin(), v.model_or_plane_is_verified_.end(), false);
        }

//        for(size_t m_id=0; m_id<transforms_.size(); m_id++) // transform hypotheses back from global coordinate system to current viewport
//                transforms_[m_id] = v.absolute_pose_.inverse() * transforms_[m_id];
    }
    else {  // correspondence grouping is done already (so we get the full models) [Faeulhammer et al 2015, MVA paper]
        v.models_ = rr_->getModels();
        v.transforms_ = rr_->getTransforms ();
        v.origin_view_id_.resize(v.models_.size());
        std::fill(v.origin_view_id_.begin(), v.origin_view_id_.end(), v.id_);
        v.model_or_plane_is_verified_.resize(v.models_.size());
        std::fill(v.model_or_plane_is_verified_.begin(), v.model_or_plane_is_verified_.end(), false);

        typename std::map<size_t, View<PointT> >::const_iterator view_it;
        for (view_it = views_.begin(); view_it != views_.end(); ++view_it) {   // add hypotheses from other views
            const View<PointT> &w = view_it->second;
            if (w.id_ == v.id_)
                continue;

            for(size_t i=0; i<w.models_.size(); i++) {
                if(w.model_or_plane_is_verified_[i]) {

                    bool add_hypothesis = true;

                    if(param_.merge_close_hypotheses_) {  // check if similar object hypotheses exist
                        Eigen::Matrix4f object_pose_in_v_co_system = v.absolute_pose_.inverse() * w.absolute_pose_ * w.transforms_[i];
                        Eigen::Vector3f centroid_w = object_pose_in_v_co_system.block<3, 1> (0, 3);
                        Eigen::Matrix3f rot_w = object_pose_in_v_co_system.block<3, 3> (0, 0);
                        const double angle_thresh_rad = param_.merge_close_hypotheses_angle_ * M_PI / 180.f ;

                        for(size_t j=0; j<v.models_.size(); j++) {
                            if(w.models_[i]->id_ != v.models_[j]->id_)
                                continue;

                            const Eigen::Matrix4f tf_v = v.transforms_[j];
                            Eigen::Vector3f centroid_v = tf_v.block<3, 1> (0, 3);
                            Eigen::Matrix3f rot_v = tf_v.block<3, 3> (0, 0);
                            Eigen::Matrix3f rot_diff = rot_v * rot_w.transpose();

                            double rotx = atan2(rot_diff(2,1), rot_diff(2,2));
                            double roty = atan2(-rot_diff(2,0), sqrt(rot_diff(2,1) * rot_diff(2,1) + rot_diff(2,2) * rot_diff(2,2)));
                            double rotz = atan2(rot_diff(1,0), rot_diff(0,0));
                            double dist = (centroid_v - centroid_w).norm();

                            if ( (dist < param_.merge_close_hypotheses_dist_) && (rotx < angle_thresh_rad) && (roty < angle_thresh_rad) && (rotz < angle_thresh_rad) ) {
                                add_hypothesis = false;
                                break;
                            }
                        }
                    }
                    if(add_hypothesis) {
                        v.models_.push_back( w.models_[i] );
                        v.transforms_.push_back( v.absolute_pose_.inverse() * w.absolute_pose_ * w.transforms_[i] );
                        v.origin_view_id_.push_back( w.origin_view_id_[i] );
                        v.model_or_plane_is_verified_.push_back( false );
                    }
                }
            }
        }

        models_ = v.models_;
        transforms_ = v.transforms_;
    }

     {
        initHVFilters();

        NguyenNoiseModel<PointT> nm (nm_param_);
        nm.setInputCloud(v.scene_);
        nm.setInputNormals( v.scene_normals_);
        nm.compute();
        v.pt_properties_ = nm.getPointProperties();

        std::vector<typename pcl::PointCloud<PointT>::Ptr> original_clouds (views_.size());
        std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normal_clouds (views_.size());
        std::vector< Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>  > transforms_to_global  (views_.size());
        std::vector<std::vector<std::vector<float> > > pt_properties (views_.size());

        size_t view_id = 0;
        for (const auto &w_m : views_) {
            const View<PointT> &w = w_m.second;
            original_clouds [view_id ] = w.scene_;
            normal_clouds [view_id] = w.scene_normals_;
            transforms_to_global [view_id] = v.absolute_pose_.inverse() * w.absolute_pose_;
            pt_properties [view_id ] = w.pt_properties_;
            view_id++;

            if (param_.run_reconstruction_filter_) {
                reconstructionFiltering(original_clouds [view_id ], normal_clouds [view_id],
                        w.absolute_pose_, w_m.first);
                NguyenNoiseModel<PointT> nm_again(nm_param_);
                nm_again.setInputCloud(original_clouds [view_id ]);
                nm_again.setInputNormals(normal_clouds [view_id]);
                nm_again.compute();
                pt_properties [view_id ] = nm_again.getPointProperties();
            }
        }

        std::vector<typename pcl::PointCloud<PointT>::ConstPtr> occlusion_clouds (original_clouds.size());
        for(size_t i=0; i < original_clouds.size(); i++)
            occlusion_clouds[i].reset(new pcl::PointCloud<PointT>(*original_clouds[i]));

        hv_algorithm_->setOcclusionCloudsAndAbsoluteCameraPoses(occlusion_clouds, transforms_to_global );

   if (views_.size() > 1 ) { // don't do this if there is only one view otherwise point cloud is not kept organized and multi-plane segmentation takes longer
            //obtain big cloud and occlusion clouds based on noise model integration
            typename pcl::PointCloud<PointT>::Ptr octree_cloud(new pcl::PointCloud<PointT>);
            NMBasedCloudIntegration<PointT> nmIntegration (nmInt_param_);
            nmIntegration.setInputClouds(original_clouds);
            nmIntegration.setTransformations(transforms_to_global);
            nmIntegration.setInputNormals(normal_clouds);
            nmIntegration.setPointProperties(pt_properties);
            nmIntegration.compute(octree_cloud);
            pcl::PointCloud<pcl::Normal>::Ptr big_normals(new pcl::PointCloud<pcl::Normal>);
            nmIntegration.getOutputNormals(big_normals);
            scene_ = octree_cloud;
            scene_normals_ = big_normals;
        }
        else {
            scene_ = v.scene_;
            scene_normals_ = v.scene_normals_;
        }
    }

    if ( hv_algorithm_ && !models_.empty() ) {
        if( !param_.use_multiview_verification_) {
            scene_ = v.scene_;
            scene_normals_ = v.scene_normals_;
        }

        if(param_.run_hypotheses_filter_) {
            std::vector< std::vector<bool> > hypotheses_in_views(models_.size());
            for(int mi = 0; mi < models_.size(); mi++) {
                    Eigen::Matrix4f h_pose = v.absolute_pose_ * v.transforms_[mi];
                    hypotheses_in_views[mi] = getHypothesisInViewsMask(models_[mi], h_pose, v.origin_view_id_[mi]);
            }
            hv_algorithm_->setVisibleCloudsForModels(hypotheses_in_views);
        }

        hypothesisVerification();
        v.model_or_plane_is_verified_ = hypothesis_is_verified_;
        v.transforms_ = transforms_; // save refined pose
    }

    scene_normals_.reset();

    pruneGraph();
    cleanupHVFilters();
    id_++;
}

template<typename PointT>
void
MultiviewRecognizer<PointT>::correspondenceGrouping ()
{
    double t_start = omp_get_wtime();

    std::vector<LocalObjectHypothesis<PointT> > ohs(local_obj_hypotheses_.size());

    size_t id=0;
    typename std::map<std::string, LocalObjectHypothesis<PointT> >::const_iterator it;
    for (it = local_obj_hypotheses_.begin (), id=0; it != local_obj_hypotheses_.end (); ++it)
        ohs[id++] = it->second;

#pragma omp parallel for schedule(dynamic)
    for (size_t i=0; i<ohs.size(); i++)
    {
        const LocalObjectHypothesis<PointT> &oh = ohs[i];

        if(oh.model_scene_corresp_.size() < 3)
            continue;

        GraphGeometricConsistencyGrouping<PointT, PointT> cg = *cg_algorithm_;

        std::vector < pcl::Correspondences > corresp_clusters;
        cg.setSceneCloud (scene_keypoints_.makeShared());
        cg.setInputCloud (oh.model_->keypoints_);

        if(cg.getRequiresNormals())
            cg.setInputAndSceneNormals(oh.model_->kp_normals_, scene_kp_normals_.makeShared());

        //we need to pass the keypoints_pointcloud and the specific object hypothesis
        cg.setModelSceneCorrespondences (oh.model_scene_corresp_);
        cg.cluster (corresp_clusters);

        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > new_transforms (corresp_clusters.size());
        typename pcl::registration::TransformationEstimationSVD < PointT, PointT > t_est;

        for (size_t cluster_id = 0; cluster_id < corresp_clusters.size(); cluster_id++)
            t_est.estimateRigidTransformation (*oh.model_->keypoints_, scene_keypoints_, corresp_clusters[cluster_id], new_transforms[cluster_id]);

        if(param_.merge_close_hypotheses_) {
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > merged_transforms (corresp_clusters.size());
            std::vector<bool> cluster_has_been_taken(corresp_clusters.size(), false);
            const double angle_thresh_rad = param_.merge_close_hypotheses_angle_ * M_PI / 180.f ;

            size_t kept=0;
            for (size_t tf_id = 0; tf_id < new_transforms.size(); tf_id++) {

                if (cluster_has_been_taken[tf_id])
                    continue;

                cluster_has_been_taken[tf_id] = true;
                const Eigen::Vector3f centroid1 = new_transforms[tf_id].block<3, 1> (0, 3);
                const Eigen::Matrix3f rot1 = new_transforms[tf_id].block<3, 3> (0, 0);

                pcl::Correspondences merged_corrs = corresp_clusters[tf_id];

                for(size_t j=tf_id; j < new_transforms.size(); j++) {
                    const Eigen::Vector3f centroid2 = new_transforms[j].block<3, 1> (0, 3);
                    const Eigen::Matrix3f rot2 = new_transforms[j].block<3, 3> (0, 0);
                    const Eigen::Matrix3f rot_diff = rot2 * rot1.transpose();

                    double rotx = atan2(rot_diff(2,1), rot_diff(2,2));
                    double roty = atan2(-rot_diff(2,0), sqrt(rot_diff(2,1) * rot_diff(2,1) + rot_diff(2,2) * rot_diff(2,2)));
                    double rotz = atan2(rot_diff(1,0), rot_diff(0,0));
                    double dist = (centroid1 - centroid2).norm();

                    if ( (dist < param_.merge_close_hypotheses_dist_) && (rotx < angle_thresh_rad) && (roty < angle_thresh_rad) && (rotz < angle_thresh_rad) ) {
                        merged_corrs.insert( merged_corrs.end(), corresp_clusters[j].begin(), corresp_clusters[j].end() );
                        cluster_has_been_taken[j] = true;
                    }
                }

                t_est.estimateRigidTransformation (*oh.model_->keypoints_, scene_keypoints_, merged_corrs, merged_transforms[kept]);
                kept++;
            }
            merged_transforms.resize(kept);

#pragma omp critical
            {
                transforms_.insert(transforms_.end(), merged_transforms.begin(), merged_transforms.end());
                models_.resize( transforms_.size(), oh.model_ );
            }
        }

        std::cout << "Merged " << corresp_clusters.size() << " clusters into " << new_transforms.size() << " clusters. Total correspondences: " << oh.model_scene_corresp_.size () << " " << oh.model_->id_ << std::endl;

        //        oh.visualize(*scene_);
    }

    double t_stop = omp_get_wtime();

    std::cout << "Correspondence Grouping took " <<  t_stop - t_start << std::endl;
}

}
