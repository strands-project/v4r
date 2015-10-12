#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/noise_model_based_cloud_integration.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/recognition/hv_go_3D.h>
#include <v4r/registration/fast_icp_with_gc.h>
#include <v4r/recognition/multiview_object_recognizer.h>
#include <v4r/recognition/segmenter.h>
#include <v4r/segmentation/multiplane_segmentation.h>
#include <v4r/segmentation/segmentation_utils.h>

#include <boost/graph/kruskal_min_spanning_tree.hpp>

namespace v4r
{

template<typename PointT>
bool
MultiviewRecognizer<PointT>::calcSiftFeatures (const typename pcl::PointCloud<PointT>::Ptr &cloud_src,
                                               typename pcl::PointCloud<PointT>::Ptr &sift_keypoints,
                                               std::vector< int > &sift_keypoint_indices,
                                               pcl::PointCloud<FeatureT>::Ptr &sift_signatures,
                                               std::vector<float> &sift_keypoint_scales)
{
    pcl::PointIndices sift_keypoint_pcl_indices;

    if(!sift_signatures)
        sift_signatures.reset(new pcl::PointCloud<FeatureT>);

    if(!sift_keypoints)
        sift_keypoints.reset(new pcl::PointCloud<PointT>);

#ifdef USE_SIFT_GPU
    SIFTLocalEstimation<PointT, FeatureT> estimator(sift_);
    bool ret = estimator.estimate (cloud_src, sift_keypoints, sift_signatures, sift_keypoint_scales);
#else
    (void)sift_keypoint_scales; //silences compiler warning of unused variable
    pcl::PointCloud<PointT>::Ptr processed_foo (new pcl::PointCloud<PointT>());

    OpenCVSIFTLocalEstimation<PointT, FeatureT > estimator;
    bool ret = estimator.estimate (cloud_src, processed_foo, sift_keypoints, sift_signatures);
#endif
    estimator.getKeypointIndices( sift_keypoint_pcl_indices );
    sift_keypoint_indices = sift_keypoint_pcl_indices.indices;
    return ret;
}



template<typename PointT>
void
MultiviewRecognizer<PointT>::estimateViewTransformationBySIFT(const pcl::PointCloud<PointT> &src_cloud,
                                                              const pcl::PointCloud<PointT> &dst_cloud,
                                                              const std::vector<int> &src_sift_keypoint_indices,
                                                              const std::vector<int> &dst_sift_keypoint_indices,
                                                              const pcl::PointCloud<FeatureT> &src_sift_signatures,
                                                              boost::shared_ptr< flann::Index<DistT> > &dst_flann_index,
                                                              std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transformations,
                                                              bool use_gc )
{
    const int K = 1;
    flann::Matrix<int> indices = flann::Matrix<int> ( new int[K], 1, K );
    flann::Matrix<float> distances = flann::Matrix<float> ( new float[K], 1, K );

    boost::shared_ptr< pcl::PointCloud<PointT> > pSiftKeypointsSrc (new pcl::PointCloud<PointT>);
    boost::shared_ptr< pcl::PointCloud<PointT> > pSiftKeypointsDst (new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(src_cloud, src_sift_keypoint_indices, *pSiftKeypointsSrc );
    pcl::copyPointCloud(dst_cloud, dst_sift_keypoint_indices, *pSiftKeypointsDst);

    pcl::CorrespondencesPtr temp_correspondences ( new pcl::Correspondences );
    temp_correspondences->resize(pSiftKeypointsSrc->size ());

    for ( size_t keypointId = 0; keypointId < pSiftKeypointsSrc->points.size (); keypointId++ )
    {
        FeatureT searchFeature = src_sift_signatures[ keypointId ];
        int size_feat = sizeof ( searchFeature.histogram ) / sizeof ( float );
        v4r::nearestKSearch ( dst_flann_index, searchFeature.histogram, size_feat, K, indices, distances );

        pcl::Correspondence corr;
        corr.distance = distances[0][0];
        corr.index_query = keypointId;
        corr.index_match = indices[0][0];
        temp_correspondences->at(keypointId) = corr;
    }

    if(!use_gc)
    {
        typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointT>::Ptr rej;
        rej.reset (new pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ());
        pcl::CorrespondencesPtr after_rej_correspondences (new pcl::Correspondences ());

        rej->setMaximumIterations (50000);
        rej->setInlierThreshold (0.02);
        rej->setInputTarget (pSiftKeypointsDst);
        rej->setInputSource (pSiftKeypointsSrc);
        rej->setInputCorrespondences (temp_correspondences);
        rej->getCorrespondences (*after_rej_correspondences);

        Eigen::Matrix4f refined_pose;
        transformations.push_back( rej->getBestTransformation () );
        pcl::registration::TransformationEstimationSVD<PointT, PointT> t_est;
        t_est.estimateRigidTransformation (*pSiftKeypointsSrc, *pSiftKeypointsDst, *after_rej_correspondences, refined_pose);
        transformations.back() = refined_pose;
    }
    else
    {
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > new_transforms;
        pcl::GeometricConsistencyGrouping<pcl::PointXYZRGB, pcl::PointXYZRGB> gcg_alg;

        gcg_alg.setGCThreshold (15);
        gcg_alg.setGCSize (0.01);
        gcg_alg.setInputCloud(pSiftKeypointsSrc);
        gcg_alg.setSceneCloud(pSiftKeypointsDst);
        gcg_alg.setModelSceneCorrespondences(temp_correspondences);

        std::vector<pcl::Correspondences> clustered_corrs;
        gcg_alg.recognize(new_transforms, clustered_corrs);
        transformations.insert(transformations.end(), new_transforms.begin(), new_transforms.end());
    }
}

template<typename PointT>
float
MultiviewRecognizer<PointT>::calcEdgeWeightAndRefineTf (const typename pcl::PointCloud<PointT>::ConstPtr &cloud_src,
                                                        const typename pcl::PointCloud<PointT>::ConstPtr &cloud_dst,
                                                        Eigen::Matrix4f &refined_transform,
                                                        const Eigen::Matrix4f &transform)
{
    typename pcl::PointCloud<PointT>::Ptr cloud_src_wo_nan ( new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr cloud_dst_wo_nan ( new pcl::PointCloud<PointT>());

    pcl::PassThrough<PointT> pass;
    pass.setFilterLimits (0.f, 5.f);
    pass.setFilterFieldName ("z");
    pass.setInputCloud (cloud_src);
    pass.setKeepOrganized (true);
    pass.filter (*cloud_src_wo_nan);

    pcl::PassThrough<PointT> pass2;
    pass2.setFilterLimits (0.f, 5.f);
    pass2.setFilterFieldName ("z");
    pass2.setInputCloud (cloud_dst);
    pass2.setKeepOrganized (true);
    pass2.filter (*cloud_dst_wo_nan);

    float w_after_icp_ = std::numeric_limits<float>::max ();
    const float best_overlap_ = 0.75f;

    v4r::FastIterativeClosestPointWithGC<pcl::PointXYZRGB> icp;
    icp.setMaxCorrespondenceDistance ( 0.02f );
    icp.setInputSource ( cloud_src_wo_nan );
    icp.setInputTarget ( cloud_dst_wo_nan );
    icp.setUseNormals (true);
    icp.useStandardCG (true);
    icp.setNoCG(true);
    icp.setOverlapPercentage (best_overlap_);
    icp.setKeepMaxHypotheses (5);
    icp.setMaximumIterations (10);
    icp.align (transform);
    w_after_icp_ = icp.getFinalTransformation ( refined_transform );

    if ( w_after_icp_ < 0 || !pcl_isfinite ( w_after_icp_ ) )
        w_after_icp_ = std::numeric_limits<float>::max ();
    else
        w_after_icp_ = best_overlap_ - w_after_icp_;

    //    transform = icp_trans; // refined transformation
    return w_after_icp_;
}


template<typename PointT>
bool
MultiviewRecognizer<PointT>::computeAbsolutePosesRecursive (const Graph & grph,
                              const ViewD start,
                              const Eigen::Matrix4f &accum,
                              std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & absolute_poses,
                              std::vector<bool> &hop_list)
{
    boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, gs_);
    boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
    for (boost::tie (ei, ei_end) = boost::out_edges (start, grph); ei != ei_end; ++ei)  {
        ViewD targ = boost::target (*ei, grph);
        size_t target_id = boost::target (*ei, grph);

        if(hop_list[target_id])
           continue;

        hop_list[target_id] = true;
        CamConnect my_e = weightmap[*ei];
        Eigen::Matrix4f intern_accum;
        Eigen::Matrix4f trans = my_e.transformation_;
        if( my_e.target_id_ != target_id) {
            Eigen::Matrix4f trans_inv;
            trans_inv = trans.inverse();
            trans = trans_inv;
        }
        intern_accum = accum * trans;
        absolute_poses[target_id] = intern_accum;
        computeAbsolutePosesRecursive (grph, targ, intern_accum, absolute_poses, hop_list);
    }

    return true;
}


template<typename PointT>
bool
MultiviewRecognizer<PointT>::computeAbsolutePoses (const Graph & grph, std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & absolute_poses)
{
  size_t num_frames = boost::num_vertices(grph);
  if (num_frames == 0) {
      absolute_poses.push_back( views_[0].transform_to_world_co_system_ );
      return false;
  }

  absolute_poses.resize( num_frames );
  std::vector<bool> hop_list (num_frames, false);
  ViewD source_view = 0;
  hop_list[0] = true;
  Eigen::Matrix4f accum = views_[0].transform_to_world_co_system_;      // CAN IT ALSO BE grph[0] instead of class member?
  absolute_poses[0] = accum;
  return computeAbsolutePosesRecursive (grph, source_view, accum, absolute_poses, hop_list);
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

    size_t num_existing_views = views_.size();
    views_.resize(num_existing_views + 1);
    View<PointT> &v = views_.back();

    v.id_ = id_++;
    v.scene_ = scene_;
    v.transform_to_world_co_system_ = pose_;
    v.absolute_pose_ = pose_;    // this might be redundant

    computeNormals<PointT>(v.scene_, v.scene_normals_, param_.normal_computation_method_);

    if( param_.chop_z_ > 0) {
        pcl::PassThrough<PointT> pass;
        pass.setFilterLimits ( 0.f, param_.chop_z_ );
        pass.setFilterFieldName ("z");
        pass.setInputCloud (v.scene_);
        pass.setKeepOrganized (true);
        pass.filter (*v.scene_f_);
        v.filtered_scene_indices_.indices = *pass.getIndices();
        pcl::copyPointCloud(*v.scene_normals_, v.filtered_scene_indices_, *scene_normals_f);
    }
    else {
        v.scene_f_ = v.scene_;
        scene_normals_f = v.scene_normals_;
    }


    if( param_.scene_to_scene_) {
        typename pcl::PointCloud<PointT>::Ptr sift_keypoints (new pcl::PointCloud<PointT>());
        calcSiftFeatures( v.scene_, sift_keypoints, v.sift_kp_indices_.indices, v.sift_signatures_, v.sift_keypoints_scales_);
        std::cout << "keypoints: " << v.sift_kp_indices_.indices.size() << std::endl;

        // In addition to matching views, we can use the computed SIFT features for recognition
//        rr_->setFeatAndKeypoints<flann::L1, FeatureT > (v.pSiftSignatures_, v.siftKeypointIndices_, SIFT);
    }


    //=====================Pose Estimation=======================
    for (size_t view_id=0; view_id<views_.size(); view_id++) {
        const View<PointT> &w = views_[view_id];
        if( w.id_ ==  v.id_ )
            continue;

        std::vector<CamConnect> transforms;
        CamConnect edge;
        edge.source_id_ = v.id_;
        edge.target_id_ = w.id_;

        if(param_.scene_to_scene_) {
            edge.model_name_ = "sift_background_matching";

            boost::shared_ptr< flann::Index<DistT> > flann_index;
            convertToFLANN<FeatureT, DistT >( v.sift_signatures_, flann_index );

            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > sift_transforms;
            estimateViewTransformationBySIFT( *w.scene_f_, *v.scene_f_, w.sift_kp_indices_.indices, v.sift_kp_indices_.indices, *w.sift_signatures_, flann_index, sift_transforms);

            for(size_t sift_tf_id = 0; sift_tf_id < sift_transforms.size(); sift_tf_id++) {
                edge.transformation_ = sift_transforms[sift_tf_id];
                transforms.push_back(edge);
            }
        }

        if (param_.use_robot_pose_) {
            edge.model_name_ = "given_pose";
            Eigen::Matrix4f tf2wco_src = w.transform_to_world_co_system_;
            Eigen::Matrix4f tf2wco_trgt = v.transform_to_world_co_system_;
            edge.transformation_ = tf2wco_src.inverse() * tf2wco_trgt;
            transforms.push_back ( edge );
        }

        if( transforms.size() ) {
            size_t best_transform_id = 0;
            float lowest_edge_weight = std::numeric_limits<float>::max();
            for ( size_t trans_id = 0; trans_id < transforms.size(); trans_id++ ) {
                CamConnect &e_tmp = transforms[trans_id];

                try {
                    Eigen::Matrix4f icp_refined_trans;
                    e_tmp.edge_weight_ = calcEdgeWeightAndRefineTf( w.scene_, v.scene_, icp_refined_trans, e_tmp.transformation_);
                    e_tmp.transformation_ = icp_refined_trans,
                    std::cout << "Edge weight is " << e_tmp.edge_weight_ << " for edge connecting vertex " <<
                                 e_tmp.source_id_ << " and " << e_tmp.target_id_ << " by " <<
                                 e_tmp.model_name_ << std::endl;

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
            boost::add_edge ( transforms[best_transform_id].source_id_, transforms[best_transform_id].target_id_, transforms[best_transform_id], gs_);
        }
    }

    boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, gs_);
    std::vector < EdgeD > spanning_tree;
    boost::kruskal_minimum_spanning_tree(gs_, std::back_inserter(spanning_tree));

    Graph grph_mst;
    std::cout << "Print the edges in the MST:" << std::endl;
    for (std::vector < EdgeD >::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei) {
        CamConnect my_e = weightmap[*ei];
        std::cout << "[" << source(*ei, gs_) << "->" << target(*ei, gs_) << "] with weight " << my_e.edge_weight_ << " by " << my_e.model_name_ << std::endl;
        boost::add_edge(source(*ei, gs_), target(*ei, gs_), weightmap[*ei], grph_mst);
    }

    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > absolute_poses;
    computeAbsolutePoses(gs_, absolute_poses);

    for(size_t v_id=0; v_id<absolute_poses.size(); v_id++)
        views_[ v_id ].absolute_pose_ = absolute_poses [ v_id ];


    rr_->setInputCloud(v.scene_f_);
    rr_->setSceneNormals(scene_normals_f);
    rr_->recognize();

    if(rr_->getSaveHypothesesParam()) {  // correspondence grouping is done already (so we get the full models) [Faeulhammer et al 2015, MVA paper]
        rr_->getSavedHypotheses(v.hypotheses_);
        rr_->getKeypointCloud(v.pKeypointsMultipipe_);
        rr_->getKeypointIndices(v.kp_indices_);
        pcl::copyPointCloud(*scene_normals_f, v.kp_indices_, *v.kp_normals_);

        obj_hypotheses_ = v.hypotheses_;

        typename pcl::PointCloud<PointT>::Ptr accum_scene (new pcl::PointCloud<PointT>());
        pcl::PointCloud<pcl::Normal>::Ptr accum_normals (new pcl::PointCloud<pcl::Normal>());
        *accum_scene += *v.scene_f_;
        *accum_normals += *scene_normals_f;

        size_t num_existing_scene_pts = v.scene_f_->points.size();

        for (size_t v_id=0; v_id<views_.size(); v_id++) {   // merge feature correspondences
            const View<PointT> &w = views_[v_id];
            if (w.id_ == v.id_)
                continue;


            //------ Transform keypoints and rotate normals----------
            typename pcl::PointCloud<PointT> cloud_aligned_tmp;
            pcl::transformPointCloud(*w.scene_f_, w.absolute_pose_, cloud_aligned_tmp);

            pcl::PointCloud<pcl::Normal> normal_aligned_tmp;
            normal_aligned_tmp.points.resize( w.filtered_scene_indices_.indices.size() );

            const Eigen::Matrix3f rot = w.absolute_pose_.block<3, 3> (0, 0);
            for (size_t n_id=0; n_id< w.filtered_scene_indices_.indices.size(); n_id++) {
                const pcl::Normal &n = w.scene_normals_[ filtered_scene_indices_.indices [n_id] ];
                normal_aligned_tmp.points[n_id].getNormalVector3fMap() = rot * n.getNormalVector3fMap ();
            }


            *accum_scene += cloud_aligned_tmp;
            *accum_normals += normal_aligned_tmp;

            typename std::map<std::string, ObjectHypothesis<PointT> >::iterator it_mp_oh;

            typename std::map<std::string, ObjectHypothesis<PointT> >::iterator it_tmp;
            for (it_tmp = w.hypotheses_.begin (); it_tmp != w.hypotheses_.end (); it_tmp++)
            {
                const std::string id = it_tmp->second.model_->id_;

                it_mp_oh = obj_hypotheses_.find(id);
                if(it_mp_oh == obj_hypotheses_.end())   // no feature correspondences exist yet
                    obj_hypotheses_.insert(std::pair<std::string, ObjectHypothesis<PointT> >(id, it_tmp->second));
                else
                {
                    ObjectHypothesis<PointT> &oh = it_mp_oh->second;
                    const ObjectHypothesis<PointT> &new_oh = it_tmp->second;
                    pcl::Correspondences model_scene_corresp_tmp = *new_oh.model_scene_corresp_;

                    size_t num_existing_corr = oh.model_scene_corresp_->size();
                    oh.model_scene_corresp_->resize( num_existing_corr + model_scene_corresp_tmp.size());
                    size_t kept=0;
                    for(size_t c_id=0; c_id< model_scene_corresp_tmp.size(); c_id++) {
                        if(0) {
                            oh.model_scene_corresp_->at( num_existing_corr + kept ) = new_oh.model_scene_corresp_->at(c_id);
                            kept++;
                        }
                    }
                    oh.model_scene_corresp_->resize(num_existing_corr + kept);
                }
            }
        }
    }
    else    // we have to do the correspondence grouping ourselve [Faeulhammer et al 2015, ICRA paper]
    {
    }
}

template<typename PointT>
void
MultiviewRecognizer<PointT>::savePCDwithPose()
{
    for (size_t view_id=0; view_id<views_.size(); view_id++) {
        setCloudPose(views_[view_id].absolute_pose_, *views_[view_id].scene_);
        std::stringstream fn; fn <<  views_[view_id].id_ + ".pcd";
        pcl::io::savePCDFileBinary(fn.str(), *views_[view_id].scene_);
    }
}
}
