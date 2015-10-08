#include <v4r/recognition/multiview_object_recognizer.h>
#include <v4r/recognition/segmenter.h>

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/segmentation/multiplane_segmentation.h>
#include <v4r/recognition/hv_go_3D.h>
#include <v4r/registration/fast_icp_with_gc.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/noise_model_based_cloud_integration.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/segmentation/segmentation_utils.h>
#include <v4r/common/miscellaneous.h>

namespace v4r
{
template<typename PointT>
bool
MultiviewRecognizer<PointT>::calcSiftFeatures (ViewD &src, MVGraph &grph)
{
    boost::shared_ptr< pcl::PointCloud<PointT> > pSiftKeypoints;
#ifdef USE_SIFT_GPU
    bool ret = estimator->estimate (grph[src].scene_f_, pSiftKeypoints, grph[src].pSiftSignatures_, grph[src].sift_keypoints_scales_);
    estimator->getKeypointIndices(grph[src].siftKeypointIndices_);
#else
    pcl::PointCloud<PointT>::Ptr processed_foo (new pcl::PointCloud<PointT>());
    bool ret = estimator->estimate (grph[src].pScenePCl_f, processed_foo, pSiftKeypoints, grph[src].pSiftSignatures_);
    estimator->getKeypointIndices( grph[src].siftKeypointIndices_ );
#endif
    return ret;
}

template<typename PointT>
void
MultiviewRecognizer<PointT>::estimateViewTransformationBySIFT (const ViewD &src, const ViewD &trgt, MVGraph &grph,
                                   boost::shared_ptr<flann::Index<flann::L1<float> > > flann_index,
                                   Eigen::Matrix4f &transformation,
                                   std::vector<EdgeD> & edges, bool use_gc )
{
    const int K = 1;
    flann::Matrix<int> indices = flann::Matrix<int> ( new int[K], 1, K );
    flann::Matrix<float> distances = flann::Matrix<float> ( new float[K], 1, K );

    boost::shared_ptr< pcl::PointCloud<PointT> > pSiftKeypointsSrc (new pcl::PointCloud<PointT>);
    boost::shared_ptr< pcl::PointCloud<PointT> > pSiftKeypointsTrgt (new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*(grph[ src].scene_f_), grph[ src].siftKeypointIndices_, *(pSiftKeypointsSrc ));
    pcl::copyPointCloud(*(grph[trgt].scene_f_), grph[trgt].siftKeypointIndices_, *(pSiftKeypointsTrgt));
    PCL_INFO ( "Calculate transform via SIFT between view %ud and %ud for a keypoint size of %ld (src) and %ld (target).",
               grph[src].id_, grph[trgt].id_, pSiftKeypointsSrc->points.size(), pSiftKeypointsTrgt->points.size() );

    pcl::CorrespondencesPtr temp_correspondences ( new pcl::Correspondences );
    temp_correspondences->resize(pSiftKeypointsSrc->size ());

    for ( size_t keypointId = 0; keypointId < pSiftKeypointsSrc->size (); keypointId++ )
    {
        FeatureT searchFeature = grph[src].pSiftSignatures_->at ( keypointId );
        int size_feat = sizeof ( searchFeature.histogram ) / sizeof ( float );
        nearestKSearch ( flann_index, searchFeature.histogram, size_feat, K, indices, distances );

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
        rej->setInputTarget (pSiftKeypointsTrgt);
        rej->setInputSource (pSiftKeypointsSrc);
        rej->setInputCorrespondences (temp_correspondences);
        rej->getCorrespondences (*after_rej_correspondences);

        transformation = rej->getBestTransformation ();
        pcl::registration::TransformationEstimationSVD<PointT, PointT> t_est;
        t_est.estimateRigidTransformation (*pSiftKeypointsSrc, *pSiftKeypointsTrgt, *after_rej_correspondences, transformation);

        bool b;
        EdgeD edge;
        tie (edge, b) = add_edge (trgt, src, grph);
        grph[edge].transformation = transformation;
        grph[edge].model_name = std::string ("scene_to_scene");
        grph[edge].source_id = grph[src].id_;
        grph[edge].target_id = grph[trgt].id_;
        edges.push_back(edge);
    }
    else
    {
        pcl::GeometricConsistencyGrouping<PointT, PointT> gcg_alg;
        gcg_alg.setGCThreshold (15);
        gcg_alg.setGCSize (0.01);
        gcg_alg.setInputCloud(pSiftKeypointsSrc);
        gcg_alg.setSceneCloud(pSiftKeypointsTrgt);
        gcg_alg.setModelSceneCorrespondences(temp_correspondences);

        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transformations;
        std::vector<pcl::Correspondences> clustered_corrs;
        gcg_alg.recognize(transformations, clustered_corrs);

        for(size_t i=0; i < transformations.size(); i++)
        {
            PointInTPtr transformed_PCl (new pcl::PointCloud<PointT>);
            pcl::transformPointCloud (*grph[src].pScenePCl, *transformed_PCl, transformations[i]);

            std::stringstream scene_stream;
            scene_stream << "scene_to_scene_cg_" << i;
            bool b;
            EdgeD edge;
            tie (edge, b) = add_edge (trgt, src, grph);
            grph[edge].transformation = transformations[i];
            grph[edge].model_name = scene_stream.str();
            grph[edge].source_id = grph[src].id_;
            grph[edge].target_id = grph[trgt].id_;
            edges.push_back(edge);
        }
    }
}

template<typename PointT>
void
MultiviewRecognizer<PointT>::extendFeatureMatchesRecursive ( MVGraph &grph,
                                ViewD &vrtx_start,
                                std::map < std::string,ObjectHypothesis<PointT> > &hypotheses,
                                typename pcl::PointCloud<PointT>::Ptr pKeypoints,
                                pcl::PointCloud<pcl::Normal>::Ptr pKeypointNormals)
{
    View &v = grph[vrtx_start];

    pcl::copyPointCloud(*(v.pKeypointsMultipipe_), *pKeypoints);
    pcl::copyPointCloud(*(v.kp_normals_), *pKeypointNormals);

    typename std::map<std::string, ObjectHypothesis<PointT> >::const_iterator it;
    for(it = v.hypotheses_.begin(); it !=v.hypotheses_.end(); ++it) //copy
    {
        const ObjectHypothesis<PointT> &oh_src = it->second;

        ObjectHypothesis<PointT> oh;
        oh.model_ = oh_src.model_;
        oh.model_keypoints.reset(new pcl::PointCloud<PointT>(*(oh_src.model_keypoints)));
        oh.model_kp_normals.reset(new pcl::PointCloud<pcl::Normal>(*(oh_src.model_kp_normals)));
        oh.model_scene_corresp.reset(new pcl::Correspondences);
        *(oh.model_scene_corresp) = *(oh_src.model_scene_corresp);
        oh.indices_to_flann_models_ = oh_src.indices_to_flann_models_;
        hypotheses.insert(std::pair<std::string, ObjectHypothesis<PointT> >(it->first, oh));
    }

    if(pKeypoints->points.size() != pKeypointNormals->points.size())
        throw std::runtime_error("keypoint clouds does not have same size as keypoint normals!");

    size_t num_keypoints_single_view = pKeypoints->points.size();

    v.has_been_hopped_ = true;

    graph_traits<MVGraph>::out_edge_iterator out_i, out_end;
    tie ( out_i, out_end ) = out_edges ( vrtx_start, grph);
    if(out_i != out_end)  //otherwise there are no edges to hop
    {   //------get hypotheses from next vertex. Just taking the first one not being hopped.----
        size_t edge_src;
        ViewD remote_vertex;
        Eigen::Matrix4f edge_transform;

        for (; out_i != out_end; ++out_i ) {
            remote_vertex  = target ( *out_i, grph );
            edge_src       = grph[*out_i].source_id;
            edge_transform = grph[*out_i].transformation;

            if (! grph[remote_vertex].has_been_hopped_) {
                if ( edge_src == v.id_)
                    edge_transform = grph[*out_i].transformation.inverse();

                std::cout << "Hopping to vertex "<< grph[remote_vertex].id_ << std::endl;

                std::map<std::string, ObjectHypothesis<PointT> > new_hypotheses;
                typename pcl::PointCloud<PointT>::Ptr pNewKeypoints (new pcl::PointCloud<PointT>);
                pcl::PointCloud<pcl::Normal>::Ptr pNewKeypointNormals (new pcl::PointCloud<pcl::Normal>);
                grph[remote_vertex].absolute_pose_ = v.absolute_pose_ * edge_transform;
                extendFeatureMatchesRecursive ( grph, remote_vertex, new_hypotheses, pNewKeypoints, pNewKeypointNormals);
                assert( pNewKeypoints->size() == pNewKeypointNormals->size() );

                //------ Transform keypoints and rotate normals----------
                const Eigen::Matrix3f rot   = edge_transform.block<3, 3> (0, 0);
                const Eigen::Vector3f trans = edge_transform.block<3, 1> (0, 3);
                for (size_t i=0; i<pNewKeypoints->size(); i++) {
                    pNewKeypoints->points[i].getVector3fMap() = rot * pNewKeypoints->points[i].getVector3fMap () + trans;
                    pNewKeypointNormals->points[i].getNormalVector3fMap() = rot * pNewKeypointNormals->points[i].getNormalVector3fMap ();
                }

                // add/merge hypotheses
                pcl::PointCloud<pcl::PointXYZ>::Ptr pKeypointsXYZ (new pcl::PointCloud<pcl::PointXYZ>);
                pKeypointsXYZ->points.resize(pKeypoints->size());
                for(size_t i=0; i < pKeypoints->size(); i++)
                    pKeypointsXYZ->points[i].getVector3fMap() = pKeypoints->points[i].getVector3fMap();

                *pKeypoints += *pNewKeypoints;
                *pKeypointNormals += *pNewKeypointNormals;

                typename std::map<std::string, ObjectHypothesis<PointT> >::const_iterator it_new_hyp;
                for(it_new_hyp = new_hypotheses.begin(); it_new_hyp !=new_hypotheses.end(); ++it_new_hyp) {
                    const std::string id = it_new_hyp->second.model_->id_;

                    typename std::map<std::string, ObjectHypothesis<PointT> >::iterator it_existing_hyp;
                    it_existing_hyp = hypotheses.find(id);
                    if (it_existing_hyp == hypotheses.end()) {
                        PCL_ERROR("There are no hypotheses (feature matches) for this model yet.");
                        ObjectHypothesis<PointT> oh;
                        oh.model_keypoints.reset(new pcl::PointCloud<PointT>
                                                            (*(it_new_hyp->second.model_keypoints)));
                        oh.model_kp_normals.reset(new pcl::PointCloud<pcl::Normal>
                                                    (*(it_new_hyp->second.model_kp_normals)));
                        oh.indices_to_flann_models_ = it_new_hyp->second.indices_to_flann_models_;
                        oh.model_scene_corresp.reset (new pcl::Correspondences);
                        *(oh.model_scene_corresp) = *(it_new_hyp->second.model_scene_corresp);
                        for(size_t i=0; i < oh.model_scene_corresp->size(); i++)
                        {
                            oh.model_scene_corresp->at(i).index_match += pKeypoints->points.size();
                        }
                        hypotheses.insert(std::pair<std::string, ObjectHypothesis<PointT> >(id, oh));
                    }

                    else { //merge hypotheses

                        assert( (it_new_hyp->second.model_scene_corresp->size() ==  it_new_hyp->second.model_keypoints->size() ) &&
                                (it_new_hyp->second.model_keypoints->size()    == it_new_hyp->second.indices_to_flann_models_.size()) );

                        size_t num_existing_corrs = it_existing_hyp->second.model_scene_corresp->size();
                        size_t num_new_corrs = it_new_hyp->second.model_scene_corresp->size();

                        it_existing_hyp->second.model_scene_corresp->resize( num_existing_corrs + num_new_corrs );
                        it_existing_hyp->second.model_keypoints->resize( num_existing_corrs + num_new_corrs );
                        it_existing_hyp->second.model_kp_normals->resize( num_existing_corrs + num_new_corrs );
                        it_existing_hyp->second.indices_to_flann_models_.resize( num_existing_corrs + num_new_corrs );

                        size_t kept_corrs = num_existing_corrs;

                        for(size_t j=0; j < it_new_hyp->second.model_scene_corresp->size(); j++) {

                            bool drop_new_correspondence = false;

                            pcl::Correspondence c_new = it_new_hyp->second.model_scene_corresp->at(j);
                            const PointT new_kp = pNewKeypoints->points[c_new.index_match];
                            pcl::PointXYZ new_kp_XYZ;
                            new_kp_XYZ.getVector3fMap() = new_kp.getVector3fMap();
                            const pcl::Normal new_kp_normal = pNewKeypointNormals->points[c_new.index_match];

                            const PointT new_model_pt = it_new_hyp->second.model_keypoints->points[ c_new.index_query ];
                            pcl::PointXYZ new_model_pt_XYZ;
                            new_model_pt_XYZ.getVector3fMap() = new_model_pt.getVector3fMap();
                            const pcl::Normal new_model_normal = it_new_hyp->second.model_kp_normals->points[ c_new.index_query ];

                            int idx_to_flann_model = it_new_hyp->second.indices_to_flann_models_[j];

                            if (!pcl::isFinite(new_kp_XYZ) || !pcl::isFinite(new_model_pt_XYZ)) {
                                PCL_WARN("Keypoint of scene or model is infinity!!");
                                continue;
                            }

                            for(size_t k=0; k < num_existing_corrs; k++) {
                                const pcl::Correspondence c_existing = it_existing_hyp->second.model_scene_corresp->at(k);
                                const PointT existing_model_pt = it_existing_hyp->second.model_keypoints->points[ c_existing.index_query ];
                                pcl::PointXYZ existing_model_pt_XYZ;
                                existing_model_pt_XYZ.getVector3fMap() = existing_model_pt.getVector3fMap();

                                const PointT existing_kp = pKeypoints->points[c_existing.index_match];
                                pcl::PointXYZ existing_kp_XYZ;
                                existing_kp_XYZ.getVector3fMap() = existing_kp.getVector3fMap();
                                const pcl::Normal existing_kp_normal = pKeypointNormals->points[c_existing.index_match];

                                float squaredDistModelKeypoints = pcl::squaredEuclideanDistance(new_model_pt_XYZ, existing_model_pt_XYZ);
                                float squaredDistSceneKeypoints = pcl::squaredEuclideanDistance(new_kp_XYZ, existing_kp_XYZ);

                                if((squaredDistSceneKeypoints < param_.distance_keypoints_get_discarded_) &&
                                        (new_kp_normal.getNormalVector3fMap().dot(existing_kp_normal.getNormalVector3fMap()) > 0.8) &&
                                        (squaredDistModelKeypoints < param_.distance_keypoints_get_discarded_)) {
                                    //                                std::cout << "Found a very close point (keypoint distance: " << squaredDistSceneKeypoints
                                    //                                          << "; model distance: " << squaredDistModelKeypoints
                                    //                                          << ") with the same model id and similar normal (Normal dot product: "
                                    //                                          << new_kp_normal.getNormalVector3fMap().dot(existing_kp_normal.getNormalVector3fMap()) << "> 0.8). Ignoring it."
                                    //                                                                      << std::endl;
                                    drop_new_correspondence = true;
                                    break;
                                }
                            }
                            if (!drop_new_correspondence) {
                                it_existing_hyp->second.indices_to_flann_models_[kept_corrs] = idx_to_flann_model; //check that
                                c_new.index_query = kept_corrs;
                                c_new.index_match += num_keypoints_single_view;
                                it_existing_hyp->second.model_scene_corresp->at(kept_corrs) = c_new;
                                it_existing_hyp->second.model_keypoints->points[kept_corrs] = new_model_pt;
                                it_existing_hyp->second.model_kp_normals->points[kept_corrs] = new_model_normal;
                                kept_corrs++;
                            }
                        }
                        it_existing_hyp->second.model_scene_corresp->resize( kept_corrs );
                        it_existing_hyp->second.model_keypoints->resize( kept_corrs );
                        it_existing_hyp->second.model_kp_normals->resize( kept_corrs );
                        it_existing_hyp->second.indices_to_flann_models_.resize( kept_corrs );

                        //                    std::cout << "INFO: Size for " << id <<
                        //                                 " of correspondes_pointcloud after merge: " << it_existing_hyp->second.model_keypoints->points.size() << std::endl;
                    }
                }
            }
        }
    }
}


template<typename PointT>
void
MultiviewRecognizer<PointT>::extendHypothesisRecursive ( MVGraph &grph, ViewD &vrtx_start, std::vector<Hypothesis<PointT> > &hyp_vec, bool use_unverified_hypotheses) //is directed edge (so the source of calling_edge is calling vertex)
{
    View &v = grph[vrtx_start];
    v.has_been_hopped_ = true;

    for ( typename std::vector<Hypothesis<PointT> >::const_iterator it_hyp = v.hypothesis_mv_.begin (); it_hyp != v.hypothesis_mv_.end (); ++it_hyp ) {
        if(!it_hyp->verified_ && !use_unverified_hypotheses)
            continue;

        bool id_already_exists = false;
        //--check if hypothesis already exists
        typename std::vector<Hypothesis<PointT> >::const_iterator it_exist_hyp;
        for ( it_exist_hyp = hyp_vec.begin (); it_exist_hyp != hyp_vec.end (); ++it_exist_hyp ) {
            if( it_hyp->id_ == it_exist_hyp->id_) {
                id_already_exists = true;
                break;
            }
        }
        if(!id_already_exists) {
            Hypothesis<PointT> ht_temp ( it_hyp->model_, it_hyp->transform_, it_hyp->origin_, false, false, it_hyp->id_ );
            hyp_vec.push_back(ht_temp);
        }
    }

    typename graph_traits<MVGraph>::out_edge_iterator out_i, out_end;
    tie ( out_i, out_end ) = out_edges ( vrtx_start, grph);
    if(out_i != out_end) {  //otherwise there are no edges to hop - get hypotheses from next vertex. Just taking the first one not being hopped.----
        size_t edge_src;
        ViewD remote_vertex;
        Eigen::Matrix4f edge_transform;

        for (; out_i != out_end; ++out_i ) {
            remote_vertex  = target ( *out_i, grph );
            edge_src       = grph[*out_i].source_id;
            edge_transform = grph[*out_i].transformation;

            if (! grph[remote_vertex].has_been_hopped_) {
                grph[remote_vertex].cumulative_weight_to_new_vrtx_ = v.cumulative_weight_to_new_vrtx_ + grph[*out_i].edge_weight;

                if ( edge_src == v.id_ )
                    edge_transform = grph[*out_i].transformation.inverse();

                std::cout << "Hopping to vertex " <<  grph[remote_vertex].id_
                          << " which has a cumulative weight of "
                          <<  grph[remote_vertex].cumulative_weight_to_new_vrtx_ << std::endl;

                std::vector<Hypothesis<PointT> > new_hypotheses;
                grph[remote_vertex].absolute_pose_ = v.absolute_pose_ * edge_transform;
                extendHypothesisRecursive ( grph, remote_vertex, new_hypotheses);

                for(typename std::vector<Hypothesis<PointT> >::iterator it_new_hyp = new_hypotheses.begin(); it_new_hyp !=new_hypotheses.end(); ++it_new_hyp) {
                    it_new_hyp->transform_ = edge_transform * it_new_hyp->transform_;
                    Hypothesis<PointT> ht_temp ( it_new_hyp->model_, it_new_hyp->transform_, it_new_hyp->origin_, true, false, it_new_hyp->id_ );
                    hyp_vec.push_back(ht_temp);
                }
            }
        }
    }
}

template<typename PointT>
void
MultiviewRecognizer<PointT>::createEdgesFromHypothesisMatchOnline ( const ViewD new_vertex, MVGraph &grph, std::vector<EdgeD> &edges )
{
    vertex_iter vertexItA, vertexEndA;
    for (boost::tie(vertexItA, vertexEndA) = vertices(grph_); vertexItA != vertexEndA; ++vertexItA)
    {
        if ( grph[*vertexItA].id_ == grph[new_vertex].id_ )
            continue;

        std::cout << " Checking vertex " << grph[*vertexItA].id_ << " which has " << grph[*vertexItA].hypothesis_mv_.size() << " hypotheses." << std::endl;
        for ( typename std::vector<Hypothesis<PointT> >::iterator it_hypA = grph[*vertexItA].hypothesis_mv_.begin (); it_hypA != grph[*vertexItA].hypothesis_mv_.end (); ++it_hypA )
        {
            if (! it_hypA->verified_)
                continue;

            for ( typename std::vector<Hypothesis<PointT> >::iterator it_hypB = grph[new_vertex].hypothesis_sv_.begin (); it_hypB != grph[new_vertex].hypothesis_sv_.end (); ++it_hypB )
            {
                if(!it_hypB->verified_)
                    continue;

                if ( it_hypB->model_id_.compare (it_hypA->model_id_ ) == 0 ) //model exists in other file (same id) --> create connection
                {
                    Eigen::Matrix4f tf_temp = it_hypB->transform_ * it_hypA->transform_.inverse (); //might be the other way around

                    //link views by an edge (for other graph)
                    EdgeD e_cpy;
                    bool b;
                    tie ( e_cpy, b ) = add_edge ( *vertexItA, new_vertex, grph );
                    grph[e_cpy].transformation = tf_temp;
                    grph[e_cpy].model_name = it_hypA->model_id_;
                    grph[e_cpy].source_id = grph[*vertexItA].id_;
                    grph[e_cpy].target_id = grph[new_vertex].id_;
                    grph[e_cpy].edge_weight = std::numeric_limits<double>::max ();
                    edges.push_back ( e_cpy );
                }
            }
        }
    }
}

template<typename PointT>
void
MultiviewRecognizer<PointT>::calcEdgeWeight (MVGraph &grph, std::vector<EdgeD> &edges)
{
#pragma omp parallel for schedule(dynamic,1) num_threads(omp_get_num_procs())
    for (size_t i=0; i<edges.size(); i++) //std::vector<Edge>::iterator edge_it = edges.begin(); edge_it!=edges.end(); ++edge_it)
    {
        EdgeD &edge = edges[i];

        const ViewD vrtx_src = source ( edge, grph );
        const ViewD vrtx_trgt = target ( edge, grph );

        Eigen::Matrix4f transform;
        if ( grph[edge].source_id == grph[vrtx_src].id_ )
            transform = grph[edge].transformation;
        else
            transform = grph[edge].transformation.inverse ();

        float w_after_icp_ = std::numeric_limits<float>::max ();
        const float best_overlap_ = 0.75f;

        Eigen::Matrix4f icp_trans;
        FastIterativeClosestPointWithGC<PointT> icp;
        icp.setMaxCorrespondenceDistance ( 0.02f );
        icp.setInputSource (grph[vrtx_src].scene_f_);
        icp.setInputTarget (grph[vrtx_trgt].scene_f_);
        icp.setUseNormals (true);
        icp.useStandardCG (true);
        icp.setNoCG(true);
        icp.setOverlapPercentage (best_overlap_);
        icp.setKeepMaxHypotheses (5);
        icp.setMaximumIterations (10);
        icp.align (transform);
        w_after_icp_ = icp.getFinalTransformation ( icp_trans );

        if ( w_after_icp_ < 0 || !pcl_isfinite ( w_after_icp_ ) )
            w_after_icp_ = std::numeric_limits<float>::max ();
        else
            w_after_icp_ = best_overlap_ - w_after_icp_;

        if ( grph[edge].source_id == grph[vrtx_src].id_ )
        {
            PCL_WARN ( "Normal...\n" );
            //icp trans is aligning source to target
            //transform is aligning source to target
            //grph[edges[edge_id]].transformation = icp_trans * grph[edges[edge_id]].transformation;
            grph[edge].transformation = icp_trans;
        }
        else
        {
            //transform is aligning target to source
            //icp trans is aligning source to target
            PCL_WARN ( "Inverse...\n" );
            //grph[edges[edge_id]].transformation = icp_trans.inverse() * grph[edges[edge_id]].transformation;
            grph[edge].transformation = icp_trans.inverse ();
        }

        grph[edge].edge_weight = w_after_icp_;

        std::cout << "WEIGHT IS: " << grph[edge].edge_weight << " coming from edge connecting " << grph[edge].source_id
                  << " and " << grph[edge].target_id << " by object_id: " << grph[edge].model_name
                  << std::endl;
    }
}

template<typename PointT>
bool
MultiviewRecognizer<PointT>::recognize (const typename pcl::PointCloud<PointT>::ConstPtr cloud,
                                const Eigen::Matrix4f &global_transform)
{
    if(!rr_)
        throw std::runtime_error("Single-View recognizer is not set. Please provide a recognizer to the multi-view recognition system!");

    std::cout << "=================================================================" << std::endl <<
                 "Started recognition for view " << ID << " in scene " << scene_name_ <<
                 "=========================================================" << std::endl << std::endl;

    size_t total_num_correspondences = 0;
    boost::shared_ptr< pcl::PointCloud<pcl::Normal> > scene_normals_f (new pcl::PointCloud<pcl::Normal> );

    if (cloud->width != 640 || cloud->height != 480)
        throw std::runtime_error("Size of input cloud is not 640x480, which is the only resolution currently supported by the verification framework.");

    ViewD new_vrtx = boost::add_vertex ( grph_ );
    View &v = grph_[new_vrtx];
    v.id_ = ID++;

    v.pScenePCl = cloud;

    v.transform_to_world_co_system_ = global_transform;
    v.absolute_pose_ = global_transform;    // this might be redundant

    computeNormals<PointT>(v.pScenePCl, v.pSceneNormals, param_.normal_computation_method_);
    pcl::copyPointCloud(*v.pSceneNormals, v.filteredSceneIndices_, *scene_normals_f);

    std::vector<EdgeD> new_edges;
    //--------------create-edges-between-views-by-Robot-Pose-----------------------------
    if( param_.use_robot_pose_ ) {
        vertex_iter vertexIt, vertexEnd;
        for (boost::tie(vertexIt, vertexEnd) = vertices(grph_); vertexIt != vertexEnd; ++vertexIt) {
            const View &w = grph_[*vertexIt];

            if( w.id_ != v.id_) {
                EdgeD edge; bool b;
                tie ( edge, b ) = add_edge ( src, target, grph_ );
                Edge &e = grph_[edge];
                Eigen::Matrix4f tf2wco_src = w.transform_to_world_co_system_;
                Eigen::Matrix4f tf2wco_trgt = v.transform_to_world_co_system_;
                e.transformation = tf2wco_src.inverse() * tf2wco_trgt;
                e.model_name = std::string ( "robot_pose" );
                e.target_id = w.id_;
                e.source_id = v.id_;
                new_edges.push_back ( edge );
            }
        }
    }

    //-------------create-edges-between-views-by-SIFT-----------------------------------
    if( param_.scene_to_scene_) {
        calcSiftFeatures ( new_vrtx, grph_ );
        std::cout << "keypoints: " << v.siftKeypointIndices_.indices.size() << std::endl;

        if (num_vertices(grph_)>1) {
            boost::shared_ptr< flann::Index<DistT> > flann_index;
            convertToFLANN<FeatureT, DistT >( v.pSiftSignatures_, flann_index );

            //#pragma omp parallel for
            vertex_iter vertexIt, vertexEnd;
            for (boost::tie(vertexIt, vertexEnd) = vertices(grph_); vertexIt != vertexEnd; ++vertexIt) {
                Eigen::Matrix4f transformation;
                const View &w = grph_[*vertexIt];

                if( w.id_ !=  v.id_ ) {
                    std::vector<EdgeD> edge;
                    estimateViewTransformationBySIFT ( *vertexIt, new_vrtx, grph_, flann_index, transformation, edge,  param_.use_gc_s2s_ );
                    for(size_t kk=0; kk < edge.size(); kk++)
                        new_edges.push_back (edge[kk]);
                }
            }
        }

        // In addition to matching views, we can use the computed SIFT features for recognition
        rr_->setFeatAndKeypoints<flann::L1, FeatureT > (v.pSiftSignatures_, v.siftKeypointIndices_, SIFT);
    }
    //----------END-create-edges-between-views-by-SIFT-----------------------------------

    rr_->setInputCloud(v.scene_f_);
    rr_->setSceneNormals(scene_normals_f);
    rr_->recognize();

    if(rr_->getSaveHypothesesParam())   // correspondence grouping is done already (so we get the full models) [Faeulhammer et al 2015, MVA paper]
    {
        rr_->getSavedHypotheses(v.hypotheses_);
        rr_->getKeypointCloud(v.pKeypointsMultipipe_);
        rr_->getKeypointIndices(v.kp_indices_);
        pcl::copyPointCloud(*scene_normals_f, v.kp_indices_, *v.kp_normals_);
    }
    else    // we have to do the correspondence grouping ourselve [Faeulhammer et al 2015, ICRA paper]
    {
        if (param_.hyp_to_hyp_)
            createEdgesFromHypothesisMatchOnline(new_vrtx, grph_, new_edges);
    }

    std::vector < pcl::Correspondences > corresp_clusters_sv;
    constructHypothesesFromFeatureMatches(v.hypotheses_, v.pKeypointsMultipipe_,
                                          v.kp_normals_, v.hypothesis_sv_,
                                          corresp_clusters_sv);
    poseRefinement();
    std::vector<bool> mask_hv_sv = hypothesesVerification();
    getVerifiedPlanes(v.verified_planes_);

    for (size_t j = 0; j < v.hypothesis_sv_.size(); j++)
        v.hypothesis_sv_[j].verified_ = mask_hv_sv[j];
    //----------END-call-single-view-recognizer------------------------------------------


    //---copy-vertices-to-graph_final----------------------------
    ViewD vrtx_final = boost::add_vertex ( grph_final_ );
    View &v_f = grph_final_[vrtx_final];

    v_f = v; // shallow copy is okay here

    if(new_edges.size()) {
        EdgeD best_edge;
        best_edge = new_edges[0];

        if(new_edges.size()>1) { // take the "best" edge for transformation between the views and add it to the final graph
            calcEdgeWeight (grph_, new_edges);
            for ( size_t i = 1; i < new_edges.size(); i++ ) {
                if ( grph_[new_edges[i]].edge_weight < grph_[best_edge].edge_weight )
                    best_edge = new_edges[i];
            }
        }
        //        bgvis_.visualizeEdge(new_edges[0], grph_);
        ViewD vrtx_src, vrtx_trgt;
        vrtx_src = source ( best_edge, grph_ );
        vrtx_trgt = target ( best_edge, grph_ );

        EdgeD e_cpy; bool b;
        tie ( e_cpy, b ) = add_edge ( vrtx_src, vrtx_trgt, grph_final_ );
        grph_final_[e_cpy] = grph_[best_edge]; // shallow copy is okay here
    }
    else
        std::cout << "No edge for this vertex." << std::endl;

    if(visualize_output_)
        bgvis_.visualizeGraph(grph_final_, vis_);

    resetHopStatus(grph_final_);

    v = v_f; // shallow copy is okay here

    //-------Clean-up-graph-------------------
    outputgraph ( grph_, "/tmp/complete_graph.dot" );
    outputgraph ( grph_final_, "/tmp/final_with_hypotheses_extension.dot" );

    savePCDwithPose();

    if( param_.extension_mode_ == 0 ) {
        // bgvis_.visualizeWorkflow(vrtx_final, grph_final_, pAccumulatedKeypoints_);
        //    bgvis_.createImage(vrtx_final, grph_final_, "/home/thomas/Desktop/test.jpg");
    }

    pruneGraph(grph_, param_.max_vertices_in_graph_);
    pruneGraph(grph_final_, param_.max_vertices_in_graph_);
    return true;
}

template<typename PointT>
void
MultiviewRecognizer<PointT>::savePCDwithPose()
{
    for (std::pair<vertex_iter, vertex_iter> vp = vertices (grph_final_); vp.first != vp.second; ++vp.first) {
        setCloudPose(grph_final_[*vp.first].absolute_pose_, *grph_final_[*vp.first].pScenePCl);
        std::stringstream fn; fn <<   grph_final_[*vp.first].id_ + ".pcd";
        pcl::io::savePCDFileBinary(fn.str(), *(grph_final_[*vp.first].pScenePCl));
    }
}
}
