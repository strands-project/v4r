#include <algorithm>
#include <glog/logging.h>
#include <v4r/common/graph_geometric_consistency.h>
#include <v4r/recognition/local_recognition_pipeline.h>
#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/recognition/multiview_recognizer.h>

#include <glog/logging.h>
#include <omp.h>
#include <pcl/common/time.h>
#include <pcl/recognition/cg/correspondence_grouping.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

namespace v4r
{

template<typename PointT>
void
MultiviewRecognizer<PointT>::do_recognize()
{
    local_obj_hypotheses_.clear();

    View v;
    v.camera_pose_ = v4r::RotTrans2Mat4f( scene_->sensor_orientation_, scene_->sensor_origin_ );

    recognition_pipeline_->setInputCloud( scene_ );
    recognition_pipeline_->setSceneNormals( scene_normals_ );

    if( table_plane_set_ )
        recognition_pipeline_->setTablePlane( table_plane_ );

    recognition_pipeline_->recognize();
    v.obj_hypotheses_ = recognition_pipeline_->getObjectHypothesis();

    table_plane_set_ = false;

    obj_hypotheses_ = v.obj_hypotheses_;

    std::set<size_t> hypotheses_ids; /// to make sure hypotheses are only transferred once

    // now add the old hypotheses
    for(int v_id = views_.size() - 1; v_id>=std::max<int>(0, views_.size()-param_.max_views_+1); v_id--)
    {
        const View &v_old = views_[v_id];
        for(const ObjectHypothesesGroup &ohg_tmp : v_old.obj_hypotheses_)
        {
            bool do_hyp_to_transfer = false;
            if ( param_.transfer_keypoint_correspondences_ && !ohg_tmp.global_hypotheses_ )
                continue;

            for( const typename ObjectHypothesis::Ptr &oh_tmp : ohg_tmp.ohs_)
            {
                if( !param_.transfer_only_verified_hypotheses_ || oh_tmp->is_verified_ )
                {
                    // check if this hypotheses is not already transferred by any other view
                    if( std::find (hypotheses_ids.begin(), hypotheses_ids.end(), oh_tmp->unique_id_ ) == hypotheses_ids.end())
                    {
                        do_hyp_to_transfer = true;
                        break;
                    }
                }
            }

            if( do_hyp_to_transfer || !param_.transfer_only_verified_hypotheses_ )
            {
                ObjectHypothesesGroup ohg;
                ohg.global_hypotheses_ = ohg_tmp.global_hypotheses_;

                for( const typename ObjectHypothesis::Ptr &oh_tmp : ohg_tmp.ohs_)
                {
                    if( param_.transfer_only_verified_hypotheses_ && !oh_tmp->is_verified_ )
                        continue;

                    if( std::find (hypotheses_ids.begin(), hypotheses_ids.end(), oh_tmp->unique_id_ ) != hypotheses_ids.end())
                        continue;

                    hypotheses_ids.insert(oh_tmp->unique_id_);

                    // create a copy (since we want to reset verification status and update transform but keep the status for old view)
                    typename ObjectHypothesis::Ptr oh_copy ( new ObjectHypothesis (*oh_tmp) );
                    oh_copy->is_verified_ = false;
                    oh_copy->transform_ = v.camera_pose_.inverse() * v_old.camera_pose_ * oh_copy->pose_refinement_ * oh_copy->transform_;
//                    oh_copy->transform_ = v_old.camera_pose_ * oh_copy->transform_; ///< ATTENTION: This depends on the input cloud (i.e. in this case the input cloud is in the global reference frame)
                    ohg.ohs_.push_back( oh_copy );
                }

                obj_hypotheses_.push_back(ohg);
            }
        }
    }

    if(param_.transfer_keypoint_correspondences_)
    {

        // get local keypoints and feature matches
        typename MultiRecognitionPipeline<PointT>::Ptr mp_recognizer =
                boost::dynamic_pointer_cast<  MultiRecognitionPipeline<PointT> > (recognition_pipeline_);

        CHECK(mp_recognizer);

        std::vector<typename RecognitionPipeline<PointT>::Ptr > sv_rec_pipelines = mp_recognizer->getRecognitionPipelines();
        for( const typename RecognitionPipeline<PointT>::Ptr &rec_pipeline : sv_rec_pipelines )
        {

            typename LocalRecognitionPipeline<PointT>::Ptr local_rec_pipeline =
                    boost::dynamic_pointer_cast<  LocalRecognitionPipeline<PointT> > (rec_pipeline);

            if(local_rec_pipeline)
            {
                v.local_obj_hypotheses_ = local_rec_pipeline->getKeypointCorrespondences();
                v.model_keypoints_ = local_rec_pipeline->getLocalObjectModelDatabase();
                pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud_xyz  (new pcl::PointCloud<pcl::PointXYZ>);
                pcl::copyPointCloud(*scene_, *scene_cloud_xyz);
                v.scene_cloud_xyz_ = scene_cloud_xyz;
                v.scene_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal>(*scene_normals_));
            }
        }

        local_obj_hypotheses_ = v.local_obj_hypotheses_;
        model_keypoints_ = v.model_keypoints_;
        scene_cloud_xyz_merged_.reset (new pcl::PointCloud<pcl::PointXYZ>(*v.scene_cloud_xyz_));
        scene_cloud_normals_merged_.reset (new pcl::PointCloud<pcl::Normal>(*v.scene_cloud_normals_));

        // now add the old hypotheses
        for(int v_id = views_.size() - 1; v_id>=std::max<int>(0, views_.size()-param_.max_views_+1); v_id--)
        {
            const View &v_old = views_[v_id];
            {
                pcl::PointCloud<pcl::PointXYZ> scene_cloud_xyz_aligned;
                pcl::PointCloud<pcl::Normal> scene_normals_aligned;

                const Eigen::Matrix4f tf2current = v.camera_pose_.inverse() * v_old.camera_pose_;

                pcl::transformPointCloud( *v_old.scene_cloud_xyz_, scene_cloud_xyz_aligned, tf2current);
                v4r::transformNormals( *v_old.scene_cloud_normals_, scene_normals_aligned, tf2current);

                size_t offset = scene_cloud_xyz_merged_->points.size();
                VLOG(2) << "offset: " << offset;

                *scene_cloud_xyz_merged_ += scene_cloud_xyz_aligned;
                *scene_cloud_normals_merged_ += scene_normals_aligned;
                VLOG(2) << "scene_cloud_xyz_merged_: " << scene_cloud_xyz_merged_->points.size()
                        << "scene_cloud_normals_merged_: " << scene_cloud_normals_merged_->points.size();

                for (const auto &oh : v_old.local_obj_hypotheses_)    // iterate through all models
                {
                    const std::string &model_id = oh.first;
                    const LocalObjectHypothesis<PointT> &loh = oh.second;

                    pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoints = model_keypoints_[model_id]->keypoints_;
                    pcl::PointCloud<pcl::Normal>::Ptr model_kp_normals = model_keypoints_[model_id]->kp_normals_;

                    pcl::Correspondences new_corrs = *loh.model_scene_corresp_;
                    size_t initial_corrs = new_corrs.size();

                    for (pcl::Correspondence &c : new_corrs) // add appropriate offset to correspondence index of the model keypoints
                    {
//                        CHECK( c.index_match < (int) scene_cloud_xyz_aligned.points.size() && c.index_match >= 0 ) << "c.index_match: " << c.index_match << ", scene_cloud_xyz_merged_->points.size(): " << scene_cloud_xyz_merged_->points.size();
//                        CHECK( c.index_match < (int) scene_normals_aligned.points.size() && c.index_match >= 0 );
//                        CHECK( c.index_query < (int) model_keypoints->points.size() && c.index_query >= 0 );
//                        CHECK( c.index_query < (int) model_kp_normals->points.size() && c.index_query >= 0 );

                        c.index_match += offset;
//                        c.index_query += model_keypoints_[ model_id ]->keypoints_->points.size();
                    }

//                    for (const pcl::Correspondence &c : new_corrs) // add appropriate offset to correspondence index of the model keypoints
//                    {
//                        CHECK( c.index_match < (int) scene_cloud_xyz_merged_->points.size() && c.index_match >= 0 ) << "c.index_match: " << c.index_match << ", scene_cloud_xyz_merged_->points.size(): " << scene_cloud_xyz_merged_->points.size();
//                        CHECK( c.index_match < (int) scene_cloud_normals_merged_->points.size() && c.index_match >= 0 );
//                        CHECK( c.index_query < (int) model_keypoints->points.size() && c.index_query >= 0 );
//                        CHECK( c.index_query < (int) model_kp_normals->points.size() && c.index_query >= 0 );
//                    }

                    const auto existing_kp_it = v_old.model_keypoints_.find(model_id);
                    CHECK( existing_kp_it != v_old.model_keypoints_.end() );
//                    const typename LocalObjectModel::ConstPtr old_model_kps = existing_kp_it->second;


//                    *model_keypoints += *old_model_kps->keypoints_;
//                    *model_kp_normals += *old_model_kps->kp_normals_;


                    auto it_mp_oh = local_obj_hypotheses_.find( model_id );
                    if( it_mp_oh == local_obj_hypotheses_.end() )   // no feature correspondences exist yet
                        local_obj_hypotheses_.insert( oh );
                    else
                    {
                        pcl::Correspondences &old_corrs = *it_mp_oh->second.model_scene_corresp_;

                        size_t kept=0; // check for redundancy
                        for(size_t new_corr_id=0; new_corr_id<new_corrs.size(); new_corr_id++)
                        {
                            const pcl::Correspondence &new_c = new_corrs[new_corr_id];
                            const Eigen::Vector3f &new_scene_xyz = scene_cloud_xyz_merged_->points[new_c.index_match].getVector3fMap();
                            const Eigen::Vector3f &new_scene_normal = scene_cloud_normals_merged_->points[new_c.index_match].getNormalVector3fMap();
                            const Eigen::Vector3f &new_model_xyz = model_keypoints->points[new_c.index_query].getVector3fMap();
                            const Eigen::Vector3f &new_model_normal = model_kp_normals->points[new_c.index_query].getNormalVector3fMap();

                            size_t is_redundant = false;
                            for(size_t old_corr_id=0; old_corr_id<old_corrs.size(); old_corr_id++)
                            {
                                pcl::Correspondence &old_c = old_corrs[old_corr_id];
                                const Eigen::Vector3f &old_scene_xyz = scene_cloud_xyz_merged_->points[old_c.index_match].getVector3fMap();
                                const Eigen::Vector3f &old_scene_normal = scene_cloud_normals_merged_->points[old_c.index_match].getNormalVector3fMap();
                                const Eigen::Vector3f &old_model_xyz = model_keypoints->points[old_c.index_query].getVector3fMap();
                                const Eigen::Vector3f &old_model_normal = model_kp_normals->points[old_c.index_query].getNormalVector3fMap();

                                if ( (old_scene_xyz-new_scene_xyz).norm() < param_.min_dist_ &&
                                     (old_model_xyz-new_model_xyz).norm() < param_.min_dist_ &&
                                      old_scene_normal.dot(new_scene_normal) > param_.max_dotp_ &&
                                      old_model_normal.dot(new_model_normal) > param_.max_dotp_ )
                                {
                                    is_redundant = true;

                                    // take the correspondence with the smaller distance
                                    if( new_c.distance < old_c.distance )
                                        old_c = new_c;

                                    break;
                                }
                            }
                            if(!is_redundant)
                                new_corrs[kept++] = new_c;
                        }
                        LOG(INFO) << "Kept " << kept << " out of " << initial_corrs << " correspondences.";
                        new_corrs.resize(kept);
                        old_corrs.insert(  old_corrs.end(), new_corrs.begin(), new_corrs.end() );
                    }
                }
            }
        }

        if(param_.visualize_)
            visualize();
        correspondenceGrouping();
    }

    v.obj_hypotheses_ = obj_hypotheses_;
    views_.push_back(v);

}


template<typename PointT>
void
MultiviewRecognizer<PointT>::visualize()
{
    size_t counter = 0;
    typename std::map<std::string, LocalObjectHypothesis<PointT> >::const_iterator it;
    for ( it = local_obj_hypotheses_.begin (); it != local_obj_hypotheses_.end (); ++it )
    {
        pcl::visualization::PCLVisualizer::Ptr vis(new pcl::visualization::PCLVisualizer);
        int vp1, vp2, vp3;
        vis->createViewPort(0,0,0.0,1,vp1);
        vis->createViewPort(0.0,0,0.0,1,vp2);
        vis->createViewPort(0.0,0,1,1,vp3);
        static bool co_init = false;

        if( counter++ > 1000 ) // only show first three
            break;

        const std::string &model_id = it->first;
        const LocalObjectHypothesis<PointT> &loh = it->second;

        pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoints = model_keypoints_[model_id]->keypoints_;
        pcl::PointCloud<pcl::Normal>::Ptr model_kp_normals = model_keypoints_[model_id]->kp_normals_;


        bool found;
        typename Model<PointT>::ConstPtr model = m_db_->getModelById("", model_id, found);
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = model->getAssembled(3);
        vis->removeAllPointClouds(vp1);
        vis->removeAllPointClouds(vp2);
        vis->removeAllPointClouds(vp3);
        vis->removeAllShapes();
        vis->removeAllShapes(vp1);
        vis->removeAllShapes(vp2);
        vis->removeAllShapes(vp3);
        vis->setBackgroundColor(1., 1., 1. );
        vis->addPointCloud(model_cloud, "model_cloud1", vp1);
        LOG(INFO) << "Visualizing keypoints for " << model_id;

        pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud_xyz_merged_vis (new pcl::PointCloud<pcl::PointXYZ>(*scene_cloud_xyz_merged_));
        scene_cloud_xyz_merged_vis->sensor_origin_ = Eigen::Vector4f::Zero();
        scene_cloud_xyz_merged_vis->sensor_orientation_ = Eigen::Quaternionf::Identity();
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> black (scene_cloud_xyz_merged_vis, 0, 0, 0);
//        vis->addPointCloud(scene_cloud_xyz_merged_vis, black, "scene", vp2);

        typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_kps_colored (new pcl::PointCloud<pcl::PointXYZRGB>);
        model_kps_colored->points.resize( loh.model_scene_corresp_->size() );
        typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_kps_colored (new pcl::PointCloud<pcl::PointXYZRGB>);
        scene_kps_colored->points.resize( loh.model_scene_corresp_->size() );

        typename pcl::PointCloud<pcl::Normal>::Ptr model_kps_normals (new pcl::PointCloud<pcl::Normal>);
        model_kps_normals->points.resize( loh.model_scene_corresp_->size() );
        typename pcl::PointCloud<pcl::Normal>::Ptr scene_kps_normals (new pcl::PointCloud<pcl::Normal>);
        scene_kps_normals->points.resize( loh.model_scene_corresp_->size() );

        for ( size_t c_id=0; c_id<loh.model_scene_corresp_->size(); c_id++ )
        {
            const pcl::Correspondence &c = loh.model_scene_corresp_->at(c_id);
            pcl::PointXYZRGB &m = model_kps_colored->points[c_id];
            pcl::PointXYZRGB &s = scene_kps_colored->points[c_id];
            float r = 255;//rand() % 255;
            float g = 0;//rand() % 255;
            float b = 0;//rand() % 255;
            m.r = r; m.g = g; m.b = b;
            s.r = r; s.g = g; s.b = b;

            m.getVector3fMap() = model_keypoints_[model_id]->keypoints_->points.at(c.index_query).getVector3fMap();
            s.getVector3fMap() = scene_cloud_xyz_merged_->points.at(c.index_match).getVector3fMap();

            pcl::Normal &mn = model_kps_normals->points[c_id];
            pcl::Normal &sn = scene_kps_normals->points[c_id];
            mn.getNormalVector3fMap() = model_keypoints_[model_id]->kp_normals_->points.at(c.index_query).getNormalVector3fMap();
            sn.getNormalVector3fMap() = scene_cloud_normals_merged_->points.at(c.index_match).getNormalVector3fMap();

        }


        if( loh.model_scene_corresp_->size() < 3 )
            continue;

        // CHECK CORRESPONDENCE GROUPING ==========

        typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_model_kps (new pcl::PointCloud<pcl::PointXYZRGB>);
        typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_scene_kps (new pcl::PointCloud<pcl::PointXYZRGB>);
        std::vector < pcl::Correspondences > corresp_clusters;
        {
            std::sort( loh.model_scene_corresp_->begin(), loh.model_scene_corresp_->end(), LocalObjectHypothesis<PointT>::gcGraphCorrespSorter);
            cg_algorithm_->setSceneCloud ( scene_cloud_xyz_merged_ );
            cg_algorithm_->setInputCloud ( model_keypoints );

            // Graph-based correspondence grouping requires normals but interface does not exist in base class - so need to try pointer casting
            typename GraphGeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ>::Ptr gcg_algorithm =
                    boost::dynamic_pointer_cast<  GraphGeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> > (cg_algorithm_);
            if( gcg_algorithm )
                gcg_algorithm->setInputAndSceneNormals(model_kp_normals, scene_cloud_normals_merged_);

            cg_algorithm_->setModelSceneCorrespondences ( loh.model_scene_corresp_ );
            cg_algorithm_->cluster (corresp_clusters);

            for( const pcl::Correspondences &cs : corresp_clusters )
            {

                float r = rand() % 255;
                float g = rand() % 255;
                float b = rand() % 255;

                for(const pcl::Correspondence &c : cs)
                {
                    pcl::PointXYZRGB m;
                    m.getVector3fMap() = model_keypoints->points[c.index_query].getVector3fMap();
                    pcl::PointXYZRGB s;
                    s.getVector3fMap() = scene_cloud_xyz_merged_->points[c.index_match].getVector3fMap();

                    m.r = r; m.g = g; m.b = b;
                    s.r = r; s.g = g; s.b = b;
                    clustered_model_kps->points.push_back(m);
                    clustered_scene_kps->points.push_back(s);
                }
            }
        }

//        Eigen::Vector4f model_centroid;
//        pcl::compute3DCentroid(*model_kps_colored, model_centroid);
//        for( pcl::PointXYZRGB &m : model_kps_colored->points)
//            m.getVector4fMap() -= model_centroid;

//        for( pcl::PointXYZRGB &m : clustered_model_kps->points)
//            m.getVector4fMap() -= model_centroid;

        Eigen::Vector4f scene_centroid;
        pcl::compute3DCentroid(*scene_, scene_centroid);
        for( pcl::PointXYZRGB &s : clustered_scene_kps->points)
            s.getVector4fMap() -= scene_centroid;

        for( pcl::PointXYZRGB &s : scene_kps_colored->points)
            s.getVector4fMap() -= scene_centroid;

        LOG(INFO) << "visualizing " << model_kps_colored->points.size() << " and " << scene_kps_colored->points.size() << " for " << model_id;
        vis->addPointCloud(model_kps_colored, "model_kps", vp1);
        vis->addPointCloud(scene_kps_colored, "scene_kps", vp2);
        vis->addPointCloudNormals<PointT,pcl::Normal>(model_kps_colored, model_kps_normals, 1, 0.02f, "model_kp_normals", vp1);
        vis->addPointCloudNormals<PointT,pcl::Normal>(scene_kps_colored, scene_kps_normals, 1, 0.02f, "scene_kp_normals", vp2);
        vis->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "model_kps");
        vis->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "scene_kps");


        vis->addPointCloud(clustered_model_kps, "model_kps_clustered", vp1);
        vis->addPointCloud(clustered_scene_kps, "scene_kps_clustered", vp2);
        vis->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "model_kps_clustered");
        vis->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "scene_kps_clustered");


        typename pcl::PointCloud<PointT>::Ptr scene_vis (new pcl::PointCloud<PointT>(*scene_));
        scene_vis->sensor_origin_ = Eigen::Vector4f::Zero();
        scene_vis->sensor_orientation_ = Eigen::Quaternionf::Identity();
        vis->addPointCloud(scene_vis, "scene_current_view", vp2);
        vis->addPointCloud(scene_vis, "scene_current_view3", vp3);

        Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();

        if( !corresp_clusters.empty() )
        {
            typename pcl::registration::TransformationEstimationSVD < pcl::PointXYZ, pcl::PointXYZ > t_est;
            t_est.estimateRigidTransformation (*model_keypoints, *scene_cloud_xyz_merged_, corresp_clusters[0], tf);
        }
        tf(0,3) -= 0.3f;
        tf(1,3) -= 0.3f;
        tf(2,3) -= 0.3f;
        typename pcl::PointCloud<PointT>::Ptr model_cloud_aligned (new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*model_cloud, *model_cloud_aligned, tf);
        vis->addPointCloud(model_cloud_aligned, "aligned_cloud", vp3);

        size_t unique_id;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pts (new pcl::PointCloud<pcl::PointXYZRGB>);
        for(const pcl::Correspondence &c : *(loh.model_scene_corresp_))
        {
            pcl::PointXYZ m = model_keypoints->points[c.index_query];
            m.getVector4fMap() = tf * m.getVector4fMap();

            pcl::PointXYZ s = scene_cloud_xyz_merged_->points[c.index_match];

            pcl::PointXYZRGB p1, p2;
            p1.getVector3fMap() = s.getVector3fMap();
            p2.getVector3fMap() = m.getVector3fMap();
            p1.r = p2.r = 255.;
            p1.g = p2.g = 0.;
            p1.b = p2.b = 0.;
            pts->points.push_back(p1);
            pts->points.push_back(p2);

            std::stringstream unique_ss; unique_ss << "Line_" << unique_id++;
            vis->addLine(m, s, 1., 0., 0., unique_ss.str(), vp3);
            vis->setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, unique_ss.str());
        }

        unique_id=0;
        for( const pcl::Correspondences &cs : corresp_clusters )
        {
            for(const pcl::Correspondence &c : cs)
            {
                pcl::PointXYZ m = model_keypoints->points[c.index_query];
                m.getVector4fMap() = tf * m.getVector4fMap();

                pcl::PointXYZ s = scene_cloud_xyz_merged_->points[c.index_match];

                pcl::PointXYZRGB p1, p2;
                p1.getVector3fMap() = s.getVector3fMap();
                p2.getVector3fMap() = m.getVector3fMap();
                p1.r = p2.r = 0.;
                p1.g = p2.g = 0.;
                p1.b = p2.b = 255.;
                pts->points.push_back(p1);
                pts->points.push_back(p2);

                std::stringstream unique_ss; unique_ss << "Line_gc_" << unique_id++;
                vis->addLine(m, s, 0., 0., 1., unique_ss.str(), vp3);
                vis->setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, unique_ss.str());
            }
        }
        vis->addPointCloud(pts, "colored_pts", vp3);
        vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "colored_pts");


        Eigen::Matrix3f rot_tmp  = tf.block<3,3>(0,0);
        Eigen::Vector3f trans_tmp = tf.block<3,1>(0,3);
        Eigen::Affine3f affine_trans;
        affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());

//        if(co_init)
//        {
//            bool found = vis->updateCoordinateSystemPose("co", affine_trans);
//            std::cout << "found coordinate system: " << found << std::endl;
//        }
//        else
        {
            co_init = true;
//            vis->addCoordinateSystem(0.2f, affine_trans, "co", vp3);
        }
        vis->spin();
    }
}


template<typename PointT>
void
MultiviewRecognizer<PointT>::correspondenceGrouping ()
{
    pcl::StopWatch t;

//#pragma omp parallel for schedule(dynamic)
    typename std::map<std::string, LocalObjectHypothesis<PointT> >::const_iterator it;
    for ( it = local_obj_hypotheses_.begin (); it != local_obj_hypotheses_.end (); ++it )
    {
        const std::string &model_id = it->first;
        const LocalObjectHypothesis<PointT> &loh = it->second;

        std::stringstream desc; desc << "Correspondence grouping for " << model_id << " ( " << loh.model_scene_corresp_->size() << ")" ;
        typename RecognitionPipeline<PointT>::StopWatch t(desc.str());

        pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoints = model_keypoints_[model_id]->keypoints_;
        pcl::PointCloud<pcl::Normal>::Ptr model_kp_normals = model_keypoints_[model_id]->kp_normals_;

        if( loh.model_scene_corresp_->size() < 3 )
            continue;

        std::sort( loh.model_scene_corresp_->begin(), loh.model_scene_corresp_->end(), LocalObjectHypothesis<PointT>::gcGraphCorrespSorter);
        std::vector < pcl::Correspondences > corresp_clusters;
        cg_algorithm_->setSceneCloud ( scene_cloud_xyz_merged_ );
        cg_algorithm_->setInputCloud ( model_keypoints );

//        oh.visualize(*scene_, *scene_keypoints_);

        // Graph-based correspondence grouping requires normals but interface does not exist in base class - so need to try pointer casting
        typename GraphGeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ>::Ptr gcg_algorithm =
                boost::dynamic_pointer_cast<  GraphGeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> > (cg_algorithm_);
        if( gcg_algorithm )
            gcg_algorithm->setInputAndSceneNormals(model_kp_normals, scene_cloud_normals_merged_);

//        for ( const auto c : *(loh.model_scene_corresp_) )
//        {
//            CHECK( c.index_match < (int) scene_cloud_xyz->points.size() && c.index_match >= 0 );
//            CHECK( c.index_match < (int) scene_normals_->points.size() && c.index_match >= 0 );
//            CHECK( c.index_query < (int) model_keypoints->points.size() && c.index_query >= 0 );
//            CHECK( c.index_query < (int) model_kp_normals->points.size() && c.index_query >= 0 );
//        }

        //we need to pass the keypoints_pointcloud and the specific object hypothesis
        cg_algorithm_->setModelSceneCorrespondences ( loh.model_scene_corresp_ );
        cg_algorithm_->cluster (corresp_clusters);

        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > new_transforms (corresp_clusters.size());
        typename pcl::registration::TransformationEstimationSVD < pcl::PointXYZ, pcl::PointXYZ > t_est;

        for (size_t cluster_id = 0; cluster_id < corresp_clusters.size(); cluster_id++)
            t_est.estimateRigidTransformation (*model_keypoints, *scene_cloud_xyz_merged_, corresp_clusters[cluster_id], new_transforms[cluster_id]);

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

                for(size_t j=tf_id+1; j < new_transforms.size(); j++) {
                    const Eigen::Vector3f centroid2 = new_transforms[j].block<3, 1> (0, 3);
                    const Eigen::Matrix3f rot2 = new_transforms[j].block<3, 3> (0, 0);
                    const Eigen::Matrix3f rot_diff = rot2 * rot1.transpose();

                    double rotx = std::abs( atan2(rot_diff(2,1), rot_diff(2,2)));
                    double roty = std::abs( atan2(-rot_diff(2,0), sqrt(rot_diff(2,1) * rot_diff(2,1) + rot_diff(2,2) * rot_diff(2,2))) );
                    double rotz = std::abs( atan2(rot_diff(1,0), rot_diff(0,0)) );
                    double dist = (centroid1 - centroid2).norm();

                    if ( (dist < param_.merge_close_hypotheses_dist_) && (rotx < angle_thresh_rad) && (roty < angle_thresh_rad) && (rotz < angle_thresh_rad) ) {
                        merged_corrs.insert( merged_corrs.end(), corresp_clusters[j].begin(), corresp_clusters[j].end() );
                        cluster_has_been_taken[j] = true;
                    }
                }

                t_est.estimateRigidTransformation ( *model_keypoints, *scene_cloud_xyz_merged_, merged_corrs, merged_transforms[kept] );
                kept++;
            }
            merged_transforms.resize(kept);

            #pragma omp critical
            {
                for(size_t jj=0; jj<merged_transforms.size(); jj++)
                {
                    typename ObjectHypothesis::Ptr new_oh (new ObjectHypothesis);
                    new_oh->model_id_ = loh.model_id_;
                    new_oh->class_id_ = "";
                    new_oh->transform_ = merged_transforms[jj];
                    new_oh->confidence_ = corresp_clusters.size();
                    new_oh->corr_ = corresp_clusters[jj];

                    ObjectHypothesesGroup new_ohg;
                    new_ohg.global_hypotheses_ = false;
                    new_ohg.ohs_.push_back( new_oh );
                    obj_hypotheses_.push_back( new_ohg );
                }
                LOG(INFO) << "Merged " << corresp_clusters.size() << " clusters into " << kept << " clusters. Total correspondences: " << loh.model_scene_corresp_->size () << " " << loh.model_id_;
            }
        }
        else {
            #pragma omp critical
            {
                for(size_t jj=0; jj<new_transforms.size(); jj++)
                {
                    typename ObjectHypothesis::Ptr new_oh (new ObjectHypothesis);
                    new_oh->model_id_ = loh.model_id_;
                    new_oh->class_id_ = "";
                    new_oh->transform_ = new_transforms[jj];
                    new_oh->confidence_ = corresp_clusters.size();
                    new_oh->corr_ = corresp_clusters[jj];

                    ObjectHypothesesGroup new_ohg;
                    new_ohg.global_hypotheses_ = false;
                    new_ohg.ohs_.push_back( new_oh );
                    obj_hypotheses_.push_back( new_ohg );
                }
            }
        }
    }

    LOG(INFO) << "Correspondence Grouping took " << t.getTime();
}

template<typename PointT>
void
MultiviewRecognizer<PointT>::initialize(const std::string &trained_dir, bool retrain)
{
    recognition_pipeline_->initialize( trained_dir, retrain );

    if( param_.transfer_keypoint_correspondences_ ) // if correspondence grouping is done here, don't do it in multi-pipeline
    {
        typename MultiRecognitionPipeline<PointT>::Ptr mp_recognizer =
                boost::dynamic_pointer_cast<  MultiRecognitionPipeline<PointT> > (recognition_pipeline_);

        CHECK(mp_recognizer);

        std::vector<typename RecognitionPipeline<PointT>::Ptr > sv_rec_pipelines = mp_recognizer->getRecognitionPipelines();
        for( typename RecognitionPipeline<PointT>::Ptr &rec_pipeline : sv_rec_pipelines )
        {

            typename LocalRecognitionPipeline<PointT>::Ptr local_rec_pipeline =
                    boost::dynamic_pointer_cast<  LocalRecognitionPipeline<PointT> > (rec_pipeline);

            if(local_rec_pipeline)
                local_rec_pipeline->disableHypothesesGeneration();
        }
    }
}



template class V4R_EXPORTS MultiviewRecognizer<pcl::PointXYZRGB>;
}
