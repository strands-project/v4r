#include <algorithm>
#include <glog/logging.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/recognition/multiview_recognizer.h>

namespace v4r
{

template<typename PointT>
void
MultiviewRecognizer<PointT>::recognize()
{
    obj_hypotheses_.clear();

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

    v.obj_hypotheses_ = obj_hypotheses_;
    views_.push_back(v);
}

template class V4R_EXPORTS MultiviewRecognizer<pcl::PointXYZRGB>;
}
