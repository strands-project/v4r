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


    // now add the old hypotheses
    for(const View v_old : views_)
    {
        for(const ObjectHypothesesGroup &ohg_tmp : v_old.obj_hypotheses_)
        {
            bool hyp_exists = false;
            for( const typename ObjectHypothesis::Ptr &oh_tmp : ohg_tmp.ohs_)
            {
                if( !param_.transfer_only_verified_hypotheses_ || oh_tmp->is_verified_ )
                {
                    hyp_exists = true;
                    break;
                }
            }

            if( hyp_exists || !param_.transfer_only_verified_hypotheses_ )
            {
                ObjectHypothesesGroup ohg;
                ohg.global_hypotheses_ = ohg_tmp.global_hypotheses_;

                for( const typename ObjectHypothesis::Ptr &oh_tmp : ohg_tmp.ohs_)
                {
                    if( param_.transfer_only_verified_hypotheses_ && !oh_tmp->is_verified_ )
                        continue;

                    // create a copy (since we want to reset verification status and update transform but keep the status for old view)
                    typename ObjectHypothesis::Ptr oh_copy ( new ObjectHypothesis);
                    *oh_copy = *oh_tmp;

                    oh_copy->is_verified_ = false;
                    oh_copy->transform_ = v.camera_pose_.inverse() * v_old.camera_pose_ * oh_copy->transform_;
                    ohg.ohs_.push_back( oh_copy );
                }

                obj_hypotheses_.push_back(ohg);
            }
        }
    }

    views_.push_back(v);
}

template class V4R_EXPORTS MultiviewRecognizer<pcl::PointXYZRGB>;
}
