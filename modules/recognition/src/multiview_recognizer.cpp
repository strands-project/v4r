#include <glog/logging.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/recognition/multiview_recognizer.h>

namespace v4r
{

template<typename PointT>
void
MultiviewRecognizer<PointT>::recognize()
{
    const Eigen::Matrix4f camera_pose = v4r::RotTrans2Mat4f( scene_->sensor_orientation_, scene_->sensor_origin_ );

    recognition_pipeline_->setInputCloud( scene_ );
    recognition_pipeline_->setSceneNormals( scene_normals_ );

    if( table_plane_set_ )
        recognition_pipeline_->setTablePlane( table_plane_ );

    recognition_pipeline_->recognize();

    table_plane_set_ = false;

    std::vector<ObjectHypothesesGroup<PointT> > ohg_view = recognition_pipeline_->getObjectHypothesis();

    for(ObjectHypothesesGroup<PointT> &ohg_tmp : ohg_view)
    {
        for( typename ObjectHypothesis<PointT>::Ptr &oh_tmp : ohg_tmp.ohs_)
        {
            oh_tmp->transform_ = camera_pose * oh_tmp->transform_;
        }
    }

    obj_hypotheses_.insert( obj_hypotheses_.end(), ohg_view.begin(), ohg_view.end() );
}

template class V4R_EXPORTS MultiviewRecognizer<pcl::PointXYZRGB>;
}
