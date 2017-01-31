#include <glog/logging.h>
#include <v4r/recognition/multiview_recognizer.h>

namespace v4r
{

template<typename PointT>
void
MultiviewRecognizer<PointT>::recognize()
{
    for( const View &v : views_ )
    {
        recognition_pipeline_->setInputCloud( v.cloud_ );

        if( v.cloud_normals_ )
            recognition_pipeline_->setSceneNormals( v.cloud_normals_ );

        recognition_pipeline_->recognize();
        std::vector<ObjectHypothesesGroup<PointT> > ohg_view = recognition_pipeline_->getObjectHypothesis();

        for(ObjectHypothesesGroup<PointT> &ohg_tmp : ohg_view)
        {
            for( typename ObjectHypothesis<PointT>::Ptr &oh_tmp : ohg_tmp.ohs_)
            {
                oh_tmp->transform_ = v.camera_pose_ * oh_tmp->transform_;
            }
        }

        obj_hypotheses_.insert( obj_hypotheses_.end(), ohg_view.begin(), ohg_view.end() );
    }

    views_.clear();
}

template class V4R_EXPORTS MultiviewRecognizer<pcl::PointXYZRGB>;
}
