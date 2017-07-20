#include <glog/logging.h>
#include <omp.h>

#include <v4r/recognition/multi_pipeline_recognizer.h>

namespace v4r
{

template<typename PointT>
void
MultiRecognitionPipeline<PointT>::initialize(const std::string &trained_dir, bool force_retrain)
{
    for(auto &r:recognition_pipelines_)
    {
        r->setModelDatabase(m_db_);
        r->setNormalEstimator(normal_estimator_);
        r->setVisualizationParameter(vis_param_);
        r->initialize(trained_dir, force_retrain);
    }
}


template<typename PointT>
void
MultiRecognitionPipeline<PointT>::do_recognize()
{
    omp_init_lock(&rec_lock_);

//#pragma omp parallel for schedule(dynamic)
    for(size_t r_id=0; r_id < recognition_pipelines_.size(); r_id++)
    {
        typename RecognitionPipeline<PointT>::Ptr r = recognition_pipelines_[r_id];
        r->setInputCloud( scene_ );
        r->setSceneNormals( scene_normals_ );

        if( table_plane_set_ )
            r->setTablePlane( table_plane_ );

        r->recognize();

        std::vector<ObjectHypothesesGroup> oh_tmp = r->getObjectHypothesis();
        omp_set_lock(&rec_lock_);
        obj_hypotheses_.insert( obj_hypotheses_.end(), oh_tmp.begin(), oh_tmp.end() );
        omp_unset_lock(&rec_lock_);

        std::vector< std::pair<std::string,float> > elapsed_times_tmp = r->getElapsedTimes();
        elapsed_time_.insert( elapsed_time_.end(), elapsed_times_tmp.begin(), elapsed_times_tmp.end() );
    }

    omp_destroy_lock(&rec_lock_);

    table_plane_set_ = false;
}

template class V4R_EXPORTS MultiRecognitionPipeline<pcl::PointXYZRGB>;
//template class V4R_EXPORTS MultiRecognitionPipeline<pcl::PointXYZ>;    // maybe this doesn't work because of the specialized template in the initialization function (constructor)
}
