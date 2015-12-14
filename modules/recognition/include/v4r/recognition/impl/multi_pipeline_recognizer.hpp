/*
 * multi_pipeline_recognizer.h
 *
 *  Created on: Feb 24, 2013
 *      Author: aitor
 */

#ifndef MULTI_PIPELINE_RECOGNIZER_HPP_
#define MULTI_PIPELINE_RECOGNIZER_HPP_

#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <v4r/common/miscellaneous.h>

namespace v4r
{

template<typename PointT>
bool
MultiRecognitionPipeline<PointT>::initialize(bool force_retrain)
{
    for(int i=0; i < (int)recognizers_.size(); i++) {
        if(!recognizers_[i]->initialize(force_retrain)) {   // if model database changed, train whole model database again and start all over
            reinitialize();
        }
    }

    if(param_.icp_iterations_ > 0 && param_.icp_type_ == 1)
    {
        for(size_t i=0; i < recognizers_.size(); i++)
            recognizers_[i]->getDataSource()->createVoxelGridAndDistanceTransform(param_.voxel_size_icp_);
    }

    return true;
}

template<typename PointT>
void
MultiRecognitionPipeline<PointT>::reinitialize()
{
    for(size_t i=0; i < recognizers_.size(); i++)
        recognizers_[i]->reinitializeSourceOnly();

    for(size_t i=0; i < recognizers_.size(); i++)
        recognizers_[i]->reinitializeRecOnly();

    initialize(false);
}


//template<typename PointT>
//void
//MultiRecognitionPipeline<PointT>::reinitialize()
//{

//    // reinitialize source (but be aware to not do it twice)
//    std::vector<boost::shared_ptr<Source<PointT> > > common_sources;

//    for(size_t i=0; i < recognizers_.size(); i++) {
//        boost::shared_ptr<Source<PointT> > src = recognizers_[i]->getDataSource();

//        bool src_is_already_shared_by_other_recognizer = false;
//        for(size_t jj=0; jj<common_sources; jj++)
//        {
//            if ( src == common_sources ) {
//                src_is_already_shared_by_other_recognizer = true;
//                break;
//            }
//        }
//        if (!src_is_already_shared_by_other_recognizer)
//            common_sources.push_back(src);
//    }


//    for(size_t i=0; i < recognizers_.size(); i++)
//        recognizers_[i]->reinitialize();

//    initialize();
//}

template<typename PointT>
void
MultiRecognitionPipeline<PointT>::getPoseRefinement(
        const std::vector<ModelTPtr> &models,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms)
{
    models_ = models;
    transforms_ = transforms;
    poseRefinement();
    transforms = transforms_; //is this neccessary?
}

template<typename PointT>
void
MultiRecognitionPipeline<PointT>::recognize()
{
    models_.clear();
    transforms_.clear();

    //first version... just call each recognizer independently...
    //more advanced version should compute normals and preprocess the input cloud so that
    //we avoid recomputing stuff shared among the different pipelines
    std::vector<int> input_icp_indices;

    std::cout << "Number of recognizers:" << recognizers_.size() << std::endl;

    //typename std::map<std::string, ObjectHypothesis<PointT> > object_hypotheses_;
    obj_hypotheses_.clear();
    scene_keypoints_.reset(new pcl::PointCloud<PointT>);
    scene_kp_indices_.indices.clear();

    for(size_t i=0; i < recognizers_.size(); i++)
    {
        recognizers_[i]->setInputCloud(scene_);

        if(recognizers_[i]->requiresSegmentation()) // this might not work in the current state!!
        {
            if( recognizers_[i]->acceptsNormals() )
            {
                if ( !scene_normals_ || scene_normals_->points.size() != scene_->points.size() )
                    computeNormals<PointT>(scene_, scene_normals_, param_.normal_computation_method_);

                recognizers_[i]->setSceneNormals(scene_normals_);
            }

            for(size_t c=0; c < segmentation_indices_.size(); c++)
            {
                recognizers_[i]->recognize();
                std::vector<ModelTPtr> models_tmp = recognizers_[i]->getModels ();
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_tmp = recognizers_[i]->getTransforms ();

                models_.insert(models_.end(), models_tmp.begin(), models_tmp.end());
                transforms_.insert(transforms_.end(), transforms_tmp.begin(), transforms_tmp.end());
                input_icp_indices.insert(input_icp_indices.end(), segmentation_indices_[c].indices.begin(), segmentation_indices_[c].indices.end());
            }
        }
        else
        {
//            recognizers_[i]->setSaveHypotheses(param_.save_hypotheses_);  // shouldn't this be false?
            recognizers_[i]->recognize();

            if(!recognizers_[i]->getSaveHypothesesParam())
            {
                std::vector<ModelTPtr> models = recognizers_[i]->getModels ();
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms = recognizers_[i]->getTransforms ();

                models_.insert(models_.end(), models.begin(), models.end());
                transforms_.insert(transforms_.end(), transforms.begin(), transforms.end());
            }
            else
            {
                typename std::map<std::string, ObjectHypothesis<PointT> > oh_tmp;
                recognizers_[i]->getSavedHypotheses(oh_tmp);

                pcl::PointIndices kp_idx_tmp;
                typename pcl::PointCloud<PointT>::Ptr kp_tmp(new pcl::PointCloud<PointT>);
                recognizers_[i]->getKeypointCloud(kp_tmp);
                recognizers_[i]->getKeypointIndices(kp_idx_tmp);

                *scene_keypoints_ += *kp_tmp;

                typename std::map<std::string, ObjectHypothesis<PointT> >::iterator it_mp_oh;

                typename std::map<std::string, ObjectHypothesis<PointT> >::iterator it_tmp;
                for (it_tmp = oh_tmp.begin (); it_tmp != oh_tmp.end (); ++it_tmp)
                {
                    const std::string id = it_tmp->second.model_->id_;

                    it_mp_oh = obj_hypotheses_.find(id);
                    if(it_mp_oh == obj_hypotheses_.end())   // no feature correspondences exist yet
                        obj_hypotheses_.insert(std::pair<std::string, ObjectHypothesis<PointT> >(id, it_tmp->second));
                    else
                    {
                        ObjectHypothesis<PointT> &oh = it_mp_oh->second;
                        const ObjectHypothesis<PointT> &new_oh = it_tmp->second;
                        oh.model_scene_corresp_->insert(     oh.model_scene_corresp_->  end(),
                                                         new_oh.model_scene_corresp_->begin(),
                                                         new_oh.model_scene_corresp_->  end() );
                    }
                }
            }
        }
    }

    if( !param_.save_hypotheses_ && cg_algorithm_)
    {
        correspondenceGrouping();

        if (param_.icp_iterations_ > 0 || hv_algorithm_)
        {
            //Prepare scene and model clouds for the pose refinement step
            getDataSource()->voxelizeAllModels (param_.voxel_size_icp_);
        }

        if ( param_.icp_iterations_ > 0 )
            poseRefinement();

        if ( hv_algorithm_ && models_.size() )
            hypothesisVerification();
    }

    scene_normals_.reset();
}

template<typename PointT>
void MultiRecognitionPipeline<PointT>::correspondenceGrouping ()
{
    if(cg_algorithm_->getRequiresNormals() && (!scene_normals_ || scene_normals_->points.size() != scene_->points.size()))
        computeNormals<PointT>(scene_, scene_normals_, param_.normal_computation_method_);

    typename std::map<std::string, ObjectHypothesis<PointT> >::iterator it;
    for (it = obj_hypotheses_.begin (); it != obj_hypotheses_.end (); ++it)
    {
        ObjectHypothesis<PointT> &oh = it->second;

        if(oh.model_scene_corresp_->size() < 3)
            continue;

        std::vector < pcl::Correspondences > corresp_clusters;
        cg_algorithm_->setSceneCloud (scene_);
        cg_algorithm_->setInputCloud (oh.model_->keypoints_);

        if(cg_algorithm_->getRequiresNormals())
            cg_algorithm_->setInputAndSceneNormals(oh.model_->kp_normals_, scene_normals_);

        //we need to pass the keypoints_pointcloud and the specific object hypothesis
        cg_algorithm_->setModelSceneCorrespondences (oh.model_scene_corresp_);
        cg_algorithm_->cluster (corresp_clusters);

        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > new_transforms (corresp_clusters.size());
        typename pcl::registration::TransformationEstimationSVD < PointT, PointT > t_est;

        for (size_t i = 0; i < corresp_clusters.size(); i++)
            t_est.estimateRigidTransformation (*oh.model_->keypoints_, *scene_, corresp_clusters[i], new_transforms[i]);

        if(param_.merge_close_hypotheses_) {
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > merged_transforms (corresp_clusters.size());
            std::vector<bool> cluster_has_been_taken(corresp_clusters.size(), false);
            const double angle_thresh_rad = param_.merge_close_hypotheses_angle_ * M_PI / 180.f ;

            size_t kept=0;
            for (size_t i = 0; i < new_transforms.size(); i++) {

                if (cluster_has_been_taken[i])
                    continue;

                cluster_has_been_taken[i] = true;
                const Eigen::Vector3f centroid1 = new_transforms[i].block<3, 1> (0, 3);
                const Eigen::Matrix3f rot1 = new_transforms[i].block<3, 3> (0, 0);

                pcl::Correspondences merged_corrs = corresp_clusters[i];

                for(size_t j=i; j < new_transforms.size(); j++) {
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

                t_est.estimateRigidTransformation (*oh.model_->keypoints_, *scene_, merged_corrs, merged_transforms[kept]);
                kept++;
            }
            merged_transforms.resize(kept);
            new_transforms = merged_transforms;
        }

        std::cout << "Merged " << corresp_clusters.size() << " clusters into " << new_transforms.size() << " clusters. Total correspondences: " << oh.model_scene_corresp_->size () << " " << it->first << std::endl;

        //        oh.visualize(*scene_);

        size_t existing_hypotheses = models_.size();
        models_.resize( existing_hypotheses + new_transforms.size(), oh.model_  );
        transforms_.insert(transforms_.end(), new_transforms.begin(), new_transforms.end());
    }
}

template<typename PointT>
bool
MultiRecognitionPipeline<PointT>::isSegmentationRequired() const
{
    bool ret_value = false;
    for(size_t i=0; (i < recognizers_.size()) && !ret_value; i++)
        ret_value = recognizers_[i]->requiresSegmentation();

    return ret_value;
}

template<typename PointT>
typename boost::shared_ptr<Source<PointT> >
MultiRecognitionPipeline<PointT>::getDataSource () const
{
    //NOTE: Assuming source is the same or contains the same models for all recognizers...
    //Otherwise, we should create a combined data source so that all models are present

    return recognizers_[0]->getDataSource();
}

}

#endif /* MULTI_PIPELINE_RECOGNIZER_H_ */
