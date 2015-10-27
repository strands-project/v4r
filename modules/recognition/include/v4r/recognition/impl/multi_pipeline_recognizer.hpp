/*
 * multi_pipeline_recognizer.h
 *
 *  Created on: Feb 24, 2013
 *      Author: aitor
 */

#ifndef MULTI_PIPELINE_RECOGNIZER_HPP_
#define MULTI_PIPELINE_RECOGNIZER_HPP_

#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/common/miscellaneous.h>

namespace v4r
{

template<typename PointT>
void
MultiRecognitionPipeline<PointT>::initialize()
{
    if(param_.icp_iterations_ > 0 && param_.icp_type_ == 1)
    {
        for(size_t i=0; i < recognizers_.size(); i++)
            recognizers_[i]->getDataSource()->createVoxelGridAndDistanceTransform(param_.voxel_size_icp_);
    }
}

template<typename PointT>
void
MultiRecognitionPipeline<PointT>::reinitialize()
{
    for(size_t i=0; i < recognizers_.size(); i++)
        recognizers_[i]->reinitialize();

    initialize();
}

template<typename PointT>
void
MultiRecognitionPipeline<PointT>::reinitialize(const std::vector<std::string> & load_ids)
{
    for(size_t i=0; i < recognizers_.size(); i++)
        recognizers_[i]->reinitialize(load_ids);

    initialize();
}

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

        if(recognizers_[i]->requiresSegmentation())
        {
            if( recognizers_[i]->acceptsNormals() )
            {
                if ( !scene_normals_ || scene_normals_->points.size() != scene_->points.size() )
                    computeNormals<PointT>(scene_, scene_normals_, param_.normal_computation_method_);

                recognizers_[i]->setSceneNormals(scene_normals_);
            }

            for(size_t c=0; c < segmentation_indices_.size(); c++)
            {
                recognizers_[i]->setIndices(segmentation_indices_[c].indices);
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
            recognizers_[i]->setIndices(indices_);
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

            input_icp_indices.insert(input_icp_indices.end(), indices_.begin(), indices_.end());
        }
    }

    if( !param_.save_hypotheses_ && cg_algorithm_)
    {
        correspondenceGrouping();

        if (param_.icp_iterations_ > 0 || hv_algorithm_)
        {
            //Prepare scene and model clouds for the pose refinement step

            std::sort( input_icp_indices.begin(), input_icp_indices.end() );
            input_icp_indices.erase( std::unique( input_icp_indices.begin(), input_icp_indices.end() ), input_icp_indices.end() );

            pcl::PointIndices ind;
            ind.indices = input_icp_indices;
            icp_scene_indices_.reset(new pcl::PointIndices(ind));
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

    typename std::map<std::string, ObjectHypothesis<PointT> >::iterator it_map;
    for (it_map = obj_hypotheses_.begin (); it_map != obj_hypotheses_.end (); ++it_map)
    {
        ObjectHypothesis<PointT> &oh = it_map->second;

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

        std::cout << "Instances: " << corresp_clusters.size () << ", total correspondences: " << oh.model_scene_corresp_->size () << " " << it_map->first << std::endl;

        size_t existing_hypotheses = models_.size();
        models_.resize( existing_hypotheses + corresp_clusters.size () );
        transforms_.resize( existing_hypotheses + corresp_clusters.size () );

        for (size_t i = 0; i < corresp_clusters.size (); i++)
        {
            models_[existing_hypotheses + i] = oh.model_;
            typename pcl::registration::TransformationEstimationSVD < PointT, PointT > t_est;
            t_est.estimateRigidTransformation (*oh.model_->keypoints_, *scene_, corresp_clusters[i], transforms_[existing_hypotheses + i]);
        }
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
