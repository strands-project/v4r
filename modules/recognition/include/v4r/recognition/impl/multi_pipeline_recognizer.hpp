/*
 * multi_pipeline_recognizer.h
 *
 *  Created on: Feb 24, 2013
 *      Author: aitor
 */

#ifndef MULTI_PIPELINE_RECOGNIZER_HPP_
#define MULTI_PIPELINE_RECOGNIZER_HPP_

#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/common/normal_estimator.h>
//#include "multi_object_graph_CG.h"
//#include <pcl/visualization/pcl_visualizer.h>

template<typename PointInT>
void
v4r::MultiRecognitionPipeline<PointInT>::initialize()
{
    if(ICP_iterations_ > 0 && icp_type_ == 1)
    {
        for(size_t i=0; i < recognizers_.size(); i++)
            recognizers_[i]->getDataSource()->createVoxelGridAndDistanceTransform(VOXEL_SIZE_ICP_);
    }
}

template<typename PointInT>
void
v4r::MultiRecognitionPipeline<PointInT>::reinitialize()
{
    for(size_t i=0; i < recognizers_.size(); i++)
        recognizers_[i]->reinitialize();

    initialize();
}

template<typename PointInT>
void
v4r::MultiRecognitionPipeline<PointInT>::reinitialize(const std::vector<std::string> & load_ids)
{
    for(size_t i=0; i < recognizers_.size(); i++)
        recognizers_[i]->reinitialize(load_ids);

    initialize();
}

template<typename PointInT>
void
v4r::MultiRecognitionPipeline<PointInT>::getPoseRefinement(
        const std::vector<ModelTPtr> &models,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms)
{
    models_ = models;
    transforms_ = transforms;
    poseRefinement();
    transforms = transforms_; //is this neccessary?
}

template<typename PointInT>
void
v4r::MultiRecognitionPipeline<PointInT>::recognize()
{
    models_.clear();
    transforms_.clear();

    //first version... just call each recognizer independently...
    //more advanced version should compute normals and preprocess the input cloud so that
    //we avoid recomputing stuff shared among the different pipelines
    std::vector<int> input_icp_indices;
    if(cg_algorithm_)
        set_save_hypotheses_ = true;

    std::cout << "Number of recognizers:" << recognizers_.size() << std::endl;

    //typename std::map<std::string, ObjectHypothesis<PointInT> > object_hypotheses_;
    object_hypotheses_mp_.clear();
    scene_keypoints_.reset(new pcl::PointCloud<PointInT>);
    scene_kp_indices_.indices.clear();

    for(size_t i=0; i < recognizers_.size(); i++)
    {
        recognizers_[i]->setInputCloud(scene_);

        if(recognizers_[i]->requiresSegmentation())
        {
            if(recognizers_[i]->acceptsNormals() && normals_set_)
                recognizers_[i]->setSceneNormals(scene_normals_);

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
            recognizers_[i]->setSaveHypotheses(set_save_hypotheses_);
            recognizers_[i]->setIndices(indices_);
            recognizers_[i]->recognize();

            if(!set_save_hypotheses_)
            {
                std::vector<ModelTPtr> models = recognizers_[i]->getModels ();
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms = recognizers_[i]->getTransforms ();

                models_.insert(models_.end(), models.begin(), models.end());
                transforms_.insert(transforms_.end(), transforms.begin(), transforms.end());
            }
            else
            {
                typename std::map<std::string, ObjectHypothesis<PointInT> > oh_tmp;
                recognizers_[i]->getSavedHypotheses(oh_tmp);

                pcl::PointIndices kp_idx_tmp;
                typename pcl::PointCloud<PointInT>::Ptr kp_tmp(new pcl::PointCloud<PointInT>);
                recognizers_[i]->getKeypointCloud(kp_tmp);
                recognizers_[i]->getKeypointIndices(kp_idx_tmp);

                *scene_keypoints_ += *kp_tmp;

                typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it_mp_oh;

                typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it_tmp;
                for (it_tmp = oh_tmp.begin (); it_tmp != oh_tmp.end (); it_tmp++)
                {
                    const std::string id = it_tmp->second.model_->id_;

                    it_mp_oh = object_hypotheses_mp_.find(id);
                    if(it_mp_oh == object_hypotheses_mp_.end())   // no feature correspondences exist yet
                        object_hypotheses_mp_.insert(std::pair<std::string, ObjectHypothesis<PointInT> >(id, it_tmp->second));
                    else
                    {
                        ObjectHypothesis<PointInT> &oh = it_mp_oh->second;
                        const ObjectHypothesis<PointInT> &new_oh = it_tmp->second;
                        oh.model_scene_corresp_->insert(     oh.model_scene_corresp_->  end(),
                                                         new_oh.model_scene_corresp_->begin(),
                                                         new_oh.model_scene_corresp_->  end() );
                    }
                }
            }

            input_icp_indices.insert(input_icp_indices.end(), indices_.begin(), indices_.end());
        }
    }

    if(cg_algorithm_)
        correspondenceGrouping();

    if ((ICP_iterations_ > 0 || hv_algorithm_)  && cg_algorithm_) {
        //Prepare scene and model clouds for the pose refinement step

        std::sort( input_icp_indices.begin(), input_icp_indices.end() );
        input_icp_indices.erase( std::unique( input_icp_indices.begin(), input_icp_indices.end() ), input_icp_indices.end() );

        pcl::PointIndices ind;
        ind.indices = input_icp_indices;
        icp_scene_indices_.reset(new pcl::PointIndices(ind));
        getDataSource()->voxelizeAllModels (VOXEL_SIZE_ICP_);
    }

    if ( ICP_iterations_ > 0  && cg_algorithm_ )
        poseRefinement();

    if ( hv_algorithm_ && models_.size() && cg_algorithm_ )
        hypothesisVerification();
}

template<typename PointInT>
void v4r::MultiRecognitionPipeline<PointInT>::correspondenceGrouping()
{
//    pcl::PointCloud<pcl::Normal>::Ptr scene_kp_normals(new pcl::PointCloud<pcl::Normal>);
    if(cg_algorithm_->getRequiresNormals())
    {
        pcl::PointCloud<pcl::Normal>::Ptr all_scene_normals;

        //compute them...
        PCL_WARN("Need to compute normals due to the cg algorithm\n");
        all_scene_normals.reset(new pcl::PointCloud<pcl::Normal>);
        PointInTPtr processed (new pcl::PointCloud<PointInT>);

        if(!normals_set_)
        {
            boost::shared_ptr<v4r::PreProcessorAndNormalEstimator<PointInT, pcl::Normal> > normal_estimator;
            normal_estimator.reset (new v4r::PreProcessorAndNormalEstimator<PointInT, pcl::Normal>);
            normal_estimator->setCMR (false);
            normal_estimator->setDoVoxelGrid (false);
            normal_estimator->setRemoveOutliers (false);
            normal_estimator->setValuesForCMRFalse (0.003f, 0.02f);
            normal_estimator->setForceUnorganized(true);
            normal_estimator->estimate (scene_, processed, all_scene_normals);
        }
        else
        {
            PCL_WARN("Using scene normals given from user code\n");
            processed = scene_;
            all_scene_normals = scene_normals_;
        }
//        pcl::copyPointCloud(*all_scene_normals, scene_kp_indices_.indices, *scene_kp_normals);
    }

    typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it_map;
    for (it_map = object_hypotheses_mp_.begin (); it_map != object_hypotheses_mp_.end (); it_map++)
    {
        ObjectHypothesis<PointInT> &oh = it_map->second;

        if(oh.model_scene_corresp_->size() < 3)
            continue;

        oh.scene_normals_ = scene_normals_;

        std::vector < pcl::Correspondences > corresp_clusters;

        if(cg_algorithm_->getRequiresNormals())
            cg_algorithm_->setInputAndSceneNormals(oh.model_->kp_normals_, oh.scene_normals_);

        cg_algorithm_->setSceneCloud (oh.scene_);
        cg_algorithm_->setInputCloud (oh.model_->keypoints_);
        cg_algorithm_->setModelSceneCorrespondences (oh.model_scene_corresp_);
        cg_algorithm_->cluster (corresp_clusters);

        std::cout << "Instances:" << corresp_clusters.size () << " Total correspondences:" << oh.model_scene_corresp_->size () << " " << it_map->first << std::endl;

        for (size_t i = 0; i < corresp_clusters.size (); i++)
        {
            Eigen::Matrix4f best_trans;
            typename pcl::registration::TransformationEstimationSVD < PointInT, PointInT > t_est;
            t_est.estimateRigidTransformation (*oh.model_->keypoints_, *oh.scene_, corresp_clusters[i], best_trans);

            models_.push_back (oh.model_);
            transforms_.push_back (best_trans);
        }
    }
}

template<typename PointInT>
bool
v4r::MultiRecognitionPipeline<PointInT>::isSegmentationRequired() const
{
    bool ret_value = false;
    for(size_t i=0; (i < recognizers_.size()) && !ret_value; i++)
        ret_value = recognizers_[i]->requiresSegmentation();

    return ret_value;
}

template<typename PointInT>
typename boost::shared_ptr<v4r::Source<PointInT> >
v4r::MultiRecognitionPipeline<PointInT>::getDataSource () const
{
    //NOTE: Assuming source is the same or contains the same models for all recognizers...
    //Otherwise, we should create a combined data source so that all models are present

    return recognizers_[0]->getDataSource();
}

#endif /* MULTI_PIPELINE_RECOGNIZER_H_ */
