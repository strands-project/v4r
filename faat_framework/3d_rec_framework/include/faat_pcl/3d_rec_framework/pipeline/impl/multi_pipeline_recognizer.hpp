/*
 * multi_pipeline_recognizer.h
 *
 *  Created on: Feb 24, 2013
 *      Author: aitor
 */

#ifndef MULTI_PIPELINE_RECOGNIZER_HPP_
#define MULTI_PIPELINE_RECOGNIZER_HPP_

#include <faat_pcl/3d_rec_framework/pipeline/multi_pipeline_recognizer.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/normal_estimator.h>

template<typename PointInT>
void
faat_pcl::rec_3d_framework::MultiRecognitionPipeline<PointInT>::initialize()
{
  if(ICP_iterations_ > 0 && icp_type_ == 1)
  {
    for(size_t i=0; i < recognizers_.size(); i++)
    {
      recognizers_[i]->getDataSource()->createVoxelGridAndDistanceTransform(VOXEL_SIZE_ICP_);
    }
  }
}

template<typename PointInT>
void
faat_pcl::rec_3d_framework::MultiRecognitionPipeline<PointInT>::recognize()
{

  /*if(models_)
    models_->clear();

  if(transforms_)
    transforms_->clear();*/

  models_.reset (new std::vector<ModelTPtr>);
  transforms_.reset (new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);

  //first version... just call each recognizer independently...
  //more advanced version should compute normals and preprocess the input cloud so that
  //we avoid recomputing stuff shared among the different pipelines
  std::vector<int> input_icp_indices;
  bool set_save_hypotheses = false;
  if(cg_algorithm_)
    set_save_hypotheses = true;

  std::cout << "set_save_hypotheses:" << set_save_hypotheses << std::endl;
  std::cout << "recognizers size:" << recognizers_.size() << std::endl;

  typename std::map<std::string, ObjectHypothesis<PointInT> > object_hypotheses;
  typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it_map_oh;
  typename pcl::PointCloud<PointInT>::Ptr keypoints_cloud(new pcl::PointCloud<PointInT>);

  for(size_t i=0; (i < recognizers_.size()); i++)
  {
    recognizers_[i]->setInputCloud(input_);

    if(recognizers_[i]->requiresSegmentation())
    {
      PCL_WARN("this recognizers requires segmentation...\n");
      if(recognizers_[i]->acceptsNormals() && normals_set_)
      {
          PCL_WARN("recognizer accepts normals, setting them\n");
          recognizers_[i]->setSceneNormals(scene_normals_);
      }
      else
      {
          std::cout << "normals set:" << normals_set_ << std::endl;
          std::cout << "recognizer accepts normals:" << recognizers_[i]->acceptsNormals() << std::endl;
      }

      for(size_t c=0; c < segmentation_indices_.size(); c++)
      {
        recognizers_[i]->setIndices(segmentation_indices_[c].indices);
        recognizers_[i]->recognize();
        boost::shared_ptr < std::vector<ModelTPtr> > models = recognizers_[i]->getModels ();
        boost::shared_ptr < std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms = recognizers_[i]->getTransforms ();

        models_->insert(models_->end(), models->begin(), models->end());
        transforms_->insert(transforms_->end(), transforms->begin(), transforms->end());
        input_icp_indices.insert(input_icp_indices.end(), segmentation_indices_[c].indices.begin(), segmentation_indices_[c].indices.end());
      }
    }
    else
    {
      recognizers_[i]->setSaveHypotheses(set_save_hypotheses);
      recognizers_[i]->setIndices(indices_);
      recognizers_[i]->recognize();
      boost::shared_ptr < std::vector<ModelTPtr> > models = recognizers_[i]->getModels ();
      boost::shared_ptr < std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms = recognizers_[i]->getTransforms ();

      if(!set_save_hypotheses)
      {
        models_->insert(models_->end(), models->begin(), models->end());
        transforms_->insert(transforms_->end(), transforms->begin(), transforms->end());
      }
      else
      {
        typename std::map<std::string, ObjectHypothesis<PointInT> > object_hypotheses_single_pipeline;
        typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it_map;
        recognizers_[i]->getSavedHypotheses(object_hypotheses_single_pipeline);
        typename pcl::PointCloud<PointInT>::Ptr keypoints_cloud_single(new pcl::PointCloud<PointInT>);
        recognizers_[i]->getKeypointCloud(keypoints_cloud_single);

        for (it_map = object_hypotheses_single_pipeline.begin ();
             it_map != object_hypotheses_single_pipeline.end (); it_map++)
        {
          std::string id = it_map->second.model_->id_;
          //std::cout << id << " " << it_map->second.correspondences_to_inputcloud->size() << std::endl;

          it_map_oh = object_hypotheses.find(id);
          if(it_map_oh == object_hypotheses.end())
          {
            object_hypotheses[id] = it_map->second;
          }
          else
          {
            //std::cout << "found... need to merge hypotheses" << std::endl;
            /*
                typename pcl::PointCloud<PointInT>::Ptr correspondences_pointcloud; //points in model coordinates
                pcl::PointCloud<pcl::Normal>::Ptr normals_pointcloud; //points in model coordinates
                boost::shared_ptr<std::vector<float> > feature_distances_;
                pcl::CorrespondencesPtr correspondences_to_inputcloud; //indices between correspondences_pointcloud and scene cloud (keypoints extracted by each local_recognizer)
                int num_corr_;
                std::vector<int> indices_to_flann_models_;
             */

            //update correspondences indices so they match each recognizer (keypoints_cloud to correspondences_pointcloud)
            for(size_t kk=0; kk < it_map->second.correspondences_to_inputcloud->size(); kk++)
            {
              pcl::Correspondence c = it_map->second.correspondences_to_inputcloud->at(kk);
              c.index_match += keypoints_cloud->points.size();
              c.index_query += it_map_oh->second.correspondences_pointcloud->points.size();
              it_map_oh->second.correspondences_to_inputcloud->push_back(c);
            }

            *it_map_oh->second.correspondences_pointcloud += * it_map->second.correspondences_pointcloud;
            *it_map_oh->second.normals_pointcloud += * it_map->second.normals_pointcloud;
            it_map_oh->second.feature_distances_->insert(it_map_oh->second.feature_distances_->end(),
                                                          it_map->second.feature_distances_->begin(),
                                                          it_map->second.feature_distances_->end());

            it_map_oh->second.num_corr_ += it_map->second.num_corr_;
          }
        }

        *keypoints_cloud += *keypoints_cloud_single;
      }

      input_icp_indices.insert(input_icp_indices.end(), indices_.begin(), indices_.end());
    }
  }

  if(set_save_hypotheses && object_hypotheses.size() > 0)
  {
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
    if(cg_algorithm_->getRequiresNormals())
    {
      pcl::PointCloud<pcl::Normal>::Ptr all_scene_normals;

      //compute them...
      PCL_ERROR("Need to compute normals due to the cg algorithm\n");
      all_scene_normals.reset(new pcl::PointCloud<pcl::Normal>);
      PointInTPtr processed (new pcl::PointCloud<PointInT>);

      if(!normals_set_)
      {
          pcl::ScopeTime t("compute normals\n");
          boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointInT, pcl::Normal> > normal_estimator;
          normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointInT, pcl::Normal>);
          normal_estimator->setCMR (false);
          normal_estimator->setDoVoxelGrid (false);
          normal_estimator->setRemoveOutliers (false);
          normal_estimator->setValuesForCMRFalse (0.003f, 0.02f);
          normal_estimator->setForceUnorganized(true);
          normal_estimator->estimate (input_, processed, all_scene_normals);
      }
      else
      {
          PCL_WARN("Using those given from user code\n");
          processed = input_;
          all_scene_normals = scene_normals_;
      }

      {
        pcl::ScopeTime t("finding correct indices...\n");
        std::vector<int> correct_indices;
        getIndicesFromCloud<PointInT>(processed, keypoints_cloud, correct_indices);
        pcl::copyPointCloud(*all_scene_normals, correct_indices, *scene_normals);
      }
    }

    PCL_ERROR("set_save_hypotheses, doing correspondence grouping at MULTIpipeline level\n");
    typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it_map;
    for (it_map = object_hypotheses.begin (); it_map != object_hypotheses.end (); it_map++)
    {
      if(it_map->second.correspondences_to_inputcloud->size() < 3)
        continue;

      std::string id = it_map->second.model_->id_;
      //std::cout << id << " " << it_map->second.correspondences_to_inputcloud->size() << std::endl;
      std::vector < pcl::Correspondences > corresp_clusters;
      cg_algorithm_->setSceneCloud (keypoints_cloud);
      cg_algorithm_->setInputCloud ((*it_map).second.correspondences_pointcloud);

      if(cg_algorithm_->getRequiresNormals())
      {
        //std::cout << "CG alg requires normals..." << ((*it_map).second.normals_pointcloud)->points.size() << " " << (scene_normals)->points.size() << std::endl;
        cg_algorithm_->setInputAndSceneNormals((*it_map).second.normals_pointcloud, scene_normals);
      }
      //we need to pass the keypoints_pointcloud and the specific object hypothesis
      cg_algorithm_->setModelSceneCorrespondences ((*it_map).second.correspondences_to_inputcloud);
      cg_algorithm_->cluster (corresp_clusters);

      std::cout << "Instances:" << corresp_clusters.size () << " Total correspondences:" << (*it_map).second.correspondences_to_inputcloud->size () << " " << it_map->first << std::endl;

      for (size_t i = 0; i < corresp_clusters.size (); i++)
      {
        //std::cout << "size cluster:" << corresp_clusters[i].size() << std::endl;
        Eigen::Matrix4f best_trans;
        typename pcl::registration::TransformationEstimationSVD < PointInT, PointInT > t_est;
        t_est.estimateRigidTransformation (*(*it_map).second.correspondences_pointcloud, *keypoints_cloud, corresp_clusters[i], best_trans);

        models_->push_back ((*it_map).second.model_);
        transforms_->push_back (best_trans);
      }
    }
  }

  if (ICP_iterations_ > 0 || hv_algorithm_) {
    //Prepare scene and model clouds for the pose refinement step

    std::sort( input_icp_indices.begin(), input_icp_indices.end() );
    input_icp_indices.erase( std::unique( input_icp_indices.begin(), input_icp_indices.end() ), input_icp_indices.end() );

    pcl::PointIndices ind;
    ind.indices = input_icp_indices;
    icp_scene_indices_.reset(new pcl::PointIndices(ind));
    getDataSource()->voxelizeAllModels (VOXEL_SIZE_ICP_);
  }

  if (ICP_iterations_ > 0)
  {
    poseRefinement();
  }

  if (hv_algorithm_ && (models_->size () > 0))
  {
    std::cout << "Do hypothesis verification..." << models_->size () << std::endl;
    hypothesisVerification();
  }
}

template<typename PointInT>
bool
faat_pcl::rec_3d_framework::MultiRecognitionPipeline<PointInT>::isSegmentationRequired()
{
  bool ret_value = false;
  for(size_t i=0; (i < recognizers_.size()) && !ret_value; i++)
  {
    ret_value = recognizers_[i]->requiresSegmentation();
  }

  return ret_value;
}

template<typename PointInT>
typename boost::shared_ptr<faat_pcl::rec_3d_framework::Source<PointInT> >
faat_pcl::rec_3d_framework::MultiRecognitionPipeline<PointInT>::getDataSource ()
{
  //NOTE: Assuming source is the same or contains the same models for all recognizers...
  //Otherwise, we should create a combined data source so that all models are present

  return recognizers_[0]->getDataSource();
}

#endif /* MULTI_PIPELINE_RECOGNIZER_H_ */
