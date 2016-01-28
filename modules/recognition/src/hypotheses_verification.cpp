#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/features/uniform_sampling.h>

namespace v4r
{

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::addModels (std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models, bool occlusion_reasoning)
{
  mask_.clear();
  complete_models_ = models;

  if (!occlusion_reasoning) {   // just copy complete models
      visible_models_.resize(models.size());
      for(size_t i=0; i<models.size(); i++) {
          if(!visible_models_[i])
              visible_models_[i].reset(new pcl::PointCloud<ModelT>);
          pcl::copyPointCloud(*models[i], *visible_models_[i]);
      }
  }
  else
  {
    //we need to reason about occlusions before setting the model
    if (!scene_cloud_)
      throw std::runtime_error("setSceneCloud should be called before adding the model if reasoning about occlusions...");

    if(!occlusion_cloud_set_) {
        PCL_WARN("Occlusion cloud not set, using scene_cloud instead...\n");
        occlusion_cloud_ = scene_cloud_;
    }


    ZBuffering<ModelT, SceneT> zbuffer_scene (param_.zbuffer_scene_resolution_, param_.zbuffer_scene_resolution_, 1.f);
    if (!occlusion_cloud_->isOrganized ())
    {
        PCL_WARN("Scene not organized... filtering using computed depth buffer\n");
        zbuffer_scene.computeDepthMap (*occlusion_cloud_, true);
    }

    visible_indices_.resize(models.size());
    visible_models_.resize(models.size());

    #pragma omp parallel for schedule (dynamic)
    for (size_t i = 0; i < models.size (); i++)
    {
      //self-occlusions
      typename pcl::PointCloud<ModelT>::Ptr filter_self_occ (new pcl::PointCloud<ModelT> ());
      typename pcl::PointCloud<ModelT>::Ptr filter_self_occ_and_scene (new pcl::PointCloud<ModelT> ());
      ZBuffering<ModelT, SceneT> zbuffer_self_occlusion (param_.zbuffer_self_occlusion_resolution_, param_.zbuffer_self_occlusion_resolution_, 1.f);
      zbuffer_self_occlusion.computeDepthMap (*models[i], true);

      std::vector<int> self_occlusion_indices;
      zbuffer_self_occlusion.filter (*models[i], self_occlusion_indices, param_.occlusion_thres_);
      pcl::copyPointCloud (*models[i], self_occlusion_indices, *filter_self_occ);

      //scene-occlusions
      std::vector<int> indices_cloud_occlusion;
      if (occlusion_cloud_->isOrganized ())
      {
        filter_self_occ_and_scene = filter<ModelT,SceneT> (*occlusion_cloud_, *filter_self_occ, param_.focal_length_, param_.occlusion_thres_, indices_cloud_occlusion);
        visible_indices_[i].resize(filter_self_occ_and_scene->points.size());

        for(size_t k=0; k < indices_cloud_occlusion.size(); k++)
            visible_indices_[i][k] = self_occlusion_indices[indices_cloud_occlusion[k]];

        if(normals_set_ && requires_normals_) {
          pcl::PointCloud<pcl::Normal>::Ptr filtered_normals (new pcl::PointCloud<pcl::Normal> ());
          pcl::copyPointCloud(*complete_normal_models_[i], visible_indices_[i], *filtered_normals);
          visible_normal_models_.push_back(filtered_normals);
        }
      }
      else
        zbuffer_scene.filter (*filter_self_occ, *filter_self_occ_and_scene, param_.occlusion_thres_);

      visible_models_[i] = filter_self_occ_and_scene;
    }
  }

  occlusion_cloud_set_ = false;
  normals_set_ = false;
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::setSceneCloud (const typename pcl::PointCloud<SceneT>::Ptr & scene_cloud)
{
  complete_models_.clear();
  visible_models_.clear();
  visible_normal_models_.clear();

  scene_cloud_ = scene_cloud;
  scene_cloud_downsampled_.reset(new pcl::PointCloud<SceneT>());

  if(param_.resolution_ <= 0.f)
      scene_cloud_downsampled_.reset(new pcl::PointCloud<SceneT>(*scene_cloud));
  else
  {
    /*pcl::VoxelGrid<SceneT> voxel_grid;
    voxel_grid.setInputCloud (scene_cloud);
    voxel_grid.setLeafSize (resolution_, resolution_, resolution_);
    voxel_grid.setDownsampleAllData(true);
    voxel_grid.filter (*scene_cloud_downsampled_);*/

    pcl::UniformSampling<SceneT> us;
    us.setRadiusSearch(param_.resolution_);
    us.setInputCloud(scene_cloud_);
    pcl::PointCloud<int> sampled_indices;
    us.compute(sampled_indices);
    scene_sampled_indices_.clear();
    scene_sampled_indices_.resize(sampled_indices.points.size());
    for(size_t i=0; i < scene_sampled_indices_.size(); i++)
        scene_sampled_indices_[i] = sampled_indices.points[i];

    pcl::copyPointCloud(*scene_cloud_, scene_sampled_indices_, *scene_cloud_downsampled_);
  }

  //initialize kdtree for search
  scene_downsampled_tree_.reset (new pcl::search::KdTree<SceneT>);
  scene_downsampled_tree_->setInputCloud(scene_cloud_downsampled_);
}

template class V4R_EXPORTS HypothesisVerification<pcl::PointXYZ,pcl::PointXYZ>;
template class V4R_EXPORTS HypothesisVerification<pcl::PointXYZRGB,pcl::PointXYZRGB>;
}
