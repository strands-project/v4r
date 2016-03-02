#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/features/uniform_sampling.h>

namespace v4r
{

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::addModels (std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models, std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> &model_normals)
{
    size_t existing_models = recognition_models_.size();
    recognition_models_.resize( existing_models + models.size() );

    for(size_t i=0; i<models.size(); i++)
    {
        recognition_models_[existing_models + i].reset(new HVRecognitionModel<ModelT>);
        HVRecognitionModel<ModelT> &rm = *recognition_models_[existing_models + i];

        rm.is_planar_ = false;
        rm.complete_cloud_.reset(new pcl::PointCloud<ModelT>(*models[i]));
        rm.visible_cloud_.reset( new pcl::PointCloud<ModelT> );

        if (!param_.do_occlusion_reasoning_)   // just copy complete models
            *rm.visible_cloud_ = *models[i];
        else
        { //we need to reason about occlusions before setting the model
            ZBuffering<ModelT, SceneT> zbuffer_scene (param_.zbuffer_scene_resolution_, param_.zbuffer_scene_resolution_, 1.f);
            if (!occlusion_cloud_->isOrganized ())
            {
                PCL_WARN("Scene not organized... filtering using computed depth buffer\n");
                zbuffer_scene.computeDepthMap (*occlusion_cloud_, true);
            }

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
                rm.visible_indices_.resize(filter_self_occ_and_scene->points.size());

                for(size_t k=0; k < indices_cloud_occlusion.size(); k++)
                    rm.visible_indices_[k] = self_occlusion_indices[indices_cloud_occlusion[k]];

                if(normals_set_ && requires_normals_) {
                    rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal> ());
                    pcl::copyPointCloud(*rm.complete_cloud_normals_, rm.visible_indices_, *rm.visible_cloud_normals_);
                }
            }
            else
                zbuffer_scene.filter (*filter_self_occ, *filter_self_occ_and_scene, param_.occlusion_thres_);

            rm.visible_cloud_ = filter_self_occ_and_scene;
        }


        // copy normals if provided
        if ( i<model_normals.size() )
        {
            rm.complete_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal> (*model_normals[i]) );

            if (!param_.do_occlusion_reasoning_)   // just copy complete models
                rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal> (*rm.complete_cloud_normals_) );
            else
            {
                rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal>);
                pcl::copyPointCloud(*rm.complete_cloud_normals_, rm.visible_indices_, *rm.visible_cloud_normals_);
            }
        }
    }
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::setSceneCloud (const typename pcl::PointCloud<SceneT>::Ptr & scene_cloud)
{
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
