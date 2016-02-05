#include <v4r/common/miscellaneous.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/features/uniform_sampling.h>
#include <pcl/common/time.h>
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/registration/icp.h>
#include <omp.h>

namespace v4r
{

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::addModels (std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models, std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> &model_normals)
{
    pcl::ScopeTime t("pose refinement and computing visible model points");

    size_t existing_models = recognition_models_.size();
    recognition_models_.resize( existing_models + models.size() );

    if(param_.icp_iterations_) {
        refined_model_transforms_.clear();
        refined_model_transforms_.resize( models.size() );
    }

    #pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i<models.size(); i++)
    {
        recognition_models_[existing_models + i].reset(new HVRecognitionModel<ModelT>);
        HVRecognitionModel<ModelT> &rm = *recognition_models_[existing_models + i];

        rm.is_planar_ = false;
        rm.complete_cloud_.reset(new pcl::PointCloud<ModelT>(*models[i]));
        rm.visible_cloud_.reset( new pcl::PointCloud<ModelT> );

        bool redo;

        do
        {
            redo = false;
            if (!param_.do_occlusion_reasoning_)   // just copy complete models
                *rm.visible_cloud_ = *models[i];
            else
            { //we need to reason about occlusions before setting the model

                typename pcl::PointCloud<ModelT>::Ptr filter_self_occ (new pcl::PointCloud<ModelT> ());
                typename pcl::PointCloud<ModelT>::Ptr filter_self_occ_and_scene (new pcl::PointCloud<ModelT> ());
                typename ZBuffering<ModelT>::Parameter zbuffParam;
                zbuffParam.inlier_threshold_ = param_.zbuffer_self_occlusion_resolution_;
                zbuffParam.f_ = param_.focal_length_;
                zbuffParam.width_ = 640;
                zbuffParam.height_ = 480;
                zbuffParam.u_margin_ = 5;
                zbuffParam.v_margin_ = 5;
                zbuffParam.compute_focal_length_ = true;
                ZBuffering<ModelT> zbuffer_scene (zbuffParam);
                if (!occlusion_cloud_->isOrganized ())
                {
                    PCL_WARN("Scene not organized... filtering using computed depth buffer\n");
                    zbuffer_scene.computeDepthMap (*occlusion_cloud_);
                }

                zbuffParam.compute_focal_length_ = false;
                ZBuffering<ModelT> zbuffer_self_occ (zbuffParam);
                zbuffer_self_occ.computeDepthMap (*models[i]);

                std::vector<int> self_occlusion_indices;
                zbuffer_self_occ.filter (*models[i], self_occlusion_indices);
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
                    zbuffer_scene.filter (*filter_self_occ, *filter_self_occ_and_scene);

                rm.visible_cloud_ = filter_self_occ_and_scene;
            }

            if(param_.icp_iterations_ && !refined_model_transforms_[i])
            {
                refined_model_transforms_[i].reset(new Eigen::Matrix4f (poseRefinement(rm)));

                pcl::PointCloud<ModelT> aligned_cloud;
                pcl::transformPointCloud(*rm.visible_cloud_, aligned_cloud, *refined_model_transforms_[i]);
                *rm.visible_cloud_  = aligned_cloud;
                pcl::transformPointCloud(*rm.complete_cloud_, aligned_cloud, *refined_model_transforms_[i]);
                *rm.complete_cloud_ = aligned_cloud;
                redo = true;
            }
        }
        while(redo);


        // copy normals if provided
        if ( i<model_normals.size() )
        {
            rm.complete_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal> (*model_normals[i]) );

            if (refined_model_transforms_[i])
                v4r::transformNormals(*model_normals[i], *rm.complete_cloud_normals_, *refined_model_transforms_[i]);

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
Eigen::Matrix4f
HypothesisVerification<ModelT, SceneT>::poseRefinement(const HVRecognitionModel<ModelT> &rm) const
{
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    ModelT minPoint, maxPoint;
    pcl::getMinMax3D(*rm.complete_cloud_, minPoint, maxPoint);
    float margin = 0.05;
    minPoint.x -= margin;
    minPoint.y -= margin;
    minPoint.z -= margin;

    maxPoint.x += margin;
    maxPoint.y += margin;
    maxPoint.z += margin;

    typename pcl::PointCloud<SceneT>::Ptr scene_cloud_downsampled_cropped (new pcl::PointCloud<SceneT>);
    pcl::CropBox<SceneT> cropFilter;
    cropFilter.setInputCloud (scene_cloud_downsampled_);
    cropFilter.setMin(minPoint.getVector4fMap());
    cropFilter.setMax(maxPoint.getVector4fMap());
    cropFilter.filter (*scene_cloud_downsampled_cropped);

    pcl::IterativeClosestPoint<ModelT, SceneT> icp;
    icp.setInputSource(rm.visible_cloud_);
    icp.setInputTarget(scene_cloud_downsampled_cropped);
    icp.setMaximumIterations(param_.icp_iterations_);
    pcl::PointCloud<ModelT> aligned_visible_model;
    icp.align(aligned_visible_model);
//            std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
    if(icp.hasConverged())
        transform = icp.getFinalTransformation();

    return transform;
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
