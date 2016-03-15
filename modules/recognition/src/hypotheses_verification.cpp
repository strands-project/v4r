#include <v4r/common/miscellaneous.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/common/zbuffering.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/features/uniform_sampling.h>
#include <pcl/common/time.h>
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/registration/icp.h>
#include <fstream>
#include <omp.h>

namespace v4r
{

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::computeVisibleModelsAndRefinePose()
{
    if(param_.icp_iterations_) {
        refined_model_transforms_.clear();
        refined_model_transforms_.resize( recognition_models_.size() );
    }

    typename ZBuffering<SceneT>::Parameter zbuffParam;
    zbuffParam.f_ = param_.focal_length_;
    zbuffParam.width_ = 640;
    zbuffParam.height_ = 480;

    ZBuffering<SceneT> zbuf(zbuffParam);
    Eigen::MatrixXf depth_image_scene;
    std::vector<int> visible_scene_indices;
    zbuf.computeDepthMap(*scene_cloud_, depth_image_scene, visible_scene_indices);

    #pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i<recognition_models_.size(); i++)
    {
        HVRecognitionModel<ModelT> &rm = *recognition_models_[i];
        rm.visible_cloud_.reset( new pcl::PointCloud<ModelT> );

        bool redo;
        do
        {
            redo = false;
            if (!param_.do_occlusion_reasoning_) // just copy complete models
            {
                *rm.visible_cloud_ = *rm.complete_cloud_;
                rm.visible_indices_.resize( rm.complete_cloud_->points.size());
                for(size_t pt=0; pt<rm.visible_indices_.size(); pt++)
                    rm.visible_indices_[pt] = pt;
            }
            else //occlusion reasoning based on self-occlusion and occlusion from scene cloud(s)
            {
                Eigen::MatrixXf depth_image_model;
                std::vector<int> visible_model_indices;
                ZBuffering<ModelT> zbufM(zbuffParam);
                zbufM.computeDepthMap(*rm.complete_cloud_, depth_image_model, visible_model_indices);
                std::vector<int> indices_map = zbufM.getIndicesMap();

                // now compare visible cloud with scene occlusion cloud
                for (size_t u=0; u<param_.img_width_; u++)
                {
                    for (size_t v=0; v<param_.img_height_; v++)
                    {
                        if ( depth_image_scene(v,u) + param_.occlusion_thres_ < depth_image_model(v,u) )
                            indices_map[v*param_.img_width_ + u] = -1;
                    }
                }

                rm.image_mask_.resize( indices_map.size(), false );
                rm.visible_indices_.resize( indices_map.size() );
                size_t kept=0;
                for(size_t pt=0; pt< indices_map.size(); pt++)
                {
                    if(indices_map[pt] >= 0)
                    {
                        rm.image_mask_[pt] = true;
                        rm.visible_indices_[kept] = indices_map[pt];
                        kept++;
                    }
                }
                rm.visible_indices_.resize(kept);
                pcl::copyPointCloud (*rm.complete_cloud_, rm.visible_indices_, *rm.visible_cloud_);

                if(normals_set_ && requires_normals_) {
                    rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal> ());
                    pcl::copyPointCloud(*rm.complete_cloud_normals_, rm.visible_indices_, *rm.visible_cloud_normals_);
                }
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


        if (param_.icp_iterations_ && refined_model_transforms_[i]) {
            pcl::PointCloud<pcl::Normal> aligned_normals;
            v4r::transformNormals(*rm.complete_cloud_normals_, aligned_normals, *refined_model_transforms_[i]);
            *rm.complete_cloud_normals_ = aligned_normals;
        }

        rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal>);
        pcl::copyPointCloud(*rm.complete_cloud_normals_, rm.visible_indices_, *rm.visible_cloud_normals_);

        rm.processSilhouette(param_.do_smoothing_, param_.smoothing_radius_, param_.do_erosion_, param_.erosion_radius_, param_.img_width_);
    }
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::addModels (std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models,
                                                   std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> &model_normals)
{
    size_t existing_models = recognition_models_.size();
    recognition_models_.resize( existing_models + models.size() );

    #pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i<models.size(); i++)
    {
        recognition_models_[existing_models + i].reset(new HVRecognitionModel<ModelT>);
        HVRecognitionModel<ModelT> &rm = *recognition_models_[existing_models + i];
        rm.complete_cloud_.reset(new pcl::PointCloud<ModelT>(*models[i]));
        rm.complete_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal> (*model_normals[i]) );
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
}

template class V4R_EXPORTS HypothesisVerification<pcl::PointXYZ,pcl::PointXYZ>;
template class V4R_EXPORTS HypothesisVerification<pcl::PointXYZRGB,pcl::PointXYZRGB>;
}
