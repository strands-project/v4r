#include <v4r/common/miscellaneous.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/common/zbuffering.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <pcl/keypoints/uniform_sampling.h>
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
HypothesisVerification<ModelT, SceneT>::computeModelOcclusionByScene(HVRecognitionModel<ModelT> &rm, const std::vector<Eigen::MatrixXf> &depth_image_scene)
{
    typename ZBuffering<SceneT>::Parameter zbuffParam;
    zbuffParam.f_ = param_.focal_length_;
    zbuffParam.width_ = param_.img_width_;
    zbuffParam.height_ = param_.img_height_;

    std::vector<bool> image_mask_mv(rm.complete_cloud_->points.size(), false);
    for(size_t view=0; view<depth_image_scene.size(); view++)
    {
        // project into respective view
        pcl::PointCloud<ModelT> aligned_cloud;
        const Eigen::Matrix4f tf = absolute_camera_poses_[view].inverse();
        pcl::transformPointCloud(*rm.complete_cloud_, aligned_cloud, tf);

        Eigen::MatrixXf depth_image_model;
        std::vector<int> visible_model_indices;
        ZBuffering<ModelT> zbufM(zbuffParam);
        zbufM.computeDepthMap(aligned_cloud, depth_image_model, visible_model_indices);

        //                    std::ofstream f("/tmp/model_depth.txt");
        //                    f << depth_image_model;
        //                    f.close();

        boost::shared_ptr<std::vector<int> > indices_map = zbufM.getIndicesMap();

        // now compare visible cloud with scene occlusion cloud
        for (size_t u=0; u<param_.img_width_; u++)
        {
            for (size_t v=0; v<param_.img_height_; v++)
            {
                if ( depth_image_scene[view](v,u) + param_.occlusion_thres_ < depth_image_model(v,u) )
                    indices_map->at(v*param_.img_width_ + u) = -1;
            }
        }

        for(size_t pt=0; pt<indices_map->size(); pt++)
        {
            if(indices_map->at(pt) >= 0)
            {
                rm.image_mask_[view][pt] = true;
                image_mask_mv[ indices_map->at(pt) ] = true;
            }
        }
    }

    rm.visible_indices_ = createIndicesFromMask<int>(image_mask_mv);
    pcl::copyPointCloud (*rm.complete_cloud_, rm.visible_indices_, *rm.visible_cloud_);

}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::computeVisibleModelsAndRefinePose()
{
    refined_model_transforms_.clear();
    refined_model_transforms_.resize( recognition_models_.size(), Eigen::Matrix4f::Identity() );

    typename ZBuffering<SceneT>::Parameter zbuffParam;
    zbuffParam.f_ = param_.focal_length_;
    zbuffParam.width_ = param_.img_width_;
    zbuffParam.height_ = param_.img_height_;

    ZBuffering<SceneT> zbuf(zbuffParam);
    std::vector<Eigen::MatrixXf> depth_image_scene (occlusion_clouds_.size());

#pragma omp parallel for schedule(dynamic)
    for(size_t view=0; view<occlusion_clouds_.size(); view++)
    {
        std::vector<int> visible_scene_indices;
        zbuf.computeDepthMap(*occlusion_clouds_[view], depth_image_scene[view], visible_scene_indices);
    }

#pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i<recognition_models_.size(); i++)
    {
        HVRecognitionModel<ModelT> &rm = *recognition_models_[i];
        rm.visible_cloud_.reset( new pcl::PointCloud<ModelT> );
        rm.image_mask_.resize(occlusion_clouds_.size(), std::vector<bool> (param_.img_width_ * param_.img_height_, false) );

        if (!param_.do_occlusion_reasoning_) // just copy complete models
        {
            *rm.visible_cloud_ = *rm.complete_cloud_;
            rm.visible_indices_.resize( rm.complete_cloud_->points.size());
            for(size_t pt=0; pt<rm.visible_indices_.size(); pt++)
                rm.visible_indices_[pt] = pt;

            if(param_.icp_iterations_)
            {
                refined_model_transforms_[i] = poseRefinement(rm);
                pcl::transformPointCloud(*rm.complete_cloud_, *rm.complete_cloud_, refined_model_transforms_[i]);
                pcl::copyPointCloud (*rm.complete_cloud_, rm.visible_indices_, *rm.visible_cloud_);
            }
        }
        else //occlusion reasoning based on self-occlusion and occlusion from scene cloud(s)
        {
            computeModelOcclusionByScene(rm, depth_image_scene);    // just do ICP on visible cloud (given by initial estimate)

            if(param_.icp_iterations_ )
            {
                refined_model_transforms_[i] = poseRefinement(rm) * refined_model_transforms_[i];
                pcl::transformPointCloud(*rm.complete_cloud_, *rm.complete_cloud_, refined_model_transforms_[i]);
                pcl::copyPointCloud (*rm.complete_cloud_, rm.visible_indices_, *rm.visible_cloud_);
            }

            computeModelOcclusionByScene(rm, depth_image_scene);    // compute updated visible cloud
        }

        if (param_.icp_iterations_)
        {
            v4r::transformNormals(*rm.complete_cloud_normals_, *rm.complete_cloud_normals_, refined_model_transforms_[i]);
        }

        rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal>);
        pcl::copyPointCloud(*rm.complete_cloud_normals_, rm.visible_indices_, *rm.visible_cloud_normals_);
        rm.processSilhouette(param_.do_smoothing_, param_.smoothing_radius_, param_.do_erosion_, param_.erosion_radius_, param_.img_width_);
    }
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::addModels (const std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models,
                                                   const std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> &model_normals)
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
HypothesisVerification<ModelT, SceneT>::poseRefinement(HVRecognitionModel<ModelT> &rm) const
{
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
    pcl::CropBox<SceneT> cropFilter(true);
    cropFilter.setInputCloud (scene_cloud_downsampled_);
    cropFilter.setMin(minPoint.getVector4fMap());
    cropFilter.setMax(maxPoint.getVector4fMap());
    cropFilter.filter (*scene_cloud_downsampled_cropped);
//    boost::shared_ptr <const std::vector<int> > indices = cropFilter.getRemovedIndices();
//    std::vector<bool> mask_inv = createMaskFromIndices(*indices, scene_cloud_downsampled_->points.size());
//    rm.scene_indices_in_crop_box_ = createIndicesFromMask<int>(mask_inv, true);

    pcl::IterativeClosestPoint<ModelT, SceneT> icp;
    icp.setInputSource(rm.visible_cloud_);
    icp.setInputTarget(scene_cloud_downsampled_cropped);
    icp.setMaximumIterations(param_.icp_iterations_);
    pcl::PointCloud<ModelT> aligned_visible_model;
    icp.align(aligned_visible_model);
    if(icp.hasConverged())
        return icp.getFinalTransformation();

    return Eigen::Matrix4f::Identity();
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::downsampleSceneCloud()
{

    if(param_.resolution_mm_ <= 0)
    {
        scene_cloud_downsampled_.reset(new pcl::PointCloud<SceneT>(*scene_cloud_));
        scene_normals_downsampled_.reset(new pcl::PointCloud<pcl::Normal>(*scene_normals_));
    }
    else
    {
        /*pcl::VoxelGrid<SceneT> voxel_grid;
    voxel_grid.setInputCloud (scene_cloud);
    voxel_grid.setLeafSize (resolution_, resolution_, resolution_);
    voxel_grid.setDownsampleAllData(true);
    voxel_grid.filter (*scene_cloud_downsampled_);*/

        scene_cloud_downsampled_.reset(new pcl::PointCloud<SceneT>());
        scene_normals_downsampled_.reset(new pcl::PointCloud<pcl::Normal>());

        pcl::UniformSampling<SceneT> us;
        double resolution = (float)param_.resolution_mm_ / 1000.f;
        us.setRadiusSearch( resolution );
        us.setInputCloud( scene_cloud_ );
        pcl::PointCloud<int> sampled_indices;
        us.compute(sampled_indices);
        scene_sampled_indices_.clear();
        scene_sampled_indices_.resize(sampled_indices.points.size());
        for(size_t i=0; i < scene_sampled_indices_.size(); i++)
            scene_sampled_indices_[i] = sampled_indices.points[i];

        pcl::copyPointCloud(*scene_cloud_, scene_sampled_indices_, *scene_cloud_downsampled_);
        pcl::copyPointCloud(*scene_normals_, scene_sampled_indices_, *scene_normals_downsampled_);
    }

    removeSceneNans();
}



template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::removeSceneNans()
{
    CHECK( scene_cloud_downsampled_->points.size () == scene_normals_downsampled_->points.size() &&
           scene_cloud_downsampled_->points.size () == scene_sampled_indices_.size() );

    size_t kept = 0;
    for (size_t i = 0; i < scene_cloud_downsampled_->points.size (); i++) {
        if ( pcl::isFinite( scene_cloud_downsampled_->points[i]) && pcl::isFinite( scene_normals_downsampled_->points[i] ))
        {
            scene_cloud_downsampled_->points[kept] = scene_cloud_downsampled_->points[i];
            scene_normals_downsampled_->points[kept] = scene_normals_downsampled_->points[i];
            scene_sampled_indices_[kept] = scene_sampled_indices_[i];
            kept++;
        }
    }
    scene_sampled_indices_.resize(kept);
    scene_cloud_downsampled_->points.resize(kept);
    scene_normals_downsampled_->points.resize (kept);
    scene_cloud_downsampled_->width = scene_normals_downsampled_->width = kept;
    scene_cloud_downsampled_->height = scene_normals_downsampled_->height = 1;
}

template class V4R_EXPORTS HypothesisVerification<pcl::PointXYZ,pcl::PointXYZ>;
template class V4R_EXPORTS HypothesisVerification<pcl::PointXYZRGB,pcl::PointXYZRGB>;
}
