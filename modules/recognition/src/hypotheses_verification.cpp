#include <v4r/common/normals.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/common/zbuffering.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/segmentation/ClusterNormalsToPlanesPCL.h>
#include <v4r/segmentation/smooth_Euclidean_segmenter.h>
#include <v4r/segmentation/multiplane_segmenter.h>

#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/common/angles.h>
#include <pcl/common/time.h>
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>

#include <fstream>
#include <iomanip>
#include <omp.h>
#include <functional>
#include <numeric>

#include <opencv2/opencv.hpp>

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
        for (int u=0; u<param_.img_width_; u++)
        {
            for (int v=0; v<param_.img_height_; v++)
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
    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
    {
        for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
        {
            HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];
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
                    rm.refined_pose_ = poseRefinement(rm);
                    pcl::transformPointCloud(*rm.complete_cloud_, *rm.complete_cloud_, rm.refined_pose_);
                    pcl::copyPointCloud (*rm.complete_cloud_, rm.visible_indices_, *rm.visible_cloud_);
                }
            }
            else //occlusion reasoning based on self-occlusion and occlusion from scene cloud(s)
            {
                computeModelOcclusionByScene(rm, depth_image_scene);    // just do ICP on visible cloud (given by initial estimate)

                if(param_.icp_iterations_ )
                {
                    Eigen::Matrix4f old_tf = rm.refined_pose_;
                    rm.refined_pose_ = poseRefinement(rm) * old_tf;
                    pcl::transformPointCloud(*rm.complete_cloud_, *rm.complete_cloud_, rm.refined_pose_);
                    pcl::copyPointCloud (*rm.complete_cloud_, rm.visible_indices_, *rm.visible_cloud_);
                }

                computeModelOcclusionByScene(rm, depth_image_scene);    // compute updated visible cloud
            }

            if (param_.icp_iterations_)
            {
                v4r::transformNormals(*rm.complete_cloud_normals_, *rm.complete_cloud_normals_, rm.refined_pose_);
            }

            rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal>);
            pcl::copyPointCloud(*rm.complete_cloud_normals_, rm.visible_indices_, *rm.visible_cloud_normals_);
            rm.processSilhouette(param_.do_smoothing_, param_.smoothing_radius_, param_.do_erosion_, param_.erosion_radius_, param_.img_width_);
        }
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



template<typename ModelT, typename SceneT>
mets::gol_type
HypothesisVerification<ModelT, SceneT>::evaluateSolution (const std::vector<bool> & active, int changed)
{
    int sign = 1;
    if ( !active[changed]) //it has been deactivated
    {
        sign = -1;
        tmp_solution_(changed) = 0;
    }
    else
        tmp_solution_(changed) = 1;

    float num_active_hypotheses = tmp_solution_.sum();
#pragma omp parallel for schedule(dynamic)
    for(int row_id=0; row_id < scene_explained_weight_compressed_.rows(); row_id++)
    {
        double max = std::numeric_limits<double>::min();
        for(size_t col_id=0; col_id<active.size(); col_id++)
        {
            if ( active[col_id] && scene_explained_weight_compressed_(row_id,col_id)>max)
                max = scene_explained_weight_compressed_(row_id,col_id);
        }
        max_scene_explained_weight_(row_id)=max;
    }

    if(num_active_hypotheses > 0.5f)    // since we do not use integer
    {
        model_fitness_ = model_fitness_v_.dot(tmp_solution_);
        scene_fitness_ = max_scene_explained_weight_.sum();
        pairwise_cost_ = 0.5 * tmp_solution_.transpose() * intersection_cost_ * tmp_solution_;
        cost_ = -(   param_.regularizer_ * scene_fitness_ + model_fitness_
                     - param_.clutter_regularizer_ * pairwise_cost_ );
    }
    else
    {
        cost_ = model_fitness_ = scene_fitness_ =  pairwise_cost_ = 0.f;
    }

    if(cost_logger_) {
        cost_logger_->increaseEvaluated();
        cost_logger_->addCostEachTimeEvaluated(cost_);
    }

    return static_cast<mets::gol_type> (cost_); //return the dual to our max problem
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::computePairwiseIntersection()
{
    intersection_cost_ = Eigen::MatrixXf::Zero(global_hypotheses_.size(), global_hypotheses_.size());

    for(size_t i=1; i<global_hypotheses_.size(); i++)
    {
        HVRecognitionModel<ModelT> &rm_a = *global_hypotheses_[i];
        for(size_t j=0; j<i; j++)
        {
            const HVRecognitionModel<ModelT> &rm_b = *global_hypotheses_[j];

            size_t num_intersections = 0, total_rendered_points = 0;

            for(size_t view=0; view<rm_a.image_mask_.size(); view++)
            {
                for(size_t px=0; px<rm_a.image_mask_[view].size(); px++)
                {
                    if( rm_a.image_mask_[view][px] && rm_b.image_mask_[view][px])
                        num_intersections++;

                    if ( rm_a.image_mask_[view][px] || rm_b.image_mask_[view][px] )
                        total_rendered_points++;
                }
            }

            float conflict_cost = static_cast<float> (num_intersections) / total_rendered_points;
            intersection_cost_(i,j) = intersection_cost_(j,i) = conflict_cost;
        }

        if(!param_.visualize_pairwise_cues_)
            rm_a.image_mask_.clear();
    }
}


template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::removeModelsWithLowVisibility()
{
    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
    {
        for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
        {
            typename HVRecognitionModel<ModelT>::Ptr &rm = obj_hypotheses_groups_[i][jj];

            if( (float)rm->visible_cloud_->points.size() / (float)rm->complete_cloud_->points.size() < param_.min_visible_ratio_)
            {
                rm->rejected_due_to_low_visibility_ = true;

                if(!param_.visualize_model_cues_)
                    rm->freeSpace();
            }
        }
    }
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::computePlanarSurfaces ()
{
    planes_.clear();

    if( occlusion_clouds_.size() == 1 && param_.plane_method_ == 0) {   // this method needs an organized point cloud (so only available in single-view case)
        MultiplaneSegmenter<SceneT> mps;
        mps.setInputCloud( scene_cloud_downsampled_ );
        mps.setNormalsCloud( scene_normals_downsampled_ );
        mps.computeTablePlanes();
        planes_ = mps.getAllPlanes();
    }
    else {
        typename ClusterNormalsToPlanesPCL<SceneT>::Parameter p_param;
        p_param.inlDistSmooth = 0.05f;
        p_param.minPoints = 5000;
        p_param.inlDist = 0.02f;
        p_param.thrAngle = 30.f;
        p_param.K_ = 10;
        p_param.normal_computation_method_ = 2;
        ClusterNormalsToPlanesPCL<SceneT> pest(p_param);

        std::vector<PlaneModel<SceneT> > planes_tmp;
        pest.compute(scene_cloud_downsampled_, *scene_normals_downsampled_, planes_tmp);
        planes_.resize( planes_tmp.size() );
        for(size_t p_id=0; p_id<planes_tmp.size(); p_id++)
        {
            planes_[p_id].reset ( new PlaneModel<SceneT>);
            *planes_[p_id] = planes_tmp[p_id];
        }
    }

    for(size_t p_id = 0; p_id<planes_.size(); p_id++)
    {
        PlaneModel<SceneT> &pm = *planes_[p_id];
        pm.visualize();
    }
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::computePlaneIntersection ()
{


    pcl::visualization::PCLVisualizer vis;
    int vp1,vp2,vp3;
    vis.createViewPort(0,0,0.33,1,vp1);
    vis.createViewPort(0.33,0,0.66,1,vp2);
    vis.createViewPort(0.66,0,1,1,vp3);

    for(size_t p_id = 0; p_id<planes_.size(); p_id++)
    {
        PlaneModel<SceneT> &pm = *planes_[p_id];
        typename pcl::PointCloud<SceneT>::Ptr convex_hull_const = pm.getConvexHullCloud();

        for (size_t i = 0; i < recognition_models_.size(); i++)
        {
            HVRecognitionModel<ModelT> &rm = *recognition_models_[i];

            vis.removeAllPointClouds();
            vis.addPointCloud(scene_cloud_downsampled_, "scene", vp1);
            vis.addPointCloud(rm.visible_cloud_, "model", vp2);
            vis.addPointCloud(pm.projectPlaneCloud(0.005f), "plane", vp2);

            typename pcl::PointCloud<ModelT>::Ptr intersect_cloud (new pcl::PointCloud<ModelT>);
            intersect_cloud->points.resize( rm.visible_cloud_->points.size() );

            for (size_t idx = 0; idx < rm.visible_cloud_->points.size (); idx++)
            {
                const ModelT &m = rm.visible_cloud_->points[idx];
                bool is_in = pcl::isPointIn2DPolygon(m, *convex_hull_const);

                ModelT &pt = intersect_cloud->points[idx];
                pt.getVector3fMap() = m.getVector3fMap();
                pt.r = pt.g = pt.b = 0.f;

                if(is_in)
                    pt.r = 255.f;
                else
                    pt.g = 255.f;
            }
            vis.addPointCloud(intersect_cloud, "intersect", vp3);
            vis.spin();
        }
    }
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::extractEuclideanClustersSmooth()
{
    typename SmoothEuclideanSegmenter<SceneT>::Parameter param;
    param.cluster_tolerance_ = param_.cluster_tolerance_;
    param.curvature_threshold_ = param_.curvature_threshold_;
    param.eps_angle_threshold_deg_ = param_.eps_angle_threshold_deg_;
    param.min_points_ = param_.min_points_;
    param.z_adaptive_ = param_.z_adaptive_;
//    param.compute_planar_patches_only_ = true;

    SmoothEuclideanSegmenter<SceneT> seg(param);
    seg.setInputCloud(scene_cloud_downsampled_);
    seg.setNormalsCloud(scene_normals_downsampled_);
    seg.setSearchMethod(octree_scene_downsampled_);
    seg.segment();
    std::vector<pcl::PointIndices > clusters;
    seg.getSegmentIndices( clusters );

    scene_smooth_labels_.clear();
    scene_smooth_labels_.resize(scene_cloud_downsampled_->points.size(), 0);
    smooth_label_count_.resize( clusters.size() + 1);

    size_t total_labeled_pts = 0;
    for (size_t i = 0; i < clusters.size (); i++)
    {
        smooth_label_count_[i+1] = clusters[i].indices.size();
        total_labeled_pts +=clusters[i].indices.size();
        for (size_t j = 0; j < clusters[i].indices.size (); j++)
        {
            int idx = clusters[i].indices[j];
            scene_smooth_labels_[ idx ] = i+1;
        }
    }
    smooth_label_count_[0] = scene_cloud_downsampled_->points.size() - total_labeled_pts;
}

template<typename ModelT, typename SceneT>
bool
HypothesisVerification<ModelT, SceneT>::individualRejection(HVRecognitionModel<ModelT> &rm)
{
    if (param_.check_smooth_clusters_)
    {
        for(size_t cluster_id=1; cluster_id<rm.explained_pts_per_smooth_cluster_.size(); cluster_id++)  // don't check label 0
        {
            if ( smooth_label_count_[cluster_id] > 100 &&  rm.explained_pts_per_smooth_cluster_[cluster_id] > 100 &&
                 (float)(rm.explained_pts_per_smooth_cluster_[cluster_id]) / smooth_label_count_[cluster_id] < param_.min_ratio_cluster_explained_ )
            {
                rm.rejected_due_to_smooth_cluster_check_ = true;
                break;
            }
        }
    }

    float visible_ratio = rm.visible_cloud_->points.size() / (float)rm.complete_cloud_->points.size();
    float model_fitness = rm.model_fit_ / rm.visible_cloud_->points.size();

    CHECK(param_.min_model_fitness_lower_bound_ <= param_.min_model_fitness_upper_bound_); // scale model fitness threshold with the visible ratio of model. Highly occluded objects need to have a very strong evidence

    float scale = std::min<float>( 1.f, visible_ratio/0.5f );
    float range = param_.min_model_fitness_upper_bound_ - param_.min_model_fitness_lower_bound_;
    float model_fitness_threshold = param_.min_model_fitness_upper_bound_ - scale * range;

    if( model_fitness < model_fitness_threshold)
        rm.rejected_due_to_low_model_fitness_ = true;

    return rm.isRejected();
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::initialize()
{
    global_hypotheses_.clear();
    this->downsampleSceneCloud();

    {
        pcl::ScopeTime t("Computing octree");
        octree_scene_downsampled_.reset(new pcl::octree::OctreePointCloudSearch<SceneT>( (double)param_.resolution_mm_ / 1000.f));
        octree_scene_downsampled_->setInputCloud(scene_cloud_downsampled_);
        octree_scene_downsampled_->addPointsFromInputCloud();
    }

    if(occlusion_clouds_.empty()) // we can treat single-view as multi-view case with just one view
    {
        occlusion_clouds_.push_back(scene_cloud_);
        absolute_camera_poses_.push_back( Eigen::Matrix4f::Identity() );
    }

//    computePlanarSurfaces();

#pragma omp parallel sections
    {
#pragma omp section
        {
            pcl::ScopeTime t("pose refinement and computing visible model points");
            computeVisibleModelsAndRefinePose();
        }

#pragma omp section
        {
            if(param_.check_smooth_clusters_)
                extractEuclideanClustersSmooth();
        }
    }

    removeModelsWithLowVisibility();
    ColorTransformOMP::initializeLUT();

#pragma omp parallel sections
    {
#pragma omp section
        if(!param_.ignore_color_even_if_exists_)
        {
            pcl::ScopeTime t("Converting scene color values");
            convertToLABcolor(*scene_cloud_downsampled_, scene_color_channels_);
        }

#pragma omp section
        {
            pcl::ScopeTime t("Converting model color values");
            for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
            {
                for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                {
                    HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];

                    if(!rm.isRejected())
                    {
                        removeNanNormals(rm);

                        if(!param_.ignore_color_even_if_exists_)
                            convertToLABcolor(*rm.visible_cloud_, rm.pt_color_);
                    }
                }
            }
        }


#pragma omp section
        {
            pcl::ScopeTime t("Computing model to scene fitness");
#pragma omp parallel for schedule(dynamic)
            for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
            {
                for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                {
                    HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];

                    if(!rm.isRejected())
                        computeModel2SceneDistances(rm);
                }
            }
        }
    }

    if(param_.check_plane_intersection_)
    {
        pcl::ScopeTime t("Computing plane intersection");
        //        computePlaneIntersection();
    }


    if(param_.use_histogram_specification_)
    {
        pcl::ScopeTime t("Computing histogramm specification");
        for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
        {
            for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
            {
                HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];

                if(!rm.isRejected())
                {
                    computeLoffset(rm);
                }
            }
        }
    }


    {
        pcl::ScopeTime t("Computing fitness score between models and scene");
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
        {
            for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
            {
                HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];

                if(!rm.isRejected())
                    computeModel2SceneFitness(rm);
            }
        }
    }

    // do non-maxima surpression on hypotheses groups w.r.t. to the model fitness
    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
    {
        std::vector<typename HVRecognitionModel<ModelT>::Ptr > ohg = obj_hypotheses_groups_[i];

        if (ohg.size() > 1)
        {
            std::sort(ohg.begin(), ohg.end(), HVRecognitionModel<ModelT>::modelFitCompare);

            for(size_t jj=0; jj<ohg.size()-1; jj++)
                ohg[jj]->rejected_due_to_better_hypothesis_in_group_ = true;
        }
    }

    size_t kept_hypotheses = 0;
    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
    {
        for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
        {
            HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];
            if( rm.isRejected() )
                continue;

            if ( !individualRejection(rm) )
                kept_hypotheses++;
        }
    }

    if( !kept_hypotheses )
        return;

    global_hypotheses_.resize( kept_hypotheses );
    scene_explained_weight_ = Eigen::MatrixXf(scene_cloud_downsampled_->points.size(), kept_hypotheses);

    kept_hypotheses = 0;
    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
    {
        for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
        {
            typename HVRecognitionModel<ModelT>::Ptr rm = obj_hypotheses_groups_[i][jj];

            if( rm->isRejected() )
                continue;

            if ( !individualRejection(*rm) )
            {
                scene_explained_weight_.col(kept_hypotheses) = rm->scene_explained_weight_;
                global_hypotheses_[kept_hypotheses] = rm;
                kept_hypotheses++;
            }
        }
    }


    {
        pcl::ScopeTime t("Compressing scene explained matrix");
        // remove rows of scene explained matrix, whose point is not explained by any hypothesis. Because it is usually very sparse and would take a lot of computation time.
        scene_explained_weight_compressed_ = scene_explained_weight_;
        Eigen::VectorXf min_tmp = scene_explained_weight_.rowwise().maxCoeff();
        size_t kept=0;
        for(size_t pt=0; pt<scene_cloud_downsampled_->points.size(); pt++)
        {
            if( min_tmp(pt) > std::numeric_limits<float>::epsilon() )
            {
                scene_explained_weight_compressed_.row(kept).swap(scene_explained_weight_compressed_.row(pt));
                kept++;
            }
        }
        scene_explained_weight_compressed_.conservativeResize(kept, scene_explained_weight_.cols());

        if(!kept)
            return;
    }


    {
        pcl::ScopeTime t("Computing pairwise intersection");
        computePairwiseIntersection();
    }

    if(param_.visualize_model_cues_)
    {
        for(size_t i=0; i<global_hypotheses_.size(); i++)
        {
            HVRecognitionModel<ModelT> &rm = *global_hypotheses_[i];
            visualizeGOCuesForModel(rm);
        }
    }

    if(param_.visualize_pairwise_cues_)
        visualizePairwiseIntersection();

    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
    {
        for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
        {
            typename HVRecognitionModel<ModelT>::Ptr rm = obj_hypotheses_groups_[i][jj];
            rm->freeSpace();
        }
    }
}

template<typename ModelT, typename SceneT>
std::vector<bool>
HypothesisVerification<ModelT, SceneT>::optimize ()
{
    std::vector<bool> temp_solution ( global_hypotheses_.size(), param_.initial_status_);
    if(param_.initial_status_)
        tmp_solution_ = Eigen::VectorXf::Ones ( global_hypotheses_.size());
    else
        tmp_solution_ = Eigen::VectorXf::Zero ( global_hypotheses_.size());

    //store model fitness into vector
    model_fitness_v_ = Eigen::VectorXf( global_hypotheses_.size() );
    for (size_t i = 0; i < global_hypotheses_.size(); i++)
        model_fitness_v_[i] = global_hypotheses_[i]->model_fit_;

    GHVSAModel<ModelT, SceneT> model;

    double initial_cost  = 0.f;
    max_scene_explained_weight_ = Eigen::VectorXf::Zero(scene_cloud_downsampled_->points.size());
    model.cost_ = static_cast<mets::gol_type> ( initial_cost );
    model.setSolution (temp_solution);
    model.setOptimizer (this);

    GHVSAModel<ModelT, SceneT> *best = new GHVSAModel<ModelT, SceneT> (model);
    GHVmove_manager<ModelT, SceneT> neigh ( global_hypotheses_.size (), param_.use_replace_moves_ );
    neigh.setExplainedPointIntersections(intersection_cost_);

    //mets::best_ever_solution best_recorder (best);
    cost_logger_.reset(new GHVCostFunctionLogger<ModelT, SceneT>(*best));
    mets::noimprove_termination_criteria noimprove (param_.max_iterations_);

    if(param_.visualize_go_cues_)
        cost_logger_->setVisualizeFunction(visualize_cues_during_logger_);

    switch( param_.opt_type_ )
    {
    case OptimizationType::LocalSearch:
    {
        mets::local_search<GHVmove_manager<ModelT, SceneT> > local ( model, *(cost_logger_.get()), neigh, 0, false);
        {
            pcl::ScopeTime t ("local search...");
            local.search ();
        }
        break;
    }
    case OptimizationType::TabuSearch:
    {
        //Tabu search
        //mets::simple_tabu_list tabu_list ( initial_solution.size() * sqrt ( 1.0*initial_solution.size() ) ) ;
        mets::simple_tabu_list tabu_list ( 5 * temp_solution.size()) ;
        mets::best_ever_criteria aspiration_criteria ;

        std::cout << "max iterations:" << param_.max_iterations_ << std::endl;
        mets::tabu_search<GHVmove_manager<ModelT, SceneT> > tabu_search(model,  *(cost_logger_.get()), neigh, tabu_list, aspiration_criteria, noimprove);
        //mets::tabu_search<move_manager> tabu_search(model, best_recorder, neigh, tabu_list, aspiration_criteria, noimprove);

        {
            pcl::ScopeTime t ("TABU search...");
            try {
                tabu_search.search ();
            } catch (mets::no_moves_error e) {
                //} catch (std::exception e) {

            }
        }
        break;
    }
    case OptimizationType::TabuSearchWithLSRM:
    {
        GHVmove_manager<ModelT, SceneT> neigh4 ( global_hypotheses_.size(), false);
        neigh4.setExplainedPointIntersections(intersection_cost_);

        mets::simple_tabu_list tabu_list ( temp_solution.size() * sqrt ( 1.0*temp_solution.size() ) ) ;
        mets::best_ever_criteria aspiration_criteria ;
        mets::tabu_search<GHVmove_manager<ModelT, SceneT> > tabu_search(model,  *(cost_logger_.get()), neigh4, tabu_list, aspiration_criteria, noimprove);
        //mets::tabu_search<move_manager> tabu_search(model, best_recorder, neigh, tabu_list, aspiration_criteria, noimprove);

        {
            pcl::ScopeTime t("TABU search + LS (RM)...");
            try { tabu_search.search (); }
            catch (mets::no_moves_error e) { }

            std::cout << "Tabu search finished... starting LS with RM" << std::endl;

            //after TS, we do LS with RM
            GHVmove_manager<ModelT, SceneT> neigh4RM ( global_hypotheses_.size(), true);
            neigh4RM.setExplainedPointIntersections(intersection_cost_);

            mets::local_search<GHVmove_manager<ModelT, SceneT> > local ( model, *(cost_logger_.get()), neigh4RM, 0, false);
            {
                pcl::ScopeTime t_local_search ("local search...");
                local.search ();
                (void)t_local_search;
            }
        }
        break;

    }
    case OptimizationType::SimulatedAnnealing:
    {
        //Simulated Annealing
        //mets::linear_cooling linear_cooling;
        mets::exponential_cooling linear_cooling;
        mets::simulated_annealing<GHVmove_manager<ModelT, SceneT> > sa (model,  *(cost_logger_.get()), neigh, noimprove, linear_cooling, initial_temp_, 1e-7, 1);
        sa.setApplyAndEvaluate (true);

        {
            pcl::ScopeTime t ("SA search...");
            sa.search ();
        }
        break;
    }
    default:
        throw std::runtime_error("Specified optimization type not implememted!");
    }

    best_seen_ = static_cast<const GHVSAModel<ModelT, SceneT>&> (cost_logger_->best_seen());
    std::cout << "*****************************" << std::endl
              << "Final cost:" << best_seen_.cost_ << std::endl
              << "Number of ef evaluations:" << cost_logger_->getTimesEvaluated() << std::endl
              << "Number of accepted moves:" << cost_logger_->getAcceptedMovesSize() << std::endl
              << "*****************************" << std::endl;

    delete best;
    return best_seen_.solution_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::verify()
{
    {
        pcl::ScopeTime t("initialization");
        initialize();
    }

    if(param_.visualize_go_cues_)
        visualize_cues_during_logger_ = boost::bind(&HypothesisVerification<ModelT, SceneT>::visualizeGOCues, this, _1, _2, _3);

    {
        pcl::ScopeTime t("Optimizing object hypotheses verification cost function");
        solution_ = optimize ();
    }

    cleanUp();
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::specifyHistogram (const Eigen::MatrixXf & src, const Eigen::MatrixXf & dst, Eigen::MatrixXf & lookup) const
{
    if( src.cols() != dst.cols() || src.rows() != dst.rows() )
        throw std::runtime_error ("The given matrices to speficyHistogram must have the same size!");

    //normalize histograms
    size_t dims = src.cols();
    size_t bins = src.rows();

    Eigen::MatrixXf src_normalized(bins, dims), dst_normalized (bins, dims);

    for(size_t i=0; i < dims; i++) {
        src_normalized.col(i) = src.col(i) / src.col(i).sum();
        dst_normalized.col(i) = dst.col(i) / dst.col(i).sum();
    }

    Eigen::MatrixXf src_cumulative = Eigen::MatrixXf::Zero(bins, dims);
    Eigen::MatrixXf dst_cumulative = Eigen::MatrixXf::Zero(bins, dims);
    lookup = Eigen::MatrixXf::Zero(bins, dims);

    for (size_t dim = 0; dim < dims; dim++)
    {
        src_cumulative (0, dim) = src_normalized (0, dim);
        dst_cumulative (0, dim) = dst_normalized (0, dim);
        for (size_t bin = 1; bin < bins; bin++)
        {
            src_cumulative (bin, dim) = src_cumulative (bin - 1, dim) + src_normalized (bin, dim);
            dst_cumulative (bin, dim) = dst_cumulative (bin - 1, dim) + dst_normalized (bin, dim);
        }

        int last = 0;
        for (size_t bin = 0; bin < bins; bin++)
        {
            for (size_t z = last; z < bins; z++)
            {
                if (src_cumulative (z, dim) - dst_cumulative (bin, dim) >= 0)
                {
                    if (z > 0 && (dst_cumulative (bin, dim) - src_cumulative (z - 1, dim)) < (src_cumulative (z, dim) - dst_cumulative (bin, dim)))
                        z--;

                    lookup(bin, dim) = z;
                    last = z;
                    break;
                }
            }
        }

        int min = 0;
        for (size_t k = 0; k < bins; k++)
        {
            if (lookup (k, dim) != 0)
            {
                min = lookup (k, dim);
                break;
            }
        }

        for (size_t k = 0; k < bins; k++)
        {
            if (lookup (k, dim) == 0)
                lookup (k, dim) = min;
            else
                break;
        }

        //max mapping extension
        int max = 0;
        for (int k = (bins - 1); k >= 0; k--)
        {
            if (lookup (k, dim) != 0)
            {
                max = lookup (k, dim);
                break;
            }
        }

        for (int k = (bins - 1); k >= 0; k--)
        {
            if (lookup (k, dim) == 0)
                lookup (k, dim) = max;
            else
                break;
        }
    }
}


template<typename ModelT, typename SceneT>
bool
HypothesisVerification<ModelT, SceneT>::removeNanNormals (HVRecognitionModel<ModelT> &rm)
{
    if(!rm.visible_cloud_normals_) {
        rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal>);
        computeNormals<ModelT>(rm.visible_cloud_, rm.visible_cloud_normals_, param_.normal_method_);
    }

    //check nans...
    size_t kept = 0;
    for (size_t idx = 0; idx < rm.visible_cloud_->points.size (); idx++)
    {
        if ( pcl::isFinite(rm.visible_cloud_->points[idx]) && pcl::isFinite(rm.visible_cloud_normals_->points[idx]) )
        {
            rm.visible_cloud_->points[kept] = rm.visible_cloud_->points[idx];
            rm.visible_cloud_normals_->points[kept] = rm.visible_cloud_normals_->points[idx];
            kept++;
        }
    }

    rm.visible_cloud_->points.resize (kept);
    rm.visible_cloud_normals_->points.resize (kept);
    rm.visible_cloud_->width = rm.visible_cloud_normals_->width = kept;
    rm.visible_cloud_->height = rm.visible_cloud_normals_->height = 1;

    return !rm.visible_cloud_->points.empty();
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::computeModel2SceneDistances(HVRecognitionModel<ModelT> &rm)
{
    rm.explained_pts_per_smooth_cluster_.clear();
    rm.explained_pts_per_smooth_cluster_.resize(smooth_label_count_.size(), 0);

    rm.scene_model_sqr_dist_ = -1000.f * Eigen::VectorXf::Ones(scene_cloud_downsampled_->points.size());

    rm.model_scene_c_.resize( rm.visible_cloud_->points.size () * param_.knn_inliers_ );
    size_t kept=0;

    for (size_t m_pt_id = 0; m_pt_id < rm.visible_cloud_->points.size (); m_pt_id++)
    {
        std::vector<int> nn_indices;
        std::vector<float> nn_sqr_distances;
        octree_scene_downsampled_->nearestKSearch(rm.visible_cloud_->points[m_pt_id], param_.knn_inliers_, nn_indices, nn_sqr_distances);

        for (size_t k = 0; k < nn_indices.size(); k++)
        {
            int sidx = nn_indices[ k ];
            double sqr_3D_dist = nn_sqr_distances[k];

            //              if (sqr_3D_dist > ( 3 * 3 * param_.inliers_threshold_ * param_.inliers_threshold_ ) )
            //                  continue;

            pcl::Correspondence &c = rm.model_scene_c_[ kept ];
            c.index_query = m_pt_id;
            c.index_match = sidx;
            c.distance = sqr_3D_dist;



            double old_s_m_dist = rm.scene_model_sqr_dist_(sidx);
            if ( old_s_m_dist < -1.f || sqr_3D_dist < old_s_m_dist)
            {
                rm.scene_model_sqr_dist_(sidx) = sqr_3D_dist;

                if( param_.check_smooth_clusters_ && sqr_3D_dist < (param_.occlusion_thres_ * param_.occlusion_thres_ * 1.5 * 1.5)
                        && ( old_s_m_dist < -1.f || old_s_m_dist >= (param_.occlusion_thres_ * param_.occlusion_thres_ * 1.5 * 1.5) )) // if point is not already taken
                {
                    int l = scene_smooth_labels_[sidx];
                    rm.explained_pts_per_smooth_cluster_[l] ++;
                }
            }

            kept++;
        }
    }
    rm.model_scene_c_.resize(kept);
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::computeModel2SceneFitness(HVRecognitionModel<ModelT> &rm)
{
    rm.scene_explained_weight_ = -1000.f * Eigen::VectorXf::Ones (scene_cloud_downsampled_->points.size());
    Eigen::VectorXf modelFit   = -1000.f * Eigen::VectorXf::Ones (rm.visible_cloud_->points.size());

    Eigen::VectorXi modelSceneCorrespondence; // saves the correspondence of each visible model point to its closest scene point (weighted by 3D Euclidean Distance and color). Only used for visualization.
    if(param_.visualize_model_cues_)
        modelSceneCorrespondence = -1 * Eigen::VectorXi::Ones (rm.visible_cloud_->points.size()); // negative value means no correspondence

    double w3d = 1 / (param_.inliers_threshold_ * param_.inliers_threshold_);
    double w_color_AB = 1 / (param_.color_sigma_ab_ * param_.color_sigma_ab_);
    double w_color_L = 1 / (param_.color_sigma_l_ * param_.color_sigma_l_);
    double w_normals = 1 / (param_.sigma_normals_deg_ * param_.sigma_normals_deg_);

    for(size_t i=0; i<rm.model_scene_c_.size(); i++)
    {
        const pcl::Correspondence &c = rm.model_scene_c_[i];
        double sqr_3D_dist = c.distance;
        int sidx = c.index_match;
        int midx = c.index_query;

        Eigen::Vector3f normal_m = rm.visible_cloud_normals_->points[midx].getNormalVector3fMap();
        Eigen::Vector3f normal_s = scene_normals_downsampled_->points[sidx].getNormalVector3fMap();

        normal_m.normalize();
        normal_s.normalize();
        double dotp = normal_m.dot(normal_s);
        if(dotp>0.999f)
            dotp = 0.999f;
        if(dotp<-0.999f)
            dotp = -0.999f;
        double acoss = acos (dotp);
        double normal_angle_deg = pcl::rad2deg( acoss );
        double dist = w3d * sqr_3D_dist + w_normals * normal_angle_deg * normal_angle_deg;

        const Eigen::VectorXf &color_m = rm.pt_color_.row( midx );
        const Eigen::VectorXf &color_s = scene_color_channels_.row( sidx );


        if(param_.color_space_ == ColorTransformOMP::LAB)
        {
            double Ls = color_s(0);
            double As = color_s(1);
            double Bs = color_s(2);
            double Lm = std::max(0.f, std::min(100.f, rm.L_value_offset_ + color_m(0)) );
            double Am = color_m(1);
            double Bm = color_m(2);

            double sqr_color_dist_AB = ( (As-Am)*(As-Am)+(Bs-Bm)*(Bs-Bm) );
            dist += w_color_AB * sqr_color_dist_AB;

            double sqr_color_dist_L = ( (Ls-Lm)*(Ls-Lm) );
            dist += w_color_L * sqr_color_dist_L;
        }
        else
            throw std::runtime_error("Desired color space not implemented so far!");

        float old_scene2model_dist = rm.scene_explained_weight_(sidx); ///NOTE: negative if no points explains it yet
        if ( old_scene2model_dist < -1.f || dist<old_scene2model_dist)
            rm.scene_explained_weight_(sidx) = dist;    // 0 for perfect fit

        float old_model_pt_dist = modelFit(midx);
        if( old_model_pt_dist < -1.f || dist < old_model_pt_dist )
        {
            modelFit(midx) = dist;
            if(param_.visualize_model_cues_)
                modelSceneCorrespondence(midx) = sidx;
        }
    }

    // now we compute the exponential of the distance to bound it between 0 and 1 (whereby 1 means perfect fit and 0 no fit)
#pragma omp parallel for schedule(dynamic)
    for(int sidx=0; sidx < rm.scene_explained_weight_.rows(); sidx++)
    {
        float fit = rm.scene_explained_weight_(sidx);
        fit < -1.f ? fit = 0.f : fit = exp(-fit);
        rm.scene_explained_weight_(sidx) = fit;
    }

#pragma omp parallel for schedule(dynamic)
    for(int midx=0; midx < modelFit.rows(); midx++)
    {
        float fit = modelFit(midx);
        fit < -1.f ? fit = 0.f : fit = exp(-fit);
        modelFit(midx) = fit;
    }

    rm.model_fit_ = modelFit.sum();

    rm.model_scene_c_.clear(); // not needed any more
    if(param_.visualize_model_cues_)    // we store the fit for each model point in the correspondences vector
    {
        rm.model_scene_c_.resize(rm.visible_cloud_->points.size());
        for(size_t midx=0; midx<rm.visible_cloud_->points.size(); midx++)
        {
            int sidx = modelSceneCorrespondence(midx);
            pcl::Correspondence &c = rm.model_scene_c_[midx];
            c.index_query = midx;
            c.index_match = sidx;
            if( sidx>0 )
                c.weight = modelFit(midx);
            else
                c.weight = 0.f;
        }
    }
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::computeLoffset(HVRecognitionModel<ModelT> &rm) const
{
    // pre-allocate memory
    size_t kept = 0;
    for(int sidx=0; sidx<scene_model_sqr_dist_.rows(); sidx++)
    {
        if( rm.scene_model_sqr_dist_(sidx) > -1.f )
            kept++;
    }

    Eigen::MatrixXf croppedSceneColorMatrix (kept, scene_color_channels_.cols());
    kept = 0;
    for(int sidx=0; sidx < rm.scene_model_sqr_dist_.rows(); sidx++)
    {
        if( rm.scene_model_sqr_dist_(sidx) > -1.f )
        {
            croppedSceneColorMatrix.row(kept) = scene_color_channels_.row(sidx);
            kept++;
        }
    }

    Eigen::MatrixXf histLm, histLs;
    computeHistogram(rm.pt_color_.col(0), histLm, bins_, Lmin_, Lmax_);
    computeHistogram(croppedSceneColorMatrix.col(0), histLs, bins_, Lmin_, Lmax_);

    Eigen::VectorXf histLs_normalized(bins_), histLm_normalized (bins_);
    histLs_normalized = histLs.col(0);
    float Ls_sum = histLs_normalized.sum();
    histLs_normalized /= Ls_sum;
    histLm_normalized = histLm.col(0);
    float Lm_sum = histLm_normalized.sum();
    histLm_normalized /= Lm_sum;

    float best_corr = computeHistogramIntersection(histLs_normalized, histLm_normalized);
    int best_shift = 0;

    Eigen::VectorXf histLm_normalized_shifted_old = histLm_normalized;
    Eigen::VectorXf histLm_normalized_shifted;

    for(int shift=1; shift<std::floor(bins_/2); shift++) // shift right
    {
        shiftHistogram(histLm_normalized_shifted_old, histLm_normalized_shifted, true);
        float corr = computeHistogramIntersection( histLs_normalized,  histLm_normalized_shifted);
        if (corr>best_corr)
        {
            best_corr = corr;
            best_shift = shift;
        }
        histLm_normalized_shifted_old = histLm_normalized_shifted;
    }

    histLm_normalized_shifted_old = histLm_normalized;
    for(int shift=1; shift<std::floor(bins_/2); shift++) // shift left
    {
        shiftHistogram(histLm_normalized_shifted_old, histLm_normalized_shifted, false);
        float corr = computeHistogramIntersection( histLs_normalized,  histLm_normalized_shifted);
        if (corr>best_corr)
        {
            best_corr = corr;
            best_shift = -shift;
        }
        histLm_normalized_shifted_old = histLm_normalized_shifted;
    }

    rm.L_value_offset_ = best_shift * (Lmax_ - Lmin_) / bins_;
}

//######### VISUALIZATION FUNCTIONS #####################
template<>
void
HypothesisVerification<pcl::PointXYZ, pcl::PointXYZ>::visualizeGOCuesForModel(const HVRecognitionModel<pcl::PointXYZ> &rm) const
{
    (void)rm;
    std::cerr << "The visualization function is not defined for the chosen Point Cloud Type!" << std::endl;
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::visualizeGOCuesForModel(const HVRecognitionModel<ModelT> &rm) const
{
    if(!rm_vis_) {
        rm_vis_.reset (new pcl::visualization::PCLVisualizer ("model cues"));
        rm_vis_->createViewPort(0   , 0   , 0.25,0.5 , rm_v1);
        rm_vis_->createViewPort(0.25, 0   , 0.50,0.5 , rm_v2);
        rm_vis_->createViewPort(0.50, 0   , 0.75,0.5 , rm_v3);
        rm_vis_->createViewPort(0.75, 0   , 1   ,0.5 , rm_v4);

        rm_vis_->createViewPort(0   , 0.5 , 0.25,1   , rm_v5);
        rm_vis_->createViewPort(0.25, 0.5 , 0.50,1   , rm_v6);
        rm_vis_->createViewPort(0.50, 0.5 , 0.75,1   , rm_v7);

        rm_vis_->createViewPort(0.75, 0.5 , 0.875   , 0.75   , rm_v11);
        rm_vis_->createViewPort(0.875, 0.5 , 1   , 0.75   , rm_v12);
        rm_vis_->createViewPort(0.75, 0.75 , 0.875   ,1   , rm_v11);
        rm_vis_->createViewPort(0.875, 0.75 , 1   ,1   , rm_v12);

        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v1);
        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v2);
        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v3);
        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v4);
        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v5);
        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v6);
        rm_vis_->setBackgroundColor(255.f, 255.f, 255.f, rm_v7);
    }

    rm_vis_->removeAllPointClouds();
    rm_vis_->removeAllShapes();

    if(!param_.vis_for_paper_)
        rm_vis_->addText("scene",10,10,12,1,1,1,"scene",rm_v1);

    rm_vis_->addPointCloud(scene_cloud_downsampled_, "scene1",rm_v1);

#ifdef L_HIST
    // compute color histogram for visible model points
    {
        pcl::PointCloud<ModelT> m_cloud_orig, m_cloud_color_reg, m_cloud_color_reg2;
        m_cloud_orig.points.resize( rm.visible_indices_.size() );
        m_cloud_color_reg.points.resize( rm.visible_indices_.size() );
        m_cloud_color_reg2.points.resize( rm.visible_indices_.size() );

        Eigen::MatrixXf colorVisibleCloud (rm.visible_indices_.size(), rm.pt_color_.cols());
        for(size_t i=0; i<rm.visible_indices_.size(); i++)
        {
            int idx = rm.visible_indices_[i];
            colorVisibleCloud.row(i) = rm.pt_color_.row(idx);

            ModelT &m = m_cloud_orig.points[i];
            ModelT &m2 = m_cloud_color_reg2.points[i];
            m = rm.visible_cloud_->points[i];
            m2 = rm.visible_cloud_->points[i];
            ColorTransform::CIELAB2RGB( colorVisibleCloud(i,0), colorVisibleCloud(i,1), colorVisibleCloud(i,2), m.r, m.g, m.b);

            float l_w_offset = std::max(0.f, std::min(100.f, rm.L_value_offset_ + colorVisibleCloud(i,0)) );
            ColorTransform::CIELAB2RGB( l_w_offset, colorVisibleCloud(i,1), colorVisibleCloud(i,2), m2.r, m2.g, m2.b);
        }

        Eigen::MatrixXf histLm, histLs;
        computeHistogram(colorVisibleCloud.col(0), histLm, bins_, Lmin_, Lmax_);

        // pre-allocate memory
        size_t kept = 0;
        for(size_t sidx=0; sidx<scene_explained_weight_.rows(); sidx++)
        {
            if(scene_explained_weight_(sidx,model_id) > std::numeric_limits<float>::epsilon())
                kept++;
        }
        Eigen::MatrixXf croppedSceneColorMatrix (kept, scene_color_channels_.cols());
        pcl::PointCloud<SceneT> s_cloud_orig, s_cloud_color_reg;
        s_cloud_orig.points.resize( kept );
        s_cloud_color_reg.points.resize( kept );
        kept = 0;
        for(size_t sidx=0; sidx<scene_explained_weight_.rows(); sidx++)
        {
            if(scene_explained_weight_(sidx,model_id) > std::numeric_limits<float>::epsilon())
            {
                croppedSceneColorMatrix.row(kept) = scene_color_channels_.row(sidx);
                SceneT &s = s_cloud_orig.points[kept];
                s = scene_cloud_downsampled_->points[sidx];
                ColorTransform::CIELAB2RGB( croppedSceneColorMatrix(kept,0), croppedSceneColorMatrix(kept,1), croppedSceneColorMatrix(kept,2), s.r, s.g, s.b);
                kept++;
            }
        }

        //        for(size_t idx=0; idx<rm.model_scene_c_.size(); idx++)
        //        {
        //            int scene_idx = rm.model_scene_c_[idx].index_match;
        //            croppedSceneColorMatrix.row(idx) = scene_color_channels_.row(scene_idx);

        //            SceneT &s = s_cloud_orig.points[idx];
        //            s = scene_cloud_downsampled_->points[scene_idx];
        //            ColorTransform::CIELAB2RGB( croppedSceneColorMatrix(idx,0), croppedSceneColorMatrix(idx,1), croppedSceneColorMatrix(idx,2), s.r, s.g, s.b);
        //        }
        computeHistogram(croppedSceneColorMatrix.col(0), histLs, bins_, Lmin_, Lmax_);
        Eigen::VectorXf histLs_normalized(bins_), histLm_normalized (bins_);
        histLs_normalized = histLs.col(0);
        float Ls_sum = histLs_normalized.sum();
        histLs_normalized /= Ls_sum;
        histLm_normalized = histLm.col(0);
        float Lm_sum = histLm_normalized.sum();
        histLm_normalized /= Lm_sum;

        Eigen::MatrixXf lookUpTable;
        specifyHistogram(histLs_normalized, histLm_normalized, lookUpTable);
        float bin_size = (Lmax_-Lmin_) / bins_;
        double fitness_orig=0.f, fitness_reg = 0.f, fitness_reg2 = 0.f;
        for(size_t idx=0; idx<rm.model_scene_c_.size(); idx++)
        {
            const pcl::Correspondence &c = rm.model_scene_c_[idx];
            int sidx = c.index_match;
            int midx = c.index_query;

            if(sidx<0)
                continue;

            float Ls = scene_color_channels_(sidx,0);
            float Lm = colorVisibleCloud(midx, 0);

            int pos = std::floor( (Lm - Lmin_) / bin_size);

            if(pos < 0)
                pos = 0;

            if(pos > (int)bins_)
                pos = bins_ - 1;

            float Lm_reg = lookUpTable(pos,0);

            ModelT &m = m_cloud_color_reg.points[midx];
            m = rm.visible_cloud_->points[midx];
            ColorTransform::CIELAB2RGB( Lm_reg, colorVisibleCloud(midx,1), colorVisibleCloud(midx,2), m.r, m.g, m.b);

            float Lm_reg2 = std::max(0.f, std::min(100.f, rm.L_value_offset_ + Lm) );
            fitness_orig += exp(- (Ls - Lm)*(Ls - Lm)/ (param_.color_sigma_ab_ * param_.color_sigma_ab_) );
            fitness_reg += exp(- (Ls - Lm_reg)*(Ls - Lm_reg)/ (param_.color_sigma_ab_ * param_.color_sigma_ab_) );
            fitness_reg2 += exp(- (Ls - Lm_reg2)*(Ls - Lm_reg2)/ (param_.color_sigma_ab_ * param_.color_sigma_ab_) );
        }

        std::cout << "fitness original: " << fitness_orig/rm.visible_cloud_->points.size() << "; after registration: " <<
                     fitness_reg/rm.visible_cloud_->points.size() << "; after registration2: " <<
                     fitness_reg2/rm.visible_cloud_->points.size() << "." << std::endl;

        rm_vis_->addText("explained scene cloud original", 10, 10, 10, 1,1,1,"s_cloud_orig", rm_v9);
        rm_vis_->addText("model cloud color registered", 10, 10, 10, 1,1,1,"s_cloud_color_reg2", rm_v10);
        rm_vis_->addText("model cloud original", 10, 10, 10, 1,1,1,"m_cloud_orig", rm_v11);
        rm_vis_->addText("model cloud color registered", 10, 10, 10, 1,1,1,"m_cloud_reg", rm_v12);
        rm_vis_->addPointCloud(s_cloud_orig.makeShared(), "s_cloud_original", rm_v9);
        rm_vis_->addPointCloud(m_cloud_color_reg2.makeShared(), "m_cloud_color_registered2", rm_v10);
        rm_vis_->addPointCloud(m_cloud_orig.makeShared(), "m_cloud_original", rm_v11);
        rm_vis_->addPointCloud(m_cloud_color_reg.makeShared(), "m_cloud_color_registered", rm_v12);
    }
#endif
    typename pcl::PointCloud<ModelT>::Ptr visible_cloud_colored (new pcl::PointCloud<ModelT> (*rm.complete_cloud_));
    for(size_t i=0; i<visible_cloud_colored->points.size(); i++)
    {
        ModelT &mp = visible_cloud_colored->points[i];
        mp.r = mp.g = mp.b = 0.f;
    }

    for(size_t i=0; i<rm.visible_indices_.size(); i++)
    {
        int idx = rm.visible_indices_[i];
        ModelT &mp = visible_cloud_colored->points[idx];
        mp.r = 255.f;
        mp.g = mp.b = 0.f;
    }

    std::stringstream txt; txt << "visible ratio: " << std::fixed << std::setprecision(2) << (float)rm.visible_cloud_->points.size() / (float)rm.complete_cloud_->points.size();

    if(!param_.vis_for_paper_)
        rm_vis_->addText(txt.str(),10,10,12,0,0,0,"visible model cloud",rm_v2);

    rm_vis_->addPointCloud(visible_cloud_colored, "model2",rm_v2);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "model2", rm_v2);

    typename pcl::PointCloud<ModelT>::Ptr model_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
    for(size_t p=0; p < model_fit_cloud->points.size(); p++)
    {
        ModelT &mp = model_fit_cloud->points[p];
        mp.r = mp.b = 0.f;
        mp.g = 50.f;
    }
    for(size_t cidx=0; cidx < rm.model_scene_c_.size(); cidx++)
    {
        const pcl::Correspondence &c = rm.model_scene_c_[cidx];
        int sidx = c.index_match;
        int midx = c.index_query;
        float weight = c.weight;

        if(sidx<0)
            continue;

        ModelT &mp = model_fit_cloud->points[midx];
        mp.g = (255.f - mp.g) * weight;   // scale green channel with fitness score
    }
    txt.str(""); txt << "model cost: " << std::fixed << std::setprecision(4) << rm.model_fit_ <<
                        "; normalized: " << rm.model_fit_ / rm.visible_cloud_->points.size();
    rm_vis_->addText(txt.str(),10,10,12,0,0,0,"model cost",rm_v3);
    rm_vis_->addPointCloud(model_fit_cloud, "model cost", rm_v3);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                              5, "model cost", rm_v3);


    // ---- VISUALIZE SMOOTH SEGMENTATION -------
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_smooth_labels_rgb (new pcl::PointCloud<pcl::PointXYZRGB>(*scene_cloud_downsampled_));
    if(!smooth_label_count_.empty())
    {
        Eigen::Matrix3Xf label_colors (3, smooth_label_count_.size());
        for(size_t i=0; i<smooth_label_count_.size(); i++)
        {
            float r,g,b;
            if( i==0 )
            {
                r = g = b = 255; // label 0 will be white
            }
            else
            {
                r = rand () % 255;
                g = rand () % 255;
                b = rand () % 255;
            }
            label_colors(0,i) = r;
            label_colors(1,i) = g;
            label_colors(2,i) = b;

            if(!param_.vis_for_paper_)
            {
                std::stringstream lbl_txt; lbl_txt << std::fixed << std::setprecision(2) << rm.explained_pts_per_smooth_cluster_[i] << " / " << smooth_label_count_[i];
                std::stringstream txt_id; txt_id << "smooth_cluster_txt " << i;
                rm_vis_->addText( lbl_txt.str(), 10, 10+12*i, 12, r/255, g/255, b/255, txt_id.str(), rm_v4);
            }
        }

        for(size_t i=0; i < scene_smooth_labels_.size(); i++)
        {
            int l = scene_smooth_labels_[i];
            pcl::PointXYZRGB &p = scene_smooth_labels_rgb->points[i];
            p.r = label_colors(0,l);
            p.g = label_colors(1,l);
            p.b = label_colors(2,l);
        }
        rm_vis_->addPointCloud(scene_smooth_labels_rgb, "smooth labels", rm_v4);
    }
    //---- END VISUALIZE SMOOTH SEGMENTATION-----------


    typename pcl::PointCloud<SceneT>::Ptr scene_fit_cloud (new pcl::PointCloud<SceneT> (*scene_cloud_downsampled_));

    for(int p=0; p < rm.scene_explained_weight_.rows(); p++)
    {
        SceneT &sp = scene_fit_cloud->points[p];
        sp.r = sp.b = 0.f;

        sp.g = 255.f * rm.scene_explained_weight_(p);
    }
    txt.str(""); txt << "scene pts explained (fitness: " << rm.scene_explained_weight_.sum() <<
                        "; normalized: " << rm.scene_explained_weight_.sum()/scene_cloud_downsampled_->points.size() << ")";
    rm_vis_->addText(txt.str(),10,10,12,0,0,0,"scene fitness",rm_v5);
    rm_vis_->addPointCloud(scene_fit_cloud, "scene fitness", rm_v5);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                              5, "scene fitness", rm_v5);

    rm_vis_->addText("scene and visible model",10,10,12,1,1,1,"scene_and_model",rm_v6);
    rm_vis_->addPointCloud(scene_cloud_downsampled_, "scene_model_1", rm_v6);
    rm_vis_->addPointCloud(rm.visible_cloud_, "scene_model_2", rm_v6);
    rm_vis_->addPointCloud(rm.visible_cloud_, "scene_model_4", rm_v4);

    rm_vis_->resetCamera();
    rm_vis_->spin();
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::visualizePairwiseIntersection() const
{
    if(!vis_pairwise_)
    {
        vis_pairwise_.reset( new pcl::visualization::PCLVisualizer("intersection") );
        vis_pairwise_->createViewPort(0,0,0.5,1, vp_pair_1_);
        vis_pairwise_->createViewPort(0.5,0,1,1, vp_pair_2_);
    }

    for(size_t i=1; i<global_hypotheses_.size(); i++)
    {
        const HVRecognitionModel<ModelT> &rm_a = *global_hypotheses_[i];

        for(size_t j=0; j<i; j++)
        {
            const HVRecognitionModel<ModelT> &rm_b = *global_hypotheses_[j];

            std::stringstream txt;
            txt <<  "intersection cost (" << i << ", " << j << "): " << intersection_cost_(j,i);

            vis_pairwise_->removeAllPointClouds();
            vis_pairwise_->removeAllShapes();
            vis_pairwise_->addText(txt.str(), 10, 10, 12, 1.f, 1.f, 1.f, "intersection_text", vp_pair_1_ );
            vis_pairwise_->addPointCloud(rm_a.visible_cloud_, "cloud_a", vp_pair_1_);
            vis_pairwise_->addPointCloud(rm_b.visible_cloud_, "cloud_b", vp_pair_1_);
            vis_pairwise_->setBackgroundColor(1,1,1,vp_pair_1_);
            vis_pairwise_->setBackgroundColor(1,1,1,vp_pair_2_);
            //            vis.addPointCloud(rendered_vis_m_a.makeShared(), "cloud_ar",v2);
            //            vis.addPointCloud(rendered_vis_m_b.makeShared(), "cloud_br",v2);
            vis_pairwise_->resetCamera();
            vis_pairwise_->spin();
        }
    }
}


template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::visualizeGOCues (const std::vector<bool> & active_solution, float cost, int times_evaluated) const
{
    (void)active_solution;
    (void)cost;
    (void)times_evaluated;
    std::cerr << "Visualizing GO Cues is only implemented for XYZRGB point clouds." << std::endl;
}


template<>
void
HypothesisVerification<pcl::PointXYZRGB, pcl::PointXYZRGB>::visualizeGOCues (const std::vector<bool> & active_solution, float cost, int times_evaluated) const
{
    typedef pcl::PointXYZRGB ModelT;
    typedef pcl::PointXYZRGB SceneT;

    if(!vis_go_cues_) {
        vis_go_cues_.reset(new pcl::visualization::PCLVisualizer("visualizeGOCues"));
        vis_go_cues_->createViewPort(0, 0, 0.33, 0.5, vp_scene_);
        vis_go_cues_->createViewPort(0.33, 0, 0.66, 0.5, vp_active_hypotheses_);
        vis_go_cues_->createViewPort(0.66, 0, 1, 0.5, vp_model_fitness_);
        vis_go_cues_->createViewPort(0, 0.5, 0.33, 1, vp_scene_fitness_);
    }

    vis_go_cues_->removeAllPointClouds();
    vis_go_cues_->removeAllShapes();

    double model_fitness = 0.f;
    double pairwise_cost = 0.f;
    double scene_fitness = 0.f;

    // model uni term
    size_t num_active_hypotheses = 0;
    for(size_t i=0; i<active_solution.size(); i++)
    {
        if(active_solution[i]) {
            model_fitness += global_hypotheses_[i]->model_fit_;
            num_active_hypotheses++;
        }
    }
    if(!num_active_hypotheses)
        model_fitness = 0.f;

    // scene uni term
    Eigen::MatrixXf scene_explained_weight_for_active_hypotheses = scene_explained_weight_;
    for(size_t i=0; i<active_solution.size(); i++)
    {
        if(!active_solution[i]) {
            scene_explained_weight_for_active_hypotheses.col(i) = Eigen::VectorXf::Zero(
                        scene_explained_weight_for_active_hypotheses.rows());
        }
    }

    if ( scene_explained_weight_for_active_hypotheses.cols() ) {
        Eigen::VectorXf max = scene_explained_weight_for_active_hypotheses.rowwise().maxCoeff();
        //        scene_fitness = max.sum() / scene_cloud_downsampled_->points.size();
        scene_fitness = max.sum();
    }


    // pairwise term
    for(size_t i=0; i<active_solution.size(); i++)
    {
        for(size_t j=0; j<i; j++)
        {
            if(active_solution[i] && active_solution[j])
                pairwise_cost += intersection_cost_(i,j);
        }
    }


    std::ostringstream out, model_fitness_txt, scene_fitness_txt;
    out << "Active Hypotheses" << std::endl;
    out << "Cost: " << std::setprecision(5) << cost << " , #Evaluations: " << times_evaluated;
    out << std::endl << "; pairwise cost: " << pairwise_cost << "; total cost: " << cost_ << std::endl;
    model_fitness_txt << "model fitness: " << model_fitness;
    scene_fitness_txt << "scene fitness: " << scene_fitness;


    vis_go_cues_->addText ("Scene", 1, 30, 16, 1, 1, 1, "inliers_outliers", vp_scene_);
    vis_go_cues_->addText (out.str(), 1, 30, 16, 1, 1, 1, "scene_cues", vp_active_hypotheses_);
    vis_go_cues_->addText (model_fitness_txt.str(), 1, 30, 16, 1, 1, 1, "model fitness", vp_model_fitness_);
    vis_go_cues_->addText (scene_fitness_txt.str(), 1, 30, 16, 1, 1, 1, "scene fitness", vp_scene_fitness_);
    vis_go_cues_->addPointCloud (scene_cloud_downsampled_, "scene_cloud", vp_scene_);

    //display active hypotheses
    for(size_t i=0; i < active_solution.size(); i++)
    {
        if(active_solution[i])
        {
            HVRecognitionModel<ModelT> &rm = *global_hypotheses_[i];
            std::stringstream model_name; model_name << "model_" << i;
            vis_go_cues_->addPointCloud(rm.visible_cloud_, model_name.str(), vp_active_hypotheses_);

            typename pcl::PointCloud<ModelT>::Ptr model_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
            for(size_t p=0; p < model_fit_cloud->points.size(); p++)
            {
                ModelT &mp = model_fit_cloud->points[p];
                mp.r = mp.g = 0.f;

                const pcl::Correspondence &c = rm.model_scene_c_[p];
                mp.b = 50.f + 205.f * c.weight;
            }

            model_name << "_fitness";
            vis_go_cues_->addPointCloud(model_fit_cloud, model_name.str(), vp_model_fitness_);
        }
    }

    if ( scene_explained_weight_for_active_hypotheses.cols() ) {
        Eigen::VectorXf max_fit = scene_explained_weight_for_active_hypotheses.rowwise().maxCoeff();
        typename pcl::PointCloud<SceneT>::Ptr scene_fit_cloud (new pcl::PointCloud<SceneT> (*scene_cloud_downsampled_));

        for(size_t p=0; p<scene_fit_cloud->points.size(); p++)
        {
            SceneT &sp = scene_fit_cloud->points[p];
            sp.r = sp.g = 0.f;

            sp.b = 50.f + 205.f * max_fit(p);
        }

        vis_go_cues_->addPointCloud(scene_fit_cloud, "scene fitness", vp_scene_fitness_);
    }

    vis_go_cues_->resetCamera();
    vis_go_cues_->spin();
}

//template class V4R_EXPORTS HypothesisVerification<pcl::PointXYZ,pcl::PointXYZ>;
template class V4R_EXPORTS HypothesisVerification<pcl::PointXYZRGB,pcl::PointXYZRGB>;
}
