#include <v4r/common/histogram.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/normals.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/occlusion_reasoning.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/common/plane_utils.h>
#include <v4r/common/zbuffering.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/segmentation/ClusterNormalsToPlanesPCL.h>
#include <v4r/segmentation/multiplane_segmenter.h>
#include <v4r/segmentation/segmentation_utils.h>

#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>
#include <pcl_1_8/keypoints/uniform_sampling.h>
#include <pcl/common/angles.h>
#include <pcl/common/time.h>
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/sac_segmentation.h>


#include <fstream>
#include <iomanip>
#include <omp.h>
#include <numeric>

namespace v4r
{

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::computeModelOcclusionByScene(HVRecognitionModel<ModelT> &rm) const
{
    boost::dynamic_bitset<> image_mask_mv(rm.complete_cloud_->points.size(), 0);

    for(size_t view=0; view<occlusion_clouds_.size(); view++)
    {
        // project into respective view
        typename pcl::PointCloud<ModelT>::Ptr aligned_cloud (new pcl::PointCloud<ModelT>);
        const Eigen::Matrix4f tf = absolute_camera_poses_[view].inverse();
        pcl::transformPointCloud(*rm.complete_cloud_, *aligned_cloud, tf);

        OcclusionReasoner<SceneT, ModelT> occ_reasoner;
        occ_reasoner.setCamera(cam_);
        occ_reasoner.setInputCloud( aligned_cloud );
        occ_reasoner.setOcclusionCloud( occlusion_clouds_[view] );
        occ_reasoner.setOcclusionThreshold( param_.occlusion_thres_ );
        image_mask_mv |=  occ_reasoner.computeVisiblePoints();
        rm.image_mask_[view] = occ_reasoner.getPixelMask();
    }

    rm.visible_indices_ = createIndicesFromMask<int>(image_mask_mv);
    pcl::copyPointCloud (*rm.complete_cloud_, rm.visible_indices_, *rm.visible_cloud_);    
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::computeVisibleOctreeNodes(HVRecognitionModel<ModelT> &rm)
{
    boost::dynamic_bitset<> visible_mask = v4r::createMaskFromIndices( rm.visible_indices_, rm.complete_cloud_->points.size() );
    auto octree_it = octree_model_representation_.find( rm.model_id_ );

    if(octree_it == octree_model_representation_.end())
        std::cerr << "Did not find octree representation! This should not happen!" << std::endl;

    boost::dynamic_bitset<> visible_leaf_mask (rm.complete_cloud_->points.size(), 0);

    size_t total_leafs = 0;
    size_t visible_leafs = 0;
    for (auto leaf_it = octree_it->second->leaf_begin(); leaf_it != octree_it->second->leaf_end(); ++leaf_it)
    {
        pcl::octree::OctreeContainerPointIndices& container = leaf_it.getLeafContainer();

        // add points from leaf node to indexVector
        std::vector<int> indexVector;
        container.getPointIndices (indexVector);

        if(indexVector.empty())
            continue;

        total_leafs++;

        bool is_visible = false;

        for(size_t k=0; k < indexVector.size(); k++)
        {
            if (visible_mask [ indexVector[k] ] )
            {
                is_visible = true;
                visible_leafs++;
                break;
            }
        }
        if (is_visible)
        {
            for(size_t k=0; k < indexVector.size(); k++)
                visible_leaf_mask.set( indexVector[k] );
        }
    }
    rm.visible_indices_by_octree_ = v4r::createIndicesFromMask<int>(visible_leaf_mask);
}


template<typename ModelT, typename SceneT>
Eigen::Matrix4f
HypothesisVerification<ModelT, SceneT>::refinePose(HVRecognitionModel<ModelT> &rm) const
{
//    ModelT minPoint, maxPoint;
//    pcl::getMinMax3D(*rm.complete_cloud_, minPoint, maxPoint);
//    float margin = 0.05f;
//    minPoint.x -= margin;
//    minPoint.y -= margin;
//    minPoint.z -= margin;

//    maxPoint.x += margin;
//    maxPoint.y += margin;
//    maxPoint.z += margin;

//    typename pcl::PointCloud<SceneT>::Ptr scene_cloud_downsampled_cropped (new pcl::PointCloud<SceneT>);
//    typename pcl::PointCloud<SceneT>::Ptr scene_in_model_co (new pcl::PointCloud<SceneT>);

//    const Eigen::Matrix4f tf = rm.transform_;
//    Eigen::Matrix4f tf_inv = tf.inverse();
//    pcl::transformPointCloud ( *scene_cloud_downsampled_, *scene_in_model_co, tf_inv );

//    Eigen::Matrix3f rot_tmp  = tf.block<3,3>(0,0);
//    Eigen::Vector3f trans_tmp = tf.block<3,1>(0,3);
//    Eigen::Affine3f affine_trans;
//    pcl::CropBox<SceneT> cropFilter;
//    cropFilter.setInputCloud (scene_in_model_co);
//    cropFilter.setMin(rm.model_->maxPoint_.getVector4fMap());
//    cropFilter.setMax(rm.model_->minPoint_.getVector4fMap());
//    affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
//    cropFilter.setTransform(affine_trans);
//    cropFilter.filter (*scene_cloud_downsampled_cropped);

    pcl::IterativeClosestPoint<ModelT, SceneT> icp;
    icp.setInputSource(rm.visible_cloud_);
    icp.setInputTarget(scene_cloud_downsampled_);
    icp.setMaximumIterations(param_.icp_iterations_);
    icp.setMaxCorrespondenceDistance(param_.inliers_threshold_);
    icp.setSearchMethodTarget(kdtree_scene_, true);
    pcl::PointCloud<ModelT> aligned_visible_model;
    icp.align(aligned_visible_model);

    Eigen::Matrix4f refined_tf = Eigen::Matrix4f::Identity();
    if(icp.hasConverged())
        refined_tf = icp.getFinalTransformation();
    else
        std::cout << "ICP did not converge" << std::endl;

//    pcl::visualization::PCLVisualizer vis_tmp;
//    int vp2, vp3,vp1;
//    vis_tmp.createViewPort(0,0,0.33,1, vp1);
//    vis_tmp.createViewPort(0.33,0,0.66,1, vp2);
//    vis_tmp.createViewPort(0.66,0,1,1, vp3);

//    vis_tmp.addPointCloud(rm.model_->getAssembled(-1), "model1", vp1);
//     pcl::visualization::PointCloudColorHandlerCustom<SceneT> gray (scene_cloud_downsampled_, 128, 128, 128);
//     vis_tmp.addPointCloud(scene_cloud_downsampled_, gray, "scene1", vp1);
//     vis_tmp.addPointCloud(rm.complete_cloud_, "model1", vp1);

//     vis_tmp.addPointCloud(scene_cloud_downsampled_, gray, "scene2", vp2);
//     typename pcl::PointCloud<SceneT>::Ptr model_refined (new pcl::PointCloud<SceneT>);
//     pcl::transformPointCloud(*rm.complete_cloud_, *model_refined, refined_tf);
//     vis_tmp.addPointCloud(model_refined, "model2", vp2);
//     vis_tmp.spin();


//    vis_tmp.addPointCloud(rm.model_->getAssembled(-1), "model2", vp2);
//    vis_tmp.addPointCloud(scene_cloud_downsampled_cropped, "scene2", vp2);
//    vis_tmp.addSphere(rm.model_->minPoint_, 0.03, 0, 255, 0, "min", vp2 );
//    vis_tmp.addSphere(rm.model_->maxPoint_, 0.03, 0, 255, 0, "max", vp2 );

//    vis_tmp.addPointCloud(rm.model_->getAssembled(-1), "model", vp3);
//    vis_tmp.addPointCloud(scene_cloud_downsampled_cropped, "scene3", vp3);

//    typename pcl::PointCloud<SceneT>::Ptr model_refined (new pcl::PointCloud<SceneT>);
//    pcl::transformPointCloud(*rm.complete_cloud_, *model_refined, refined_tf);
//    vis_tmp.addPointCloud(model_refined, "model2", vp3);
//    pcl::visualization::PointCloudColorHandlerCustom<SceneT> gray (scene_cloud_downsampled_, 128, 128, 128);
//    vis_tmp.addPointCloud(scene_cloud_downsampled_, gray, "input_rm_vp2_model_", vp3);
//    vis_tmp.setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_rm_vp2_model_", vp3);
//    vis_tmp.addCube( minPoint.x, maxPoint.x, minPoint.y, maxPoint.y, minPoint.z, maxPoint.z, 0, 1., 0, "bb", vp3);
//    vis_tmp.spin();

    return refined_tf;
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
        scene_cloud_downsampled_.reset(new pcl::PointCloud<SceneT>());
        scene_normals_downsampled_.reset(new pcl::PointCloud<pcl::Normal>());

        pcl_1_8::UniformSampling<SceneT> us;
        us.setRadiusSearch( param_.resolution_mm_ / 1000.0 );
        us.setInputCloud( scene_cloud_ );
        pcl::PointCloud<int> sampled_indices;
        us.compute(sampled_indices);
        scene_sampled_indices_.clear();
        scene_sampled_indices_.resize(sampled_indices.points.size());
        for(size_t i=0; i < scene_sampled_indices_.size(); i++)
            scene_sampled_indices_[i] = sampled_indices.points[i];

        pcl::copyPointCloud(*scene_cloud_, scene_sampled_indices_, *scene_cloud_downsampled_);
        pcl::copyPointCloud(*scene_normals_, scene_sampled_indices_, *scene_normals_downsampled_);

        for(size_t i=0; i<scene_normals_downsampled_->points.size(); i++)
            scene_normals_downsampled_->points[i].curvature = scene_normals_->points[ scene_sampled_indices_[i] ].curvature;
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
    for (size_t i = 0; i < scene_cloud_downsampled_->points.size (); i++)
    {
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
void
HypothesisVerification<ModelT, SceneT>::updateTerms (const boost::dynamic_bitset<> &solution)
{
    solution_tmp_ = solution;

    CHECK(solution.size() == solution_.size() );

    std::vector<size_t> changed_ids;
    Eigen::VectorXf tmp_solution_f = Eigen::VectorXf ( solution_.size());
    for(size_t i=0; i<solution.size(); i++)
    {
        solution[i] ? tmp_solution_f(i) = 1.f : tmp_solution_f(i) = 0.f;

        if( solution_[i] != solution[i] )
            changed_ids.push_back(i);
    }

    // update fitness terms from starting point

    // here we compute from scratch
    ///TODO: check if we can use previous computations and if it makes algorithm faster

    double sum_explained = 0.;
    for(auto spt_it = scene_pts_explained_vec_.begin(); spt_it!=scene_pts_explained_vec_.end(); ++spt_it)
    {
        for(const auto &pt_fit:*spt_it)
        {
            if( solution_tmp_[pt_fit.rm_id_] )
            {
                sum_explained += pt_fit.fit_;
                break;
            }
        }
    }

    model_fitness_tmp_ = 0.f; //model_fitness_v_.dot(tmp_solution_f);
    scene_fitness_tmp_ = sum_explained;
    pairwise_cost_tmp_ = 0.5 * tmp_solution_f.transpose() * intersection_cost_ * tmp_solution_f;
}



template<typename ModelT, typename SceneT>
mets::gol_type
HypothesisVerification<ModelT, SceneT>::evaluateSolution (const boost::dynamic_bitset<> &solution)
{
    CHECK( solution == solution_tmp_ );

    double cost = -( param_.regularizer_ * scene_fitness_tmp_ + model_fitness_tmp_ - param_.clutter_regularizer_ * pairwise_cost_tmp_ );

//    std::cout << "Active Hypotheses: " << solution_tmp_ << std::endl
//              << "Cost: " << cost << "; pairwise cost: " << pairwise_cost_tmp_ << "; scene fitness: " << scene_fitness_tmp_ << "; model fitness: " << model_fitness_tmp_ << std::endl << std::endl;

    if(cost_logger_)
    {
        cost_logger_->increaseEvaluated();
        cost_logger_->addCostEachTimeEvaluated(cost);
    }

    return static_cast<mets::gol_type> (cost); //return the dual to our max problem
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

            if( (float)rm->visible_indices_by_octree_.size() / (float)rm->complete_cloud_->points.size() < param_.min_visible_ratio_)
            {
                rm->rejected_due_to_low_visibility_ = true;

                std::cout << "Removed " << rm->model_id_ << " due to low visibility!" << std::endl;

                if(!param_.visualize_model_cues_)
                    rm->freeSpace();
            }
        }
    }
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::setHypotheses(const std::vector<ObjectHypothesesGroup<ModelT> > &ohs)
{
    obj_hypotheses_groups_.clear();
    obj_hypotheses_groups_.resize(ohs.size());
    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
    {
        const ObjectHypothesesGroup<ModelT> &ohg = ohs[i];

        obj_hypotheses_groups_[i].resize(ohg.ohs_.size());
        for(size_t jj=0; jj<ohg.ohs_.size(); jj++)
        {
            const ObjectHypothesis<ModelT> &oh = *ohg.ohs_[jj];
            obj_hypotheses_groups_[i][jj].reset ( new HVRecognitionModel<ModelT>(oh) );
            HVRecognitionModel<ModelT> &hv_oh = *obj_hypotheses_groups_[i][jj];

            hv_oh.complete_cloud_.reset ( new pcl::PointCloud<ModelT>);
            hv_oh.complete_cloud_normals_.reset (new pcl::PointCloud<pcl::Normal>);

            bool found_model;
            typename Model<ModelT>::ConstPtr m = m_db_->getModelById("", oh.model_id_, found_model);
            typename pcl::PointCloud<ModelT>::ConstPtr model_cloud = m->getAssembled ( -1 );
            pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud_const = m->getNormalsAssembled ( -1 );
            pcl::transformPointCloud (*model_cloud, *hv_oh.complete_cloud_, oh.transform_);
            transformNormals(*normal_cloud_const, *hv_oh.complete_cloud_normals_, oh.transform_);
        }
    }
}


template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::extractEuclideanClustersSmooth()
{
    typename pcl::PointCloud<SceneTWithNormal>::Ptr scene_downsampled_w_normals (new pcl::PointCloud<SceneTWithNormal>);
    pcl::concatenateFields(*scene_cloud_downsampled_, *scene_normals_downsampled_, *scene_downsampled_w_normals);

    boost::function<bool (const SceneTWithNormal&, const SceneTWithNormal&, float)> custom_f = boost::bind (&HypothesisVerification<ModelT, SceneT>::customRegionGrowing, this, _1, _2, _3);

    pcl::ConditionalEuclideanClustering<SceneTWithNormal> cec (false);
    cec.setInputCloud (scene_downsampled_w_normals);
    cec.setConditionFunction (custom_f);
    cec.setClusterTolerance ( param_.cluster_tolerance_ * 3. );
    cec.setMinClusterSize ( param_.min_points_ );
    cec.setMaxClusterSize (std::numeric_limits<int>::max());
    pcl::IndicesClusters clusters;
    cec.segment (clusters);

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
void
HypothesisVerification<ModelT, SceneT>::computeLOffset(HVRecognitionModel<ModelT> &rm)const
{
    Eigen::VectorXf color_s ( rm.scene_explained_weight_.nonZeros() );

    size_t kept=0;
    for (Eigen::SparseVector<float>::InnerIterator it(rm.scene_explained_weight_); it; ++it)
    {
        int sidx = it.index();
        color_s(kept++) = scene_color_channels_(sidx,0);
    }

    Eigen::VectorXf color_new = specifyHistogram( rm.pt_color_.col( 0 ), color_s, 100, 0.f, 100.f );
    rm.pt_color_.col( 0 ) = color_new;
}

template<typename ModelT, typename SceneT>
bool
HypothesisVerification<ModelT, SceneT>::individualRejection(HVRecognitionModel<ModelT> &rm) const
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

    float model_fitness = rm.model_fit_ / rm.visible_cloud_->points.size();
    rm.rejected_due_to_low_model_fitness_ = ( model_fitness < param_.min_model_fitness_lower_bound_ );

    return rm.isRejected();
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::initialize()
{
    global_hypotheses_.clear();
    downsampleSceneCloud();

#pragma omp parallel sections
    {
#pragma omp section
        {
            pcl::ScopeTime t("Computing octree");
            octree_scene_downsampled_.reset(new pcl::octree::OctreePointCloudSearch<SceneT>( param_.resolution_mm_ / 1000.0));
            octree_scene_downsampled_->setInputCloud(scene_cloud_downsampled_);
            octree_scene_downsampled_->addPointsFromInputCloud();
        }

#pragma omp section
        {
            pcl::ScopeTime ("Computing kd-tree");
            kdtree_scene_.reset( new pcl::search::KdTree<SceneT>);
            kdtree_scene_->setInputCloud (scene_cloud_downsampled_);
        }

#pragma omp section
        {
            pcl::ScopeTime t("Computing octrees for model visibility computation");
            for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
            {
                for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                {
                    HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];

                    auto model_octree_it = octree_model_representation_.find( rm.model_id_ );

                    if( model_octree_it == octree_model_representation_.end() )
                    {
                        boost::shared_ptr< pcl::octree::OctreePointCloudPointVector<ModelT> > octree(new pcl::octree::OctreePointCloudPointVector<ModelT>(0.015f) );
                        octree->setInputCloud( rm.complete_cloud_ );
                        octree->addPointsFromInputCloud();
                        octree_model_representation_ [ rm.model_id_ ] = octree;
                    }
                }
            }
        }
    }

    if(occlusion_clouds_.empty()) // we can treat single-view as multi-view case with just one view
    {
        if( scene_cloud_->isOrganized() )
            occlusion_clouds_.push_back(scene_cloud_);
        else
        {
            ZBuffering<SceneT> zbuf (cam_);
            typename pcl::PointCloud<SceneT>::Ptr organized_cloud (new pcl::PointCloud<SceneT>);
            zbuf.renderPointCloud( *scene_cloud_, *organized_cloud );
            occlusion_clouds_.push_back( organized_cloud );
        }

        absolute_camera_poses_.push_back( Eigen::Matrix4f::Identity() );
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            {
                pcl::ScopeTime t("Computing visible model points (1st run)");
#pragma omp parallel for schedule(dynamic)
                for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
                {
                    for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                    {
                        HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];
                        rm.image_mask_.resize(occlusion_clouds_.size(), boost::dynamic_bitset<> (occlusion_clouds_[0]->points.size(), 0) );

                        rm.visible_cloud_.reset( new pcl::PointCloud<ModelT> );
                        rm.image_mask_.resize(occlusion_clouds_.size(), boost::dynamic_bitset<> (occlusion_clouds_[0]->points.size(), 0) );
                        computeModelOcclusionByScene(rm);  //occlusion reasoning based on self-occlusion and occlusion from scene cloud(s)
                    }
                }
            }

            if( param_.icp_iterations_ )
            {
                {
                    std::stringstream info_txt; info_txt << "Pose refinement (" << param_.icp_iterations_ << " ICP iterations)";
                    pcl::ScopeTime t(info_txt.str().c_str());
#pragma omp parallel for schedule(dynamic)
                    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
                    {
                        for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                        {
                            HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];

                            rm.refined_pose_ = refinePose(rm);
                        }
                    }
                    (void)t;
                }

                {
                    pcl::ScopeTime t("Computing visible model points (2nd run)");
#pragma omp parallel for schedule(dynamic)
                    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
                    {
                        for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                        {
                            HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];

                            pcl::transformPointCloud(*rm.complete_cloud_, *rm.complete_cloud_, rm.refined_pose_);
                            transformNormals(*rm.complete_cloud_normals_, *rm.complete_cloud_normals_, rm.refined_pose_);
                            computeModelOcclusionByScene(rm);  //occlusion reasoning based on self-occlusion and occlusion from scene cloud(s)
                        }
                    }
                    (void)t;
                }
            }
            {
                // just mask out the visible normals as well
                for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
                {
                    for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                    {
                        HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];

                        rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal>);
                        pcl::copyPointCloud(*rm.complete_cloud_normals_, rm.visible_indices_, *rm.visible_cloud_normals_);
                    }
                }
            }

            {
                pcl::ScopeTime t("Computing 2D silhouette of visible object model"); //used for checking pairwise intersection of objects (relate amount of overlapping pixel of their 2D silhouette)
#pragma omp parallel for schedule(dynamic)
                for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
                {
                    for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                    {
                        HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];

                        rm.processSilhouette(param_.do_smoothing_, param_.smoothing_radius_, param_.do_erosion_, param_.erosion_radius_, cam_->getWidth());
                    }
                }
                (void)t;
            }

            {
                pcl::ScopeTime t("Computing visible octree nodes");
#pragma omp parallel for schedule(dynamic)
                for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
                {
                    for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                    {
                        HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];
                        computeVisibleOctreeNodes(rm);
                    }
                }
                (void)t;
            }

            removeModelsWithLowVisibility();
        }

#pragma omp section
        {
            if(param_.check_smooth_clusters_)
                extractEuclideanClustersSmooth();
        }

#pragma omp section
        if(!param_.ignore_color_even_if_exists_)
        {
            pcl::ScopeTime t("Converting scene color values");
            colorTransf_->convert(*scene_cloud_downsampled_, scene_color_channels_);
            scene_color_channels_.col(0) = (scene_color_channels_.col(0) - Eigen::VectorXf::Ones(scene_color_channels_.rows())*50.f) / 100.f;
            scene_color_channels_.col(1) = scene_color_channels_.col(0) / 120.f;
            scene_color_channels_.col(2) = scene_color_channels_.col(0) / 120.f;
        }
    }


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
                    {
                        colorTransf_->convert (*rm.visible_cloud_, rm.pt_color_);
                        rm.pt_color_.col(0) = (rm.pt_color_.col(0) - Eigen::VectorXf::Ones(rm.pt_color_.rows())*50.f) / 100.f;
                        rm.pt_color_.col(1) = rm.pt_color_.col(0) / 120.f;
                        rm.pt_color_.col(2) = rm.pt_color_.col(0) / 120.f;
                    }

                }
            }
        }
        (void)t;
    }

    {
        pcl::ScopeTime t("Computing model to scene fitness");
#pragma omp parallel for schedule(dynamic)
        for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
        {
            for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
            {
                HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];

                if( !rm.isRejected() )
                    computeModelFitness(rm);
            }
        }
        (void)t;
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
                    computeLOffset(rm);
            }
        }
        (void)t;
    }


    global_hypotheses_.resize( obj_hypotheses_groups_.size() );

    // do non-maxima surpression on all hypotheses in a hypotheses group based on model fitness
    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
    {
        std::vector<typename HVRecognitionModel<ModelT>::Ptr > ohg = obj_hypotheses_groups_[i];

        std::sort(ohg.begin(), ohg.end(), HVRecognitionModel<ModelT>::modelFitCompare);
        global_hypotheses_[i] = ohg[0];

        for(size_t jj=1; jj<ohg.size(); jj++)
            ohg[jj]->rejected_due_to_better_hypothesis_in_group_ = true;
    }

    if(param_.visualize_model_cues_)
    {
        for(size_t i=0; i<global_hypotheses_.size(); i++)
        {
            HVRecognitionModel<ModelT> &rm = *global_hypotheses_[i];
            visualizeGOCuesForModel(rm);
        }
    }

    size_t kept_hypotheses = 0;
    for(size_t i=0; i<global_hypotheses_.size(); i++)
    {
        typename HVRecognitionModel<ModelT>::Ptr rm = global_hypotheses_[i];

        if ( !rm->isRejected() && !individualRejection(*rm) )
            global_hypotheses_[kept_hypotheses++] = global_hypotheses_[i];
    }

    obj_hypotheses_groups_.clear(); // free space

    if( !kept_hypotheses )
        return;

    global_hypotheses_.resize( kept_hypotheses );

    scene_pts_explained_vec_.resize( scene_cloud_downsampled_->points.size() );

    for(size_t i=0; i<global_hypotheses_.size(); i++)
    {
        const typename HVRecognitionModel<ModelT>::Ptr rm = global_hypotheses_[i];
        for (Eigen::SparseVector<float>::InnerIterator it(rm->scene_explained_weight_); it; ++it)
            scene_pts_explained_vec_[ it.row() ].push_back( PtFitness(it.value(), i) );
    }

    for(auto spt_it = scene_pts_explained_vec_.begin(); spt_it!=scene_pts_explained_vec_.end(); ++spt_it)
        std::sort(spt_it->begin(), spt_it->end());

    {
        pcl::ScopeTime t("Computing pairwise intersection");
        computePairwiseIntersection();
        (void)t;
    }


    if(param_.visualize_pairwise_cues_)
        visualizePairwiseIntersection();
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::optimize ()
{
    solution_ = boost::dynamic_bitset<>(global_hypotheses_.size(), 0);

    if(param_.initial_status_)
        solution_.set();

    //store model fitness into vector
    model_fitness_v_ = Eigen::VectorXf( global_hypotheses_.size() );
    for (size_t i = 0; i < global_hypotheses_.size(); i++)
        model_fitness_v_[i] = global_hypotheses_[i]->model_fit_;


    GHVSAModel<ModelT, SceneT> model;
    double initial_cost  = 0.;
    model.cost_ = static_cast<mets::gol_type> ( initial_cost );
    model.setSolution( solution_ );
    model.setOptimizer (this);

    GHVSAModel<ModelT, SceneT> *best = new GHVSAModel<ModelT, SceneT> (model);
    GHVmove_manager<ModelT, SceneT> neigh ( param_.use_replace_moves_ );
    neigh.setIntersectionCost(intersection_cost_);

    //mets::best_ever_solution best_recorder (best);
    cost_logger_.reset(new GHVCostFunctionLogger<ModelT, SceneT>(*best));
    mets::noimprove_termination_criteria noimprove (param_.max_iterations_);

    if(param_.visualize_go_cues_)
        cost_logger_->setVisualizeFunction(visualize_cues_during_logger_);

    switch( param_.opt_type_ )
    {
    case HV_OptimizationType::LocalSearch:
    {
        pcl::ScopeTime t ("local search...");
        neigh.UseReplaceMoves(false);
        mets::local_search<GHVmove_manager<ModelT, SceneT> > local ( model, *(cost_logger_.get()), neigh, 0, false);
        local.search ();
        (void)t;
        break;
    }
    case HV_OptimizationType::TabuSearch:
    {
        pcl::ScopeTime t ("TABU search...");
        mets::simple_tabu_list tabu_list ( 5 * global_hypotheses_.size()) ;  // ( initial_solution.size() * sqrt ( 1.0*initial_solution.size() ) ) ;
        mets::best_ever_criteria aspiration_criteria ;

        std::cout << "max iterations:" << param_.max_iterations_ << std::endl;
        mets::tabu_search<GHVmove_manager<ModelT, SceneT> > tabu_search(model,  *(cost_logger_.get()), neigh, tabu_list, aspiration_criteria, noimprove);
        //mets::tabu_search<move_manager> tabu_search(model, best_recorder, neigh, tabu_list, aspiration_criteria, noimprove);
        try { tabu_search.search (); }
        catch (mets::no_moves_error e) { }
        (void)t;
        break;
    }
    case HV_OptimizationType::TabuSearchWithLSRM:
    {
        pcl::ScopeTime t("TABU search + LS (RM)...");
        GHVmove_manager<ModelT, SceneT> neigh4 ( false);
        neigh4.setIntersectionCost(intersection_cost_);

        mets::simple_tabu_list tabu_list ( global_hypotheses_.size() * sqrt ( 1.0*global_hypotheses_.size() ) ) ;
        mets::best_ever_criteria aspiration_criteria ;
        mets::tabu_search<GHVmove_manager<ModelT, SceneT> > tabu_search(model,  *(cost_logger_.get()), neigh4, tabu_list, aspiration_criteria, noimprove);
        //mets::tabu_search<move_manager> tabu_search(model, best_recorder, neigh, tabu_list, aspiration_criteria, noimprove);

        try { tabu_search.search (); }
        catch (mets::no_moves_error e) { }

        //after TS, we do LS with RM
        GHVmove_manager<ModelT, SceneT> neigh4RM ( true);
        neigh4RM.setIntersectionCost(intersection_cost_);
        mets::local_search<GHVmove_manager<ModelT, SceneT> > local ( model, *(cost_logger_.get()), neigh4RM, 0, false);
        local.search ();
        (void)t;
        break;

    }
    case HV_OptimizationType::SimulatedAnnealing:
    {
        pcl::ScopeTime t ("SA search...");
        //Simulated Annealing
        //mets::linear_cooling linear_cooling;
        mets::exponential_cooling linear_cooling;
        mets::simulated_annealing<GHVmove_manager<ModelT, SceneT> > sa (model,  *(cost_logger_.get()), neigh, noimprove, linear_cooling, initial_temp_, 1e-7, 1);
        sa.setApplyAndEvaluate (true);
        sa.search ();
        (void)t;
        break;
    }
    default:
        throw std::runtime_error("Specified optimization type not implememted!");
    }

    best_seen_ = static_cast<const GHVSAModel<ModelT, SceneT>&> (cost_logger_->best_seen());
    std::cout << "*****************************" << std::endl
              << "Solution: " << solution_ << std::endl
              << "Final cost: " << best_seen_.cost_ << std::endl
              << "Number of evaluations: " << cost_logger_->getTimesEvaluated() << std::endl
              << "Number of accepted moves: " << cost_logger_->getAcceptedMovesSize() << std::endl
              << "*****************************" << std::endl;

    delete best;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::verify()
{
    {
        pcl::ScopeTime t("initialization");
        initialize();
        (void)t;
    }

    if(param_.visualize_go_cues_)
        visualize_cues_during_logger_ = boost::bind(&HypothesisVerification<ModelT, SceneT>::visualizeGOCues, this, _1, _2, _3);

    {
        pcl::ScopeTime t("Optimizing object hypotheses verification cost function");
        optimize ();
    }

    cleanUp();
}


template<typename ModelT, typename SceneT>
bool
HypothesisVerification<ModelT, SceneT>::removeNanNormals (HVRecognitionModel<ModelT> &rm) const
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
HypothesisVerification<ModelT, SceneT>::computeModelFitness(HVRecognitionModel<ModelT> &rm) const
{
    rm.model_scene_c_.resize( rm.visible_cloud_->points.size () * param_.knn_inliers_ );
    size_t kept=0;

//    pcl::visualization::PCLVisualizer vis;
//    int vp1, vp2;
//    vis.createViewPort(0,0,0.5,1,vp1);
//    vis.createViewPort(0.5,0,1,1,vp2);
//    vis.addPointCloud(rm.visible_cloud_, "vis_cloud", vp1);
//    pcl::visualization::PointCloudColorHandlerCustom<SceneT> gray (scene_cloud_downsampled_, 128, 128, 128);
//    vis.addPointCloud(scene_cloud_downsampled_, gray, "scene1", vp1);
//    vis.setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "scene1");


    for (size_t midx = 0; midx < rm.visible_cloud_->points.size (); midx++)
    {
        std::vector<int> nn_indices;
        std::vector<float> nn_sqr_distances;
        octree_scene_downsampled_->nearestKSearch(rm.visible_cloud_->points[midx], param_.knn_inliers_, nn_indices, nn_sqr_distances);

//        vis.addSphere(rm.visible_cloud_->points[midx], 0.005, 0., 1., 0., "queryPoint", vp1 );

        Eigen::Vector3f normal_m = rm.visible_cloud_normals_->points[midx].getNormalVector3fMap();
        normal_m.normalize();

        for (size_t k = 0; k < nn_indices.size(); k++)
        {
            float sqr_3D_dist = nn_sqr_distances[k];
            int sidx = nn_indices[ k ];

//            std::stringstream pt_id; pt_id << "searched_pt_" << k;
//            vis.addSphere(scene_cloud_downsampled_->points[sidx], 0.005, 1., 0., 0., pt_id.str(), vp2 );
//            vis.addPointCloud(rm.visible_cloud_, "vis_cloud2", vp2);
//            vis.addPointCloud(scene_cloud_downsampled_, gray, "scene2", vp2);
//            vis.setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "scene2");

            if (sqr_3D_dist > ( 1.5f * 1.5f * param_.inliers_threshold_ * param_.inliers_threshold_ ) ) ///TODO: consider camera's noise level
                continue;


            ModelSceneCorrespondence &c = rm.model_scene_c_[ kept ];
            c.model_id_ = midx;
            c.scene_id_ = sidx;
            c.dist_3D_ = sqrt(sqr_3D_dist);

            Eigen::Vector3f normal_s = scene_normals_downsampled_->points[sidx].getNormalVector3fMap();
            normal_s.normalize();

            float dotp = std::min( 0.99999f, std::max(-0.99999f, normal_m.dot(normal_s) ) );
            c.angle_surface_normals_rad_ = acos (dotp);

            CHECK (c.angle_surface_normals_rad_ <= M_PI);
            CHECK (c.angle_surface_normals_rad_ >= 0.f );

            const Eigen::VectorXf &color_m = rm.pt_color_.row( midx );
            const Eigen::VectorXf &color_s = scene_color_channels_.row( sidx );
            c.color_distance_ = color_dist_f_(color_s, color_m);
            CHECK (c.color_distance_ >= 0.f);

            c.fitness_ = getFitness( c );
            kept++;
        }
//        vis.removeAllShapes(vp1);
    }

//    vis.spin();
    rm.model_scene_c_.resize(kept);

    std::sort( rm.model_scene_c_.begin(), rm.model_scene_c_.end() );

    rm.scene_explained_weight_ = Eigen::SparseVector<float> (scene_cloud_downsampled_->points.size());
    rm.scene_explained_weight_.reserve( rm.model_scene_c_.size() );
    Eigen::VectorXf modelFit   = Eigen::VectorXf::Zero (rm.visible_cloud_->points.size());

    boost::dynamic_bitset<> scene_explained_pts ( scene_cloud_downsampled_->points.size(), 0);
    boost::dynamic_bitset<> model_explained_pts ( rm.visible_cloud_->points.size(), 0);

    rm.explained_pts_per_smooth_cluster_.clear();
    rm.explained_pts_per_smooth_cluster_.resize(smooth_label_count_.size(), 0);

    for( const ModelSceneCorrespondence &c : rm.model_scene_c_ )
    {
        int sidx = c.scene_id_;
        int midx = c.model_id_;

        if( !scene_explained_pts[sidx] )
        {
            scene_explained_pts.set(sidx);
            rm.scene_explained_weight_.insert(sidx) = c.fitness_ ;

            if( param_.check_smooth_clusters_ )
            {
                int l = scene_smooth_labels_[sidx];
                rm.explained_pts_per_smooth_cluster_[l] ++;
            }
        }

        if( !model_explained_pts[midx] )
        {
            model_explained_pts.set(midx);
            modelFit(midx) = c.fitness_;
        }
    }

    rm.model_fit_ = modelFit.sum();
}

//######### VISUALIZATION FUNCTIONS #####################
//template<>
//void
//HypothesisVerification<pcl::PointXYZ, pcl::PointXYZ>::visualizeGOCuesForModel(const HVRecognitionModel<pcl::PointXYZ> &rm) const
//{
//    (void)rm;
//    std::cerr << "The visualization function is not defined for the chosen Point Cloud Type!" << std::endl;
//}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::visualizeGOCuesForModel(const HVRecognitionModel<ModelT> &rm) const
{
    if(!rm_vis_) {
        rm_vis_.reset (new pcl::visualization::PCLVisualizer ("model cues"));
        rm_vis_->createViewPort(0   , 0  , 0.2, 0.33, rm_vp_scene_);
        rm_vis_->createViewPort(0.2 , 0  , 0.4, 0.33, rm_vp_model_);
        rm_vis_->createViewPort(0.4 , 0  , 0.6, 0.33, rm_vp_scene_and_model_);
        rm_vis_->createViewPort(0.6 , 0  , 0.8, 0.33, rm_vp_smooth_labels_);
        rm_vis_->createViewPort(0.8 , 0  , 1  , 0.33, rm_vp_scene_fitness_);

        rm_vis_->createViewPort(0. , 0.33 , 0.2 ,0.66 , rm_vp_visible_model_);
        rm_vis_->createViewPort(0.2, 0.33 , 0.4 ,0.66 , rm_vp_model_scene_model_fit_);
        rm_vis_->createViewPort(0.4, 0.33 , 0.6 ,0.66 , rm_vp_model_scene_3d_dist_);
        rm_vis_->createViewPort(0.6, 0.33 , 0.8 ,0.66 , rm_vp_model_scene_color_dist_);
        rm_vis_->createViewPort(0.8, 0.33 , 1   ,0.66 , rm_vp_model_scene_normals_dist_);

        rm_vis_->setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2]);
    }

    rm_vis_->removeAllPointClouds();
    rm_vis_->removeAllShapes();

    rm_vis_->addPointCloud(scene_cloud_downsampled_, "scene1", rm_vp_scene_);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "scene1", rm_vp_scene_);

    pcl::visualization::PointCloudColorHandlerCustom<SceneT> gray (scene_cloud_downsampled_, 128, 128, 128);
    rm_vis_->addPointCloud(rm.visible_cloud_, "model", rm_vp_model_);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "model", rm_vp_model_);
    rm_vis_->addPointCloud(scene_cloud_downsampled_, gray, "input_rm_vp_model_", rm_vp_model_);
    rm_vis_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_rm_vp_model_");


    if(!vis_param_->no_text_)
    {
        rm_vis_->addText("scene",10,10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1] ,vis_param_->text_color_[2], "scene",rm_vp_scene_);
        rm_vis_->addText("model",10,10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1] ,vis_param_->text_color_[2], "model",rm_vp_model_);
    }

    if(!vis_param_->no_text_)
    {
        std::stringstream txt; txt << "visible ratio: " << std::fixed << std::setprecision(2) << rm.visible_indices_by_octree_.size() / (float)rm.complete_cloud_->points.size();
        rm_vis_->addText(txt.str(), 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1] ,vis_param_->text_color_[2], "visible model cloud", rm_vp_visible_model_);
    }

    typename pcl::PointCloud<ModelT>::Ptr visible_cloud_colored (new pcl::PointCloud<ModelT> (*rm.complete_cloud_));

    for(ModelT &mp : visible_cloud_colored->points)
        mp.r = mp.g = mp.b = 0.f;

    for(int idx : rm.visible_indices_by_octree_)
        visible_cloud_colored->points[idx].r = 255;

    rm_vis_->addPointCloud(visible_cloud_colored, "model2", rm_vp_visible_model_);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "model2", rm_vp_model_);

    typename pcl::PointCloud<ModelT>::Ptr model_3D_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
    typename pcl::PointCloud<ModelT>::Ptr model_color_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
    typename pcl::PointCloud<ModelT>::Ptr model_normals_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
    typename pcl::PointCloud<ModelT>::Ptr model_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
    for(size_t p=0; p < model_3D_fit_cloud->points.size(); p++)
    {
        ModelT &mp3d = model_3D_fit_cloud->points[p];
        ModelT &mpC = model_color_fit_cloud->points[p];
        ModelT &mpN = model_normals_fit_cloud->points[p];
        ModelT &mp = model_fit_cloud->points[p];
        mp3d.r = mp3d.b = mpC.r = mpC.b = mpN.r = mpN.b = mp.r = mp.b = 0.f;
        mp3d.g = mpC.g = mpN.g = mp.g = 0.f;
    }

    Eigen::VectorXf normals_fitness = Eigen::VectorXf::Zero (rm.visible_cloud_->points.size());
    Eigen::VectorXf color_fitness = Eigen::VectorXf::Zero (rm.visible_cloud_->points.size());
    Eigen::VectorXf fitness_3d = Eigen::VectorXf::Zero (rm.visible_cloud_->points.size());

    for(size_t cidx=0; cidx < rm.model_scene_c_.size(); cidx++)
    {
        const ModelSceneCorrespondence &c = rm.model_scene_c_[cidx];
        int sidx = c.scene_id_;
        int midx = c.model_id_;

        if(sidx<0)
            continue;

        normals_fitness(midx) = modelSceneNormalsCostTerm(c);
        color_fitness(midx) = modelSceneColorCostTerm(c);
        fitness_3d(midx) =  modelScene3DDistCostTerm(c);

        CHECK ( normals_fitness(midx) <= 1 );
        CHECK ( color_fitness(midx) <= 1 );
        CHECK ( fitness_3d(midx) <= 1 );
        CHECK ( getFitness(c) <= 1 );

        ModelT &mp3d = model_3D_fit_cloud->points[midx];
        ModelT &mpC = model_color_fit_cloud->points[midx];
        ModelT &mpN = model_normals_fit_cloud->points[midx];
        ModelT &mp = model_fit_cloud->points[midx];

        // scale green color channels with fitness terms
        mp3d.g = 255.f * fitness_3d(midx);
        mpC.g  = 255.f * color_fitness(midx);
        mpN.g  = 255.f * normals_fitness(midx);
        mp.g   = 255.f * getFitness(c);
    }

    if(!vis_param_->no_text_)
    {
        std::stringstream txt;
        txt.str(""); txt << std::fixed << std::setprecision(2)  << "3D fitness (" << (int)(param_.w_3D_*100) << "\%): " << (float)fitness_3d.sum() / rm.visible_indices_.size();
        rm_vis_->addText(txt.str(),10,10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "3D distance",rm_vp_model_scene_3d_dist_);
        txt.str(""); txt << "color fitness(" << (int)(param_.w_color_ *100) << "\%): " << std::fixed << std::setprecision(2) << (float)color_fitness.sum() / rm.visible_indices_.size();
        rm_vis_->addText(txt.str(),10,10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "color distance",rm_vp_model_scene_color_dist_);
        txt.str(""); txt << "normals fitness(" << (int)(param_.w_normals_*100) << "\%): " << std::fixed << std::setprecision(2) << (float)normals_fitness.sum() / rm.visible_indices_.size();
        rm_vis_->addText(txt.str(), 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2],  "normals distance",rm_vp_model_scene_normals_dist_);
        txt.str(""); txt << "model fitness: " << std::fixed << std::setprecision(2) << rm.model_fit_ << "; normalized: " << rm.model_fit_ / rm.visible_cloud_->points.size();
        rm_vis_->addText(txt.str(),10,10, vis_param_->fontsize_,vis_param_->text_color_[0], vis_param_->text_color_[1] ,vis_param_->text_color_[2], "model fitness",rm_vp_model_scene_model_fit_);
    }

    rm_vis_->addPointCloud(model_3D_fit_cloud, "3D_distance", rm_vp_model_scene_3d_dist_);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "3D_distance", rm_vp_model_scene_3d_dist_);
    rm_vis_->addPointCloud(scene_cloud_downsampled_, gray, "input_rm_vp_model_scene_3d_dist_", rm_vp_model_scene_3d_dist_);
    rm_vis_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_rm_vp_model_scene_3d_dist_");

    rm_vis_->addPointCloud(model_color_fit_cloud, "color_distance", rm_vp_model_scene_color_dist_);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "color_distance", rm_vp_model_scene_color_dist_);
    rm_vis_->addPointCloud(scene_cloud_downsampled_, gray, "input_rm_vp_model_scene_color_dist_", rm_vp_model_scene_color_dist_);
    rm_vis_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_rm_vp_model_scene_color_dist_");

    rm_vis_->addPointCloud(model_normals_fit_cloud, "normals_distance", rm_vp_model_scene_normals_dist_);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "normals_distance", rm_vp_model_scene_normals_dist_);
    rm_vis_->addPointCloud(scene_cloud_downsampled_, gray, "input_rm_vp_model_scene_normals_dist_", rm_vp_model_scene_normals_dist_);
    rm_vis_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_rm_vp_model_scene_normals_dist_");

    rm_vis_->addPointCloud(model_color_fit_cloud, "model_fitness", rm_vp_model_scene_model_fit_);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "model_fitness", rm_vp_model_scene_model_fit_);
    rm_vis_->addPointCloud(scene_cloud_downsampled_, gray, "input_rm_vp_model_scene_model_fit_", rm_vp_model_scene_model_fit_);
    rm_vis_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_rm_vp_model_scene_model_fit_");


    // ---- VISUALIZE SMOOTH SEGMENTATION -------
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_smooth_labels_rgb (new pcl::PointCloud<pcl::PointXYZRGB>(*scene_cloud_downsampled_));
    if(!smooth_label_count_.empty())
    {
        Eigen::Matrix3Xf label_colors (3, smooth_label_count_.size());
        for(size_t i=0; i<smooth_label_count_.size(); i++)
        {
            float r,g,b;
            if( i==0 )
                r = g = b = 255; // label 0 will be white
            else
            {
                r = rand () % 255;
                g = rand () % 255;
                b = rand () % 255;
            }
            label_colors(0,i) = r;
            label_colors(1,i) = g;
            label_colors(2,i) = b;

            if(!vis_param_->no_text_)
            {
                std::stringstream lbl_txt; lbl_txt << std::fixed << std::setprecision(2) << rm.explained_pts_per_smooth_cluster_[i] << " / " << smooth_label_count_[i];
                std::stringstream txt_id; txt_id << "smooth_cluster_txt " << i;
                rm_vis_->addText( lbl_txt.str(), 10, 10+12*i, vis_param_->fontsize_, r/255, g/255, b/255, txt_id.str(), rm_vp_smooth_labels_);
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
        rm_vis_->addPointCloud(scene_smooth_labels_rgb, "smooth labels", rm_vp_smooth_labels_);
    }
    //---- END VISUALIZE SMOOTH SEGMENTATION-----------


    typename pcl::PointCloud<SceneT>::Ptr scene_fit_cloud (new pcl::PointCloud<SceneT> (*scene_cloud_downsampled_));

    for(int p=0; p < rm.scene_explained_weight_.rows(); p++)
    {
        SceneT &sp = scene_fit_cloud->points[p];
        sp.r = sp.b = 0.f;
        sp.g = 255.f * rm.scene_explained_weight_.coeff(p);
    }

    if(!vis_param_->no_text_)
    {
        std::stringstream txt;
        txt.str(""); txt << "scene pts explained (fitness: " << rm.scene_explained_weight_.sum() << "; normalized: " << rm.scene_explained_weight_.sum()/scene_cloud_downsampled_->points.size() << ")";
        rm_vis_->addText(txt.str(),10,10, vis_param_->fontsize_,0,0,0,"scene fitness",rm_vp_scene_fitness_);
        rm_vis_->addText("scene and visible model",10,10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "scene_and_model",rm_vp_scene_and_model_);
        rm_vis_->addPointCloud(scene_cloud_downsampled_, "scene_model_1", rm_vp_scene_and_model_);
    }

    rm_vis_->addPointCloud(scene_fit_cloud, "scene_fitness", rm_vp_scene_fitness_);
    rm_vis_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "scene_fitness", rm_vp_scene_fitness_);

    rm_vis_->addPointCloud(rm.visible_cloud_, "scene_model_2", rm_vp_scene_and_model_);
    rm_vis_->addPointCloud(rm.visible_cloud_, "scene_model_4", rm_vp_smooth_labels_);

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
        vis_pairwise_->createViewPort(0    , 0   , 0.25 , 1   , vp_pair_1_);
        vis_pairwise_->createViewPort(0.25 , 0   , 0.5  , 1   , vp_pair_2_);
        vis_pairwise_->createViewPort(0.5  , 0   , 1    , 1   , vp_pair_3_);
        vis_pairwise_->setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2]);
    }

    for(size_t i=1; i<global_hypotheses_.size(); i++)
    {
        const HVRecognitionModel<ModelT> &rm_a = *global_hypotheses_[i];

        for(size_t j=0; j<i; j++)
        {
            const HVRecognitionModel<ModelT> &rm_b = *global_hypotheses_[j];

            std::stringstream txt; txt <<  "intersection cost (" << i << ", " << j << "): " << intersection_cost_(j,i);

            vis_pairwise_->removeAllPointClouds();
            vis_pairwise_->removeAllShapes();
            vis_pairwise_->addText(txt.str(), 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2],  "intersection_text", vp_pair_3_ );
            vis_pairwise_->addPointCloud(rm_a.visible_cloud_, "cloud_a", vp_pair_1_);
            vis_pairwise_->addPointCloud(rm_b.visible_cloud_, "cloud_b", vp_pair_2_);
            vis_pairwise_->addPointCloud(rm_a.visible_cloud_, "cloud_aa", vp_pair_3_);
            vis_pairwise_->addPointCloud(rm_b.visible_cloud_, "cloud_bb", vp_pair_3_);
            vis_pairwise_->resetCamera();
            vis_pairwise_->spin();
        }
    }
}


template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::visualizeGOCues (const boost::dynamic_bitset<> & active_solution, float cost, int times_evaluated) const
{
    if(!vis_go_cues_) {
        vis_go_cues_.reset(new pcl::visualization::PCLVisualizer("visualizeGOCues"));
        vis_go_cues_->createViewPort(0, 0, 0.33, 0.5, vp_scene_);
        vis_go_cues_->createViewPort(0.33, 0, 0.66, 0.5, vp_active_hypotheses_);
        vis_go_cues_->createViewPort(0.66, 0, 1, 0.5, vp_model_scene_3D_dist_);
        vis_go_cues_->createViewPort(0, 0.5, 0.33, 1, vp_scene_fitness_);
        vis_go_cues_->setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2]);
    }

    vis_go_cues_->removeAllPointClouds();
    vis_go_cues_->removeAllShapes();

    float model_fitness = 0.f, pairwise_cost = 0.f, scene_fitness = 0.f;

    // model uni term
    size_t num_active_hypotheses = 0;
    for(size_t i=0; i<active_solution.size(); i++)
    {
        if(active_solution[i])
        {
            model_fitness += global_hypotheses_[i]->model_fit_;
            num_active_hypotheses++;
        }
    }
    if(!num_active_hypotheses)
        model_fitness = 0.f;


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
    out << "Active Hypotheses: " << active_solution << std::endl
        << "Cost: " << std::setprecision(5) << cost << " , #Evaluations: " << times_evaluated
        << std::endl << "; pairwise cost: " << pairwise_cost << "; total cost: " << cost << std::endl;
    model_fitness_txt << "model fitness: " << model_fitness;
    scene_fitness_txt << "scene fitness: " << scene_fitness;


    vis_go_cues_->addText ("Scene", 1, 30, 16, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "inliers_outliers", vp_scene_);
    vis_go_cues_->addText (out.str(), 1, 30, 16, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "scene_cues", vp_active_hypotheses_);
    vis_go_cues_->addText (model_fitness_txt.str(), 1, 30, 16, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "model fitness", vp_model_scene_3D_dist_);
    vis_go_cues_->addText (scene_fitness_txt.str(), 1, 30, 16, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "scene fitness", vp_scene_fitness_);
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

                const ModelSceneCorrespondence &c = rm.model_scene_c_[p];
                mp.b = 50.f + 205.f * c.dist_3D_;
            }

            model_name << "_fitness";
            vis_go_cues_->addPointCloud(model_fit_cloud, model_name.str(), vp_model_scene_3D_dist_);
        }
    }


    typename pcl::PointCloud<SceneT>::Ptr scene_fit_cloud (new pcl::PointCloud<SceneT> (*scene_cloud_downsampled_));
    for( SceneT &sp : scene_fit_cloud->points)
        sp.r = sp.g = sp.b = 0.f;

    double sum_explained = 0.;
    for(size_t i=0; i<scene_pts_explained_vec_.size(); i++)
    {
        for(const auto &pt_fit:scene_pts_explained_vec_[i])
        {
            if( active_solution[pt_fit.rm_id_] )
            {
                sum_explained += pt_fit.fit_;
                scene_fit_cloud->points[i].g = 255.f * pt_fit.fit_;
                break;
            }
        }
    }

    vis_go_cues_->addPointCloud(scene_fit_cloud, "scene fitness", vp_scene_fitness_);
    vis_go_cues_->resetCamera();
    vis_go_cues_->spin();
}

#define PCL_INSTANTIATE_HypothesisVerification(ModelT, SceneT) template class V4R_EXPORTS HypothesisVerification<ModelT, SceneT>;
PCL_INSTANTIATE_PRODUCT(HypothesisVerification, ((pcl::PointXYZRGB))((pcl::PointXYZRGB)) )

}
