#include <v4r/common/histogram.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/normals.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/occlusion_reasoning.h>
#include <v4r/common/pcl_utils.h>
#include <v4r/common/zbuffering.h>
#include <v4r/recognition/hypotheses_verification.h>

#include <Eigen/Sparse>
#include <pcl_1_8/keypoints/uniform_sampling.h>
#include <pcl/common/angles.h>
#include <pcl/common/common.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_1_8/segmentation/conditional_euclidean_clustering.h>

#include <fstream>
#include <omp.h>
#include <numeric>

#include <opencv2/opencv.hpp>
//#include <opencv2/highgui.hpp>

namespace v4r
{

template<typename ModelT, typename SceneT>
bool
HypothesisVerification<ModelT, SceneT>::customRegionGrowing (const SceneTWithNormal& seed_pt, const SceneTWithNormal& candidate_pt, float squared_distance) const
{
    float curvature_threshold = param_.curvature_threshold_ ;
    float radius = param_.cluster_tolerance_;
    float eps_angle_threshold_rad = pcl::deg2rad(param_.eps_angle_threshold_deg_);

    if ( param_.z_adaptive_ )
    {
        float mult = std::max(seed_pt.z, 1.f);
//            mult *= mult;
        radius = param_.cluster_tolerance_ * mult;
        curvature_threshold = param_.curvature_threshold_ * mult;
        eps_angle_threshold_rad = eps_angle_threshold_rad * mult;
    }

    if( seed_pt.curvature > param_.curvature_threshold_)
        return false;

    if(candidate_pt.curvature > param_.curvature_threshold_)
        return false;

    if (squared_distance > radius * radius)
        return false;

    float intensity_a = .2126f * seed_pt.r + .7152f * seed_pt.g + .0722f * seed_pt.b;
    float intensity_b = .2126f * candidate_pt.r + .7152f * candidate_pt.g + .0722f * candidate_pt.b;

    if( fabs(intensity_a - intensity_b) > 5.f)
        return false;

    float dotp = seed_pt.getNormalVector3fMap().dot (candidate_pt.getNormalVector3fMap() );
    if ( dotp < cos(eps_angle_threshold_rad))
        return false;

    return true;
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::computeModelOcclusionByScene(HVRecognitionModel<ModelT> &rm) const
{
    bool found_model_foo;
    typename Model<ModelT>::ConstPtr m = m_db_->getModelById("", rm.oh_->model_id_, found_model_foo);
    typename pcl::PointCloud<ModelT>::ConstPtr model_cloud  = m->getAssembled ( -1 );  // use full resolution for rendering
    pcl::PointCloud<pcl::Normal>::ConstPtr model_normals = m->getNormalsAssembled ( -1 );
    CHECK(model_cloud->points.size() == model_normals->points.size());

    const Eigen::Matrix4f hyp_tf_2_global = rm.oh_->pose_refinement_ * rm.oh_->transform_;
    typename pcl::PointCloud<ModelT>::Ptr model_cloud_aligned (new pcl::PointCloud<ModelT>);
    pcl::transformPointCloud(*model_cloud, *model_cloud_aligned, hyp_tf_2_global);
    pcl::PointCloud<pcl::Normal>::Ptr model_normals_aligned (new pcl::PointCloud<pcl::Normal>);
    v4r::transformNormals(*model_normals, *model_normals_aligned, hyp_tf_2_global);

    boost::dynamic_bitset<> image_mask_mv(model_cloud->points.size(), 0);
    rm.image_mask_.resize(occlusion_clouds_.size(), boost::dynamic_bitset<> (occlusion_clouds_[0]->points.size(), 0) );

    for(size_t view=0; view<occlusion_clouds_.size(); view++)
    {
        // project into respective view
        typename pcl::PointCloud<ModelT>::Ptr aligned_cloud (new pcl::PointCloud<ModelT>);
        pcl::PointCloud<pcl::Normal>::Ptr aligned_normals (new pcl::PointCloud<pcl::Normal>);
        const Eigen::Matrix4f tf = absolute_camera_poses_[view].inverse();
        pcl::transformPointCloud(*model_cloud_aligned, *aligned_cloud, tf);
        v4r::transformNormals(*model_normals_aligned, *aligned_normals, tf);

        ZBufferingParameter zBparam;
        zBparam.do_noise_filtering_ = false;
        zBparam.do_smoothing_ = false;
//        zBparam.smoothing_radius_ = 5;
        zBparam.inlier_threshold_ = 0.015f;
        zBparam.use_normals_ = true;
        ZBuffering<ModelT> zbuf (cam_, zBparam);
        zbuf.setCloudNormals( aligned_normals );
        typename pcl::PointCloud<ModelT>::Ptr organized_cloud_to_be_filtered (new pcl::PointCloud<ModelT>);
        zbuf.renderPointCloud( *aligned_cloud, *organized_cloud_to_be_filtered, 2 );
//        std::vector<int> kept_indices = zbuf.getKeptIndices();
        Eigen::MatrixXi index_map = zbuf.getIndexMap();

//        static pcl::visualization::PCLVisualizer vis ("self-occlusion");
//        static int vp1, vp2, vp3, vp4;
//        vis.removeAllPointClouds();
//        vis.createViewPort(0,0,0.25,1,vp1);
//        vis.createViewPort(0.25,0,0.5,1,vp2);
//        vis.createViewPort(0.5,0,0.75,1,vp3);
//        vis.createViewPort(0.75,0,1,1,vp4);
//        vis.addPointCloud(aligned_cloud, "input", vp1);
//        vis.addPointCloud(aligned_cloud, "input3", vp3);
//        vis.addCoordinateSystem(0.04,"co",vp1);
//        vis.addCoordinateSystem(0.04,"co2",vp2);
//        vis.addCoordinateSystem(0.04,"co2",vp3);
//        vis.addPointCloud(organized_cloud_to_be_filtered, "organized", vp2);
//        boost::dynamic_bitset<> image_mask_sv( model_cloud->points.size(), 0 );
//        for (size_t u=0; u<organized_cloud_to_be_filtered->width; u++)
//        {
//            for (size_t v=0; v<organized_cloud_to_be_filtered->height; v++)
//            {
//                int original_idx = index_map(v,u);

//                if(original_idx < 0)
//                    continue;
//                image_mask_sv.set( original_idx );
//            }
//        }

//        typename pcl::PointCloud<SceneT>::Ptr scene_cloud_vis (new pcl::PointCloud<SceneT>(*scene_cloud_downsampled_));
//        scene_cloud_vis->sensor_orientation_ = Eigen::Quaternionf::Identity();
//        scene_cloud_vis->sensor_origin_ = Eigen::Vector4f::Zero(4);
//        pcl::visualization::PointCloudColorHandlerCustom<SceneT> gray (scene_cloud_vis, 128, 128, 128);
//        vis.addPointCloud(scene_cloud_vis, gray, "input_rm_vp_model_", vp1);

//        typename pcl::PointCloud<SceneT>::Ptr occlusion_cloud_view_vis (new pcl::PointCloud<SceneT>(*occlusion_clouds_[view]));
//        occlusion_cloud_view_vis->sensor_orientation_ = Eigen::Quaternionf::Identity();
//        occlusion_cloud_view_vis->sensor_origin_ = Eigen::Vector4f::Zero(4);
//        vis.addPointCloud(occlusion_cloud_view_vis, "occlusion_cloud_view_vis", vp3);
//        typename pcl::PointCloud<ModelT>::Ptr visible_cloud2 (new pcl::PointCloud<ModelT>);
//        pcl::copyPointCloud(*aligned_cloud, image_mask_sv, *visible_cloud2);
//        vis.addPointCloud(visible_cloud2, "vis_cloud2", vp4);

//        vis.spin();

        OcclusionReasoner<SceneT, ModelT> occ_reasoner;
        occ_reasoner.setCamera(cam_);
        occ_reasoner.setInputCloud( organized_cloud_to_be_filtered );
        occ_reasoner.setOcclusionCloud( occlusion_clouds_[view] );
        occ_reasoner.setOcclusionThreshold( param_.occlusion_thres_ );
        boost::dynamic_bitset<> pt_is_visible =  occ_reasoner.computeVisiblePoints();
        rm.image_mask_[view] = occ_reasoner.getPixelMask();

//        {
//            static pcl::visualization::PCLVisualizer vis2 ("occlusion_reasoning");
//            static int vp1, vp2, vp3;
//            vis2.removeAllPointClouds();
//            vis2.createViewPort(0,0,0.33,1,vp1);
//            vis2.createViewPort(0.33,0,0.66,1,vp2);
//            vis2.createViewPort(0.66,0,1,1,vp3);
//            vis2.addPointCloud(organized_cloud_to_be_filtered, "cloud_to_be_filtered", vp1);
//            vis2.addPointCloud(occlusion_cloud_view_vis, "occluder_cloud", vp2);
//            typename pcl::PointCloud<ModelT>::Ptr visible_cloud (new pcl::PointCloud<ModelT>);
//            pcl::copyPointCloud(*organized_cloud_to_be_filtered, pt_is_visible, *visible_cloud);
//            vis2.addPointCloud(visible_cloud, "vis_cloud", vp3);
//            vis2.spin();
//        }

        for (size_t u=0; u<organized_cloud_to_be_filtered->width; u++)
        {
            for (size_t v=0; v<organized_cloud_to_be_filtered->height; v++)
            {
                int idx = v*organized_cloud_to_be_filtered->width + u;

                if( img_boundary_distance_.at<float>(v,u) < param_.min_px_distance_to_image_boundary_ )
                    continue;

                if ( pt_is_visible[idx] )
                {
                    int original_idx = index_map(v,u);

                    if(original_idx < 0)
                        continue;

                    Eigen::Vector3f viewray = aligned_cloud->points[original_idx].getVector3fMap();
                    viewray.normalize();
                    Eigen::Vector3f normal = aligned_normals->points[original_idx].getNormalVector3fMap();
                    normal.normalize();

                    float dotp = viewray.dot(normal);

                    if ( fabs(dotp) < param_.min_dotproduct_model_normal_to_viewray_ )
                        continue;


                    image_mask_mv.set( original_idx );
                }
            }
        }


//        cv::Mat registration_depth_mask = cam_->getCameraDepthRegistrationMask();

//        cv::imshow("reg_mask", registration_depth_mask);
//        cv::waitKey();
    }

    std::vector<int> visible_indices_tmp_full = createIndicesFromMask<int>(image_mask_mv);
    typename pcl::PointCloud<ModelT>::Ptr visible_cloud_full ( new pcl::PointCloud<ModelT> );
    pcl::copyPointCloud (*model_cloud, visible_indices_tmp_full, *visible_cloud_full);

    // downsample
    pcl_1_8::UniformSampling<ModelT> us;
    us.setRadiusSearch( param_.resolution_mm_ / 1000.0 );
    us.setInputCloud( visible_cloud_full );
    pcl::PointCloud<int> sampled_indices;
    us.compute(sampled_indices);

    rm.visible_indices_.resize( sampled_indices.size() );
    for( size_t i=0; i<sampled_indices.size(); i++ )
    {
        int idx = visible_indices_tmp_full[ sampled_indices[i] ];
        rm.visible_indices_[i] = idx;
    }


    rm.visible_cloud_.reset( new pcl::PointCloud<ModelT> );
    rm.visible_cloud_normals_.reset ( new pcl::PointCloud<pcl::Normal> );
    pcl::copyPointCloud(*model_cloud_aligned, rm.visible_indices_, *rm.visible_cloud_);
    pcl::copyPointCloud(*model_normals_aligned, rm.visible_indices_, *rm.visible_cloud_normals_);

//    static pcl::visualization::PCLVisualizer vis;
//    vis.removeAllPointClouds();
//    vis.addPointCloud(rm.visible_cloud_, "scene");
////    vis.addPointCloud(aligned_cloud, "model");
//    vis.spin();
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::computeVisibleOctreeNodes(HVRecognitionModel<ModelT> &rm)
{
    boost::dynamic_bitset<> visible_mask = v4r::createMaskFromIndices( rm.visible_indices_, rm.num_pts_full_model_ );
    auto octree_it = octree_model_representation_.find( rm.oh_->model_id_ );

    if(octree_it == octree_model_representation_.end())
        std::cerr << "Did not find octree representation! This should not happen!" << std::endl;

    boost::dynamic_bitset<> visible_leaf_mask (rm.num_pts_full_model_, 0);

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
void
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
    icp.setTransformationEpsilon (1e-6);
    icp.setMaximumIterations(param_.icp_iterations_);
    icp.setMaxCorrespondenceDistance(param_.icp_max_correspondence_);
    icp.setSearchMethodTarget(kdtree_scene_, true);
    pcl::PointCloud<ModelT> aligned_visible_model;
    icp.align(aligned_visible_model);

    Eigen::Matrix4f pose_refinement;
    if(icp.hasConverged())
    {
        pose_refinement = icp.getFinalTransformation();
        rm.oh_->pose_refinement_ = pose_refinement * rm.oh_->pose_refinement_;
    }
    else
        LOG(WARNING) << "ICP did not converge" << std::endl;

//    static pcl::visualization::PCLVisualizer vis_tmp;
//    static int vp2, vp3,vp1;
//    vis_tmp.removeAllPointClouds();
//    vis_tmp.removeAllShapes();
//    vis_tmp.createViewPort(0,0,0.33,1, vp1);
//    vis_tmp.createViewPort(0.33,0,0.66,1, vp2);
//    vis_tmp.createViewPort(0.66,0,1,1, vp3);

//    typename pcl::PointCloud<SceneT>::Ptr scene_cloud_vis (new pcl::PointCloud<SceneT>(*scene_cloud_downsampled_));
//    scene_cloud_vis->sensor_orientation_ = Eigen::Quaternionf::Identity();
//    scene_cloud_vis->sensor_origin_ = Eigen::Vector4f::Zero(4);

//    vis_tmp.addPointCloud(rm.visible_cloud_, "model1", vp1);
//    pcl::visualization::PointCloudColorHandlerCustom<SceneT> gray (scene_cloud_vis, 128, 128, 128);
//    vis_tmp.addPointCloud(scene_cloud_vis, gray, "scene1", vp1);
//    vis_tmp.addText("before ICP", 10,10, 20, 1,1,1,"before_ICP",vp1);
////     vis_tmp.addPointCloud(rm.complete_cloud_, "model1", vp1);

//    vis_tmp.addPointCloud(scene_cloud_vis, gray, "scene2", vp2);
//    typename pcl::PointCloud<SceneT>::Ptr model_refined (new pcl::PointCloud<SceneT>);
//    pcl::transformPointCloud(*rm.visible_cloud_, *model_refined, rm.oh_->pose_refinement_);
//    vis_tmp.addText("after ICP", 10,10, 20, 1,1,1,"after_ICP",vp2);
//    vis_tmp.addPointCloud(model_refined, "model2", vp2);
//    vis_tmp.spin();


//    vis_tmp.addPointCloud(rm.model_->getAssembled(-1), "model2", vp2);
//    vis_tmp.addPointCloud(scene_cloud_downsampled_cropped, "scene2", vp2);
//    vis_tmp.addSphere(rm.model_->minPoint_, 0.03, 0, 255, 0, "min", vp2 );
//    vis_tmp.addSphere(rm.model_->maxPoint_, 0.03, 0, 255, 0, "max", vp2 );

//    vis_tmp.addPointCloud(rm.model_->getAssembled(-1), "model", vp3);
//    vis_tmp.addPointCloud(scene_cloud_downsampled_cropped, "scene3", vp3);

//    typename pcl::PointCloud<SceneT>::Ptr model_refined (new pcl::PointCloud<SceneT>);
//    pcl::transformPointCloud(*rm.complete_cloud_, *model_refined, refined_tf);
//    vis_tmp.addPointCloud(model_refined, "model2", vp3);
//    pcl::visualization::PointCloudColorHandlerCustom<SceneT> gray (scene_cloud_vis, 128, 128, 128);
//    vis_tmp.addPointCloud(scene_cloud_vis, gray, "input_rm_vp2_model_", vp3);
//    vis_tmp.setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_rm_vp2_model_", vp3);
//    vis_tmp.addCube( minPoint.x, maxPoint.x, minPoint.y, maxPoint.y, minPoint.z, maxPoint.z, 0, 1., 0, "bb", vp3);
//    vis_tmp.spin();
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

        VLOG(1) << "Downsampled scene cloud from " << scene_cloud_->points.size() << " to " << scene_normals_downsampled_->points.size() <<
                   " points using uniform sampling with a resolution of " <<  param_.resolution_mm_ / 1000.0 << "m.";
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
mets::gol_type
HypothesisVerification<ModelT, SceneT>::evaluateSolution (const boost::dynamic_bitset<> &solution)
{
    scene_pts_explained_solution_.clear();
    scene_pts_explained_solution_.resize( scene_cloud_downsampled_->points.size() );
    double cost = std::numeric_limits<double>::max();

    for(size_t i=0; i<global_hypotheses_.size(); i++)
    {
        const typename HVRecognitionModel<ModelT>::Ptr rm = global_hypotheses_[i];

        if( !solution[i])
            continue;

        for (Eigen::SparseVector<float>::InnerIterator it(rm->scene_explained_weight_); it; ++it)
            scene_pts_explained_solution_[ it.row() ].push_back( PtFitness(it.value(), i) );
    }

    for(auto spt_it = scene_pts_explained_solution_.begin(); spt_it!=scene_pts_explained_solution_.end(); ++spt_it)
        std::sort(spt_it->begin(), spt_it->end());

    double scene_fit = 0., duplicity = 0.;
    Eigen::Array<bool, Eigen::Dynamic, 1> scene_pt_is_explained( scene_cloud_downsampled_->points.size() );
    scene_pt_is_explained.setConstant( scene_cloud_downsampled_->points.size(), false);

    for(size_t s_id=0; s_id < scene_cloud_downsampled_->points.size(); s_id++)
    {
        const std::vector<PtFitness> &s_pt = scene_pts_explained_solution_[s_id];
        if(  !s_pt.empty() )
        {
            scene_fit += s_pt.back().fit_; // uses the maximum value for scene explanation
            scene_pt_is_explained(s_id) = true;
        }

        if ( s_pt.size() > 1 ) // two or more hypotheses explain the same scene point
            duplicity += s_pt[ s_pt.size() - 2 ].fit_; // uses the second best explanation
    }

    bool violates_smooth_region_check = false;
    if (param_.check_smooth_clusters_)
    {
        int max_label = scene_pt_smooth_label_id_.maxCoeff();
        for(int i=1; i<max_label; i++) // label "0" is for points not belonging to any smooth region
        {
            Eigen::Array<bool, Eigen::Dynamic, 1> s_pt_in_region = (scene_pt_smooth_label_id_.array() == i );
            Eigen::Array<bool, Eigen::Dynamic, 1> explained_pt_in_region = (s_pt_in_region.array() && scene_pt_is_explained.array());
            size_t num_explained_pts_in_region = explained_pt_in_region.count();
            size_t num_pts_in_smooth_regions = s_pt_in_region.count();

            if ( num_explained_pts_in_region > param_.min_pts_smooth_cluster_to_be_epxlained_ &&
                 (float)(num_explained_pts_in_region) / num_pts_in_smooth_regions < param_.min_ratio_cluster_explained_ )
            {
                violates_smooth_region_check = true;
                break;
            }
        }
    }

    if( !violates_smooth_region_check )
        cost = -( log( scene_fit ) - param_.clutter_regularizer_ * duplicity );

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

        if(!vis_pairwise_)
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
            if( (float)rm->visible_indices_by_octree_.size() / (float)rm->num_pts_full_model_ < param_.min_visible_ratio_)
            {
                rm->rejected_due_to_low_visibility_ = true;
                VLOG(1) << "Removed " << rm->oh_->model_id_ << " due to low visibility!";
            }
        }
    }
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::setHypotheses(std::vector<ObjectHypothesesGroup> &ohs)
{
    obj_hypotheses_groups_.clear();
    obj_hypotheses_groups_.resize(ohs.size());
    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
    {
        const ObjectHypothesesGroup &ohg = ohs[i];

        obj_hypotheses_groups_[i].resize(ohg.ohs_.size());
        for(size_t jj=0; jj<ohg.ohs_.size(); jj++)
        {
            typename ObjectHypothesis::Ptr oh = ohg.ohs_[jj];
            obj_hypotheses_groups_[i][jj].reset ( new HVRecognitionModel<ModelT>(oh) );
//            HVRecognitionModel<ModelT> &hv_oh = *obj_hypotheses_groups_[i][jj];

//            hv_oh.complete_cloud_.reset ( new pcl::PointCloud<ModelT>);
//            hv_oh.complete_cloud_normals_.reset (new pcl::PointCloud<pcl::Normal>);

//            bool found_model;
//            typename Model<ModelT>::ConstPtr m = m_db_->getModelById("", oh->model_id_, found_model);
//            typename pcl::PointCloud<ModelT>::ConstPtr model_cloud = m->getAssembled ( param_.resolution_mm_ );
//            pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud_const = m->getNormalsAssembled ( param_.resolution_mm_ );
//            pcl::transformPointCloud (*model_cloud, *hv_oh.complete_cloud_, oh->transform_);
//            transformNormals(*normal_cloud_const, *hv_oh.complete_cloud_normals_, oh->transform_);
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

    pcl_1_8::ConditionalEuclideanClustering<SceneTWithNormal> cec (false);
    cec.setInputCloud (scene_downsampled_w_normals);
    cec.setConditionFunction (custom_f);
    cec.setClusterTolerance ( param_.cluster_tolerance_ * 3. );
    cec.setMinClusterSize ( param_.min_points_ );
    cec.setMaxClusterSize (std::numeric_limits<int>::max());
    pcl_1_8::IndicesClusters clusters;
    cec.segment (clusters);

    scene_pt_smooth_label_id_ = Eigen::VectorXi::Zero( scene_cloud_downsampled_->points.size() );
    for (size_t i = 0; i < clusters.size (); i++)
    {
        for (int sidx : clusters[i].indices)
            scene_pt_smooth_label_id_( sidx ) = i+1; // label "0" for points not belonging to any smooth region
    }
}

//template<typename ModelT, typename SceneT>
//void
//HypothesisVerification<ModelT, SceneT>::computeLOffset(HVRecognitionModel<ModelT> &rm)const
//{
//    Eigen::VectorXf color_s ( rm.scene_explained_weight_.nonZeros() );

//    size_t kept=0;
//    for (Eigen::SparseVector<float>::InnerIterator it(rm.scene_explained_weight_); it; ++it)
//    {
//        int sidx = it.index();
//        color_s(kept++) = scene_color_channels_(sidx,0);
//    }

//    Eigen::VectorXf color_new = specifyHistogram( rm.pt_color_.col( 0 ), color_s, 100, 0.f, 100.f );
//    rm.pt_color_.col( 0 ) = color_new;
//}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::initialize()
{
    global_hypotheses_.clear();

    size_t num_hypotheses=0;
    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
        num_hypotheses += obj_hypotheses_groups_[i].size();

    elapsed_time_.push_back( std::pair<std::string, float>( "number of hypotheses",  num_hypotheses)) ;

    {
        StopWatch t("Downsampling scene cloud");
        downsampleSceneCloud();
    }

    elapsed_time_.push_back( std::pair<std::string, float>( "number of downsampled scene points (HV)",  scene_cloud_downsampled_->points.size()));

    if( img_boundary_distance_.empty() )
    {
        const cv::Mat depth_registration_mask = cam_->getCameraDepthRegistrationMask();

        if(depth_registration_mask.empty())
        {
            LOG(WARNING) << "Depth registration mask not set. Using the whole image!";
            img_boundary_distance_ = cv::Mat (cam_->getHeight(), cam_->getWidth(), CV_32FC1);
        }
        else
        {
            VLOG(1) << "Computing distance transform to image boundary.";
            cv::distanceTransform(depth_registration_mask, img_boundary_distance_, CV_DIST_L2, 5);
        }
    }


#pragma omp parallel sections
    {
#pragma omp section
        {
            StopWatch t("Computing octree");
            octree_scene_downsampled_.reset(new pcl::octree::OctreePointCloudSearch<SceneT>( param_.resolution_mm_ / 1000.0));
            octree_scene_downsampled_->setInputCloud(scene_cloud_downsampled_);
            octree_scene_downsampled_->addPointsFromInputCloud();
        }

#pragma omp section
        {
            StopWatch t("Computing kd-tree");
            kdtree_scene_.reset( new pcl::search::KdTree<SceneT>);
            kdtree_scene_->setInputCloud (scene_cloud_downsampled_);
        }

#pragma omp section
        {
            StopWatch t("Computing octrees for model visibility computation");
            for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
            {
                for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                {
                    HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];
                    bool found_model_foo;
                    typename Model<ModelT>::ConstPtr m = m_db_->getModelById("", rm.oh_->model_id_, found_model_foo);
                    typename pcl::PointCloud<ModelT>::ConstPtr model_cloud  = m->getAssembled ( -1 );  // use full resolution for rendering
                    rm.num_pts_full_model_ = model_cloud->points.size();

                    auto model_octree_it = octree_model_representation_.find( rm.oh_->model_id_ );
                    if( model_octree_it == octree_model_representation_.end() )
                    {
                        boost::shared_ptr< pcl::octree::OctreePointCloudPointVector<ModelT> > octree(new pcl::octree::OctreePointCloudPointVector<ModelT>(0.015f) );
                        octree->setInputCloud( model_cloud );
                        octree->addPointsFromInputCloud();
                        octree_model_representation_ [ rm.oh_->model_id_ ] = octree;
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
            StopWatch t("Input point cloud of scene is not organized. Doing depth-buffering to get organized point cloud");
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
                StopWatch t("Computing visible model points (1st run)");
#pragma omp parallel for schedule(dynamic)
                for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
                {
                    for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                    {
                        HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];
                        computeModelOcclusionByScene(rm);  //occlusion reasoning based on self-occlusion and occlusion from scene cloud(s)
                    }
                }
            }

            if( param_.icp_iterations_ )
            {
                {
                    std::stringstream desc; desc << "Pose refinement with " << param_.icp_iterations_ << " ICP iterations";
                    StopWatch t(desc.str());
#pragma omp parallel for schedule(dynamic)
                    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
                    {
                        for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                        {
                            HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];
                            refinePose(rm);
                        }
                    }
                }

                {
                    StopWatch t("Computing visible model points (2nd run)");
#pragma omp parallel for schedule(dynamic)
                    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
                    {
                        for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                        {
                            HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];
                            computeModelOcclusionByScene(rm);  //occlusion reasoning based on self-occlusion and occlusion from scene cloud(s)
                        }
                    }

                    size_t num_visible_object_points=0;
                    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
                    {
                        for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                        {
                            HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];
                            num_visible_object_points+=rm.visible_cloud_->points.size();
                        }
                    }
                    elapsed_time_.push_back( std::pair<std::string, float>( "visible object points",  num_visible_object_points));
                }
            }
//            {
//                // just mask out the visible normals as well
//                for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
//                {
//                    for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
//                    {
//                        HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];
//                        rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal>);
//                        pcl::copyPointCloud(*rm.complete_cloud_normals_, rm.visible_indices_, *rm.visible_cloud_normals_);
//                    }
//                }
//            }

            { //used for checking pairwise intersection of objects (relate amount of overlapping pixel of their 2D silhouette)
                StopWatch t("Computing 2D silhouette of visible object model");
#pragma omp parallel for schedule(dynamic)
                for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
                {
                    for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                    {
                        HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];
                        rm.processSilhouette(param_.do_smoothing_, param_.smoothing_radius_, param_.do_erosion_, param_.erosion_radius_, cam_->getWidth());
                    }
                }
            }

            {
                StopWatch t("Computing visible octree nodes");
#pragma omp parallel for schedule(dynamic)
                for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
                {
                    for(size_t jj=0; jj<obj_hypotheses_groups_[i].size(); jj++)
                    {
                        HVRecognitionModel<ModelT> &rm = *obj_hypotheses_groups_[i][jj];
                        computeVisibleOctreeNodes(rm);
                    }
                }
            }

            removeModelsWithLowVisibility();
        }

#pragma omp section
        {
            if(param_.check_smooth_clusters_)
            {
                StopWatch t("Extracting smooth clusters");
                extractEuclideanClustersSmooth();
            }
        }

#pragma omp section
        if(!param_.ignore_color_even_if_exists_)
        {
            StopWatch t("Converting scene color values");
            colorTransf_->convert(*scene_cloud_downsampled_, scene_color_channels_);
//            scene_color_channels_.col(0) = (scene_color_channels_.col(0) - Eigen::VectorXf::Ones(scene_color_channels_.rows())*50.f) / 50.f;
//            scene_color_channels_.col(1) = scene_color_channels_.col(1) / 150.f;
//            scene_color_channels_.col(2) = scene_color_channels_.col(2) / 150.f;
        }
    }


    {
        StopWatch t("Converting model color values");
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
//                        rm.pt_color_.col(0) = (rm.pt_color_.col(0) - Eigen::VectorXf::Ones(rm.pt_color_.rows())*50.f) / 50.f;
//                        rm.pt_color_.col(1) = rm.pt_color_.col(1) / 150.f;
//                        rm.pt_color_.col(2) = rm.pt_color_.col(1) / 150.f;
                    }

                }
            }
        }
    }

    {
        StopWatch t("Computing model to scene fitness");
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
    }


    global_hypotheses_.resize( obj_hypotheses_groups_.size() );

    // do non-maxima surpression on all hypotheses in a hypotheses group based on model fitness (i.e. select only the one hypothesis in group with best model fit)
    size_t kept=0;
    for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
    {
        std::vector<typename HVRecognitionModel<ModelT>::Ptr > ohg = obj_hypotheses_groups_[i];

        if (ohg.empty())
            continue;

        std::sort(ohg.begin(), ohg.end(), HVRecognitionModel<ModelT>::modelFitCompare);
        global_hypotheses_[kept++] = ohg[0];

        for(size_t jj=1; jj<ohg.size(); jj++)
        {
            ohg[jj]->rejected_due_to_better_hypothesis_in_group_ = true;
            VLOG(1) << ohg[jj]->oh_->class_id_ << " " << ohg[jj]->oh_->model_id_ << " is rejected due to better hypotheses in global hypotheses group.";
        }
    }
    global_hypotheses_.resize(kept);
    obj_hypotheses_groups_.clear(); // free space


    if( vis_model_ )
    {
        for(size_t i=0; i<global_hypotheses_.size(); i++)
        {
            VLOG(1) << "Visualizing hypothesis " << i;
            vis_model_->visualize( this, *global_hypotheses_[i]);
        }
    }

    size_t kept_hypotheses = 0;
    for(size_t i=0; i<global_hypotheses_.size(); i++)
    {
        typename HVRecognitionModel<ModelT>::Ptr rm = global_hypotheses_[i];

        VLOG(1) << rm->oh_->class_id_ << " " << rm->oh_->model_id_ << " with hypothesis id " << i <<
                   " has number of outliers: " << rm->visible_pt_is_outlier_.count() << ", scene explained weights " <<
                   rm->scene_explained_weight_.sum() << ".";

        rm->is_outlier_ = isOutlier(*rm);

        if (rm->is_outlier_)
            VLOG(1) << rm->oh_->class_id_ << " " << rm->oh_->model_id_ << " is rejected due to low model fitness score.";


        if ( !rm->isRejected() )
            global_hypotheses_[kept_hypotheses++] = global_hypotheses_[i];
        else
            VLOG(1) << rm->oh_->class_id_ << " " << rm->oh_->model_id_ << " with hypothesis id " << i << " is rejected.";
    }

    global_hypotheses_.resize( kept_hypotheses );

    elapsed_time_.push_back( std::pair<std::string, float>( "hypotheses left for global optimization",  kept_hypotheses));


    if( !kept_hypotheses )
        return;

    {
        StopWatch t("Computing pairwise intersection");
        computePairwiseIntersection();
    }

    if( vis_pairwise_ )
        vis_pairwise_->visualize(this);
}

template<typename ModelT, typename SceneT>
void
HypothesisVerification<ModelT, SceneT>::optimize ()
{
    if ( VLOG_IS_ON(1) )
    {
        VLOG(1) << global_hypotheses_.size() << " hypotheses are left for global verification after individual hypotheses rejection. These are the left hypotheses: ";
        for (size_t i=0; i<global_hypotheses_.size(); i++)
            VLOG(1) << global_hypotheses_[i]->oh_->class_id_ << " " << global_hypotheses_[i]->oh_->model_id_;
    }

    solution_ = boost::dynamic_bitset<>(global_hypotheses_.size(), 0);

    if(param_.initial_status_)
        solution_.set();

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

    if( vis_cues_ )
        cost_logger_->setVisualizeFunction(visualize_cues_during_logger_);

    switch( param_.opt_type_ )
    {
    case HV_OptimizationType::LocalSearch:
    {
        StopWatch t ("local search");
        neigh.UseReplaceMoves(false);
        mets::local_search<GHVmove_manager<ModelT, SceneT> > local ( model, *(cost_logger_.get()), neigh, 0, false);
        local.search ();
        (void)t;
        break;
    }
    case HV_OptimizationType::TabuSearch:
    {
        StopWatch t ("TABU search");
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
        StopWatch t("TABU search + LS (RM)");
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
        StopWatch t ("SA search");
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

    for(size_t i=0; i<global_hypotheses_.size(); i++)
    {
        if( solution_[i] )
            global_hypotheses_[i]->oh_->is_verified_ = true;

    }


    GHVSAModel<ModelT, SceneT> best_seen = static_cast<const GHVSAModel<ModelT, SceneT>&> (cost_logger_->best_seen());
    std::cout << "*****************************" << std::endl
              << "Solution: " << solution_ << std::endl
              << "Final cost: " << best_seen.cost_ << std::endl
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
    elapsed_time_.clear();

    {
        StopWatch t("Verification of object hypotheses");
        initialize();
    }

    if( vis_cues_ )
        visualize_cues_during_logger_ = boost::bind(&HypothesisVerification<ModelT, SceneT>::visualizeGOcues, this, _1, _2, _3);

    {
        StopWatch t("Optimizing object hypotheses verification cost function");
        optimize ();
    }

    cleanUp();
}


template<typename ModelT, typename SceneT>
bool
HypothesisVerification<ModelT, SceneT>::removeNanNormals (HVRecognitionModel<ModelT> &rm) const
{
    if(!rm.visible_cloud_normals_)
    {
        LOG(WARNING) << "Normals are not given for input model. Need to recompute. Consider to compute normals in advance!";
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
//    rm.model_scene_c_.reserve( rm.visible_cloud_->points.size () * param_.knn_inliers_ );
//    size_t kept=0;

//    pcl::visualization::PCLVisualizer vis;
//    int vp1, vp2;
//    vis.createViewPort(0,0,0.5,1,vp1);
//    vis.createViewPort(0.5,0,1,1,vp2);
//    vis.addPointCloud(rm.visible_cloud_, "vis_cloud", vp1);
//    pcl::visualization::PointCloudColorHandlerCustom<SceneT> gray (scene_cloud_downsampled_, 128, 128, 128);
//    vis.addPointCloud(scene_cloud_downsampled_, gray, "scene1", vp1);
//    vis.setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "scene1");


//    rm.visible_pt_is_outlier_.resize( rm.visible_cloud_->points.size (), 0);
    for (size_t midx = 0; midx < rm.visible_cloud_->points.size (); midx++)
    {
        std::vector<int> nn_indices;
        std::vector<float> nn_sqrd_distances;

//        bool is_outlier = true;
        double radius = search_radius_;
        octree_scene_downsampled_->radiusSearch(rm.visible_cloud_->points[midx], radius, nn_indices, nn_sqrd_distances);

//        vis.addSphere(rm.visible_cloud_->points[midx], 0.005, 0., 1., 0., "queryPoint", vp1 );

        const auto normal_m = rm.visible_cloud_normals_->points[midx].getNormalVector3fMap();
//        normal_m.normalize();

        for (size_t k = 0; k < nn_indices.size(); k++)
        {
            int sidx = nn_indices[ k ];
            float sqrd_3D_dist = nn_sqrd_distances[k];

//            std::stringstream pt_id; pt_id << "searched_pt_" << k;
//            vis.addSphere(scene_cloud_downsampled_->points[sidx], 0.005, 1., 0., 0., pt_id.str(), vp2 );
//            vis.addPointCloud(rm.visible_cloud_, "vis_cloud2", vp2);
//            vis.addPointCloud(scene_cloud_downsampled_, gray, "scene2", vp2);
//            vis.setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "scene2");

//            if (sqr_3D_dist > ( 1.5f * 1.5f * param_.inliers_threshold_ * param_.inliers_threshold_ ) ) ///TODO: consider camera's noise level
//                continue;

            ModelSceneCorrespondence c;
            c.model_id_ = midx;
            c.scene_id_ = sidx;
            c.dist_3D_ = sqrt(sqrd_3D_dist);

            const auto normal_s = scene_normals_downsampled_->points[sidx].getNormalVector3fMap();
//            normal_s.normalize();

            float dotp = std::min( 0.99999f, std::max(-0.99999f, normal_m.dot(normal_s) ) );
            c.normals_dotp_ = dotp;

//            CHECK (c.angle_surface_normals_rad_ <= M_PI) << "invalid normals: " << std::endl << normal_m << std::endl << std::endl << normal_s << std::endl << std::endl << "dotp: " << dotp << std::endl << "acos: " << c.angle_surface_normals_rad_ << std::endl;
//            CHECK (c.angle_surface_normals_rad_ >= 0.f ) << "invalid normals: " << std::endl << normal_m << std::endl << std::endl << normal_s << std::endl << std::endl << "dotp: " << dotp << std::endl << "acos: " << c.angle_surface_normals_rad_ << std::endl;

            const Eigen::VectorXf &color_m = rm.pt_color_.row( midx );
            const Eigen::VectorXf &color_s = scene_color_channels_.row( sidx );
            c.color_distance_ = color_dist_f_(color_s, color_m);

            c.fitness_ = getFitness( c );
            rm.model_scene_c_.push_back( c );

//            if(c.fitness_ > param_.min_fitness_)
//                is_outlier=false;
        }
//        vis.removeAllShapes(vp1);
//        rm.visible_pt_is_outlier_[ midx ] = is_outlier;
    }

//    vis.spin();
//    rm.model_scene_c_.resize(kept);

    std::sort( rm.model_scene_c_.begin(), rm.model_scene_c_.end() );

    if(param_.use_histogram_specification_)
    {
        boost::dynamic_bitset<> scene_pt_is_taken(scene_cloud_downsampled_->points.size(), 0);
        Eigen::VectorXf scene_color_for_model ( scene_cloud_downsampled_->points.size() );

        size_t kept=0;
        for( const ModelSceneCorrespondence &c : rm.model_scene_c_ )
        {
            int sidx = c.scene_id_;
            if( !scene_pt_is_taken[sidx] )
            {
                scene_pt_is_taken.set(sidx);
                scene_color_for_model(kept++) = scene_color_channels_(sidx,0);
            }
        }
        scene_color_for_model.conservativeResize(kept);

        if( kept )
        {
            float mean_l_value_scene = scene_color_for_model.mean();
            float mean_l_value_model = rm.pt_color_.col( 0 ).mean();

            float max_l_offset = 15.f;
            float l_compensation = std::max<float>(-max_l_offset, std::min<float>(max_l_offset, (mean_l_value_scene - mean_l_value_model)));
    //        Eigen::VectorXf color_new = specifyHistogram( rm.pt_color_.col( 0 ), scene_color_for_model, 50, min_l_value, max_l_value );
            rm.pt_color_.col( 0 ).array() = rm.pt_color_.col( 0 ).array() + l_compensation;

            for( ModelSceneCorrespondence &c : rm.model_scene_c_ )
            {
                int sidx = c.scene_id_;
                int midx = c.model_id_;

                const Eigen::VectorXf &color_m = rm.pt_color_.row( midx );
                const Eigen::VectorXf &color_s = scene_color_channels_.row( sidx );
                c.color_distance_ = color_dist_f_(color_s, color_m);
                c.fitness_ = getFitness( c );
            }
        }
    }

    boost::dynamic_bitset<> scene_explained_pts ( scene_cloud_downsampled_->points.size(), 0);
    boost::dynamic_bitset<> model_explained_pts ( rm.visible_cloud_->points.size(), 0);
    Eigen::VectorXf modelFit   = Eigen::VectorXf::Zero (rm.visible_cloud_->points.size());
    rm.scene_explained_weight_ = Eigen::SparseVector<float> (scene_cloud_downsampled_->points.size());
    rm.scene_explained_weight_.reserve( rm.model_scene_c_.size() );

    for( const ModelSceneCorrespondence &c : rm.model_scene_c_ )
    {
        int sidx = c.scene_id_;
        int midx = c.model_id_;

        if( !scene_explained_pts[sidx] )
        {
            scene_explained_pts.set(sidx);
            rm.scene_explained_weight_.insert(sidx) = c.fitness_ ;
        }

        if( !model_explained_pts[midx] )
        {
            model_explained_pts.set(midx);
            modelFit(midx) = c.fitness_;
        }
    }

    rm.model_fit_ = modelFit.sum();
    rm.oh_->confidence_ = rm.model_fit_/rm.visible_cloud_->points.size();

    VLOG(1) << "model fit of " << rm.oh_->model_id_ << ": " << rm.model_fit_ << " (normalized: " << rm.model_fit_/rm.visible_cloud_->points.size() << ").";
}

template<typename ModelT, typename SceneT>
std::vector<std::pair<std::string,float> > HypothesisVerification<ModelT, SceneT>::elapsed_time_;

#define PCL_INSTANTIATE_HypothesisVerification(ModelT, SceneT) template class V4R_EXPORTS HypothesisVerification<ModelT, SceneT>;
PCL_INSTANTIATE_PRODUCT(HypothesisVerification, ((pcl::PointXYZRGB))((pcl::PointXYZRGB)) )

}
