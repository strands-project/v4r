#include <v4r/apps/ObjectRecognizer.h>

#include <iostream>
#include <sstream>

#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <pcl/common/time.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/icp.h>

#include <v4r/change_detection/miscellaneous.h>
#include <v4r/change_detection/change_detection.h>
#include <v4r/common/camera.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/normals.h>
#include <v4r/common/graph_geometric_consistency.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/features/esf_estimator.h>
#include <v4r/features/global_simple_shape_estimator.h>
#include <v4r/features/global_concatenated.h>
#include <v4r/features/shot_local_estimator.h>
#include <v4r/features/sift_local_estimator.h>
#include <v4r/features/rops_local_estimator.h>
#include <v4r/keypoints/all_headers.h>
#include <v4r/io/filesystem.h>
#include <v4r/ml/all_headers.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/recognition/global_recognition_pipeline.h>
#include <v4r/recognition/multiview_recognizer.h>
#include <v4r/registration/noise_model_based_cloud_integration.h>
#include <v4r/segmentation/all_headers.h>


#include <v4r/segmentation/plane_utils.h>
#include <v4r/segmentation/segmentation_utils.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>

#include <sys/time.h>
#include <sys/resource.h>

namespace po = boost::program_options;

namespace v4r
{

namespace apps
{

template<typename PointT>
void ObjectRecognizer<PointT>::initialize(const std::vector<std::string> &command_line_arguments)
{
    bool visualize_hv_go_cues = false;
    bool visualize_hv_model_cues = false;
    bool visualize_hv_pairwise_cues = false;
    bool visualize_keypoints = false;
    bool visualize_global_results = false;
    bool retrain = false;

    po::options_description desc("Single-View Object Instance Recognizer\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("model_dir,m", po::value<std::string>(&models_dir_)->required(), "Models directory")
            ("visualize,v", po::bool_switch(&visualize_), "visualize recognition results")
            ("skip_verification", po::bool_switch(&skip_verification_), "if true, skips verification (only hypotheses generation)")
            ("hv_vis_cues", po::bool_switch(&visualize_hv_go_cues), "If set, visualizes cues computated at the hypothesis verification stage such as inlier, outlier points. Mainly used for debugging.")
            ("hv_vis_model_cues", po::bool_switch(&visualize_hv_model_cues), "If set, visualizes the model cues. Useful for debugging")
            ("hv_vis_pairwise_cues", po::bool_switch(&visualize_hv_pairwise_cues), "If set, visualizes the pairwise cues. Useful for debugging")
            ("rec_visualize_keypoints", po::bool_switch(&visualize_keypoints), "If set, visualizes detected keypoints.")
            ("rec_visualize_global_pipeline", po::bool_switch(&visualize_global_results), "If set, visualizes segments and results from global pipeline.")
            ("retrain", po::bool_switch(&retrain), "If set, retrains the object models no matter if they already exists.")
            ("recognizer_remove_planes", po::value<bool>(&param_.remove_planes_)->default_value(param_.remove_planes_), "if enabled, removes the dominant plane in the input cloud (given thera are at least N inliers)")
            ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(command_line_arguments).options(desc).allow_unregistered().run();
    std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; to_pass_further.push_back("-h"); }
    try { po::notify(vm); }
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;  }

    // ====== DEFINE CAMERA =======
    camera_.reset (new Camera(param_.camera_config_xml_) );

    cv::Mat_<uchar> img_mask = cv::imread(param_.depth_img_mask_, CV_LOAD_IMAGE_GRAYSCALE);
    if( img_mask.data )
        camera_->setCameraDepthRegistrationMask( img_mask );
    else
        LOG(WARNING) << "No camera depth registration mask provided. Assuming all pixels have valid depth.";


    // ====== DEFINE VISUALIZATION PARAMETER =======
    PCLVisualizationParams::Ptr vis_param (new PCLVisualizationParams);
    vis_param->no_text_ = false;
    vis_param->bg_color_ = Eigen::Vector3i(255, 255, 255);
    vis_param->text_color_ = Eigen::Vector3f(0.f, 0.f, 0.f);
    vis_param->fontsize_ = 12;
    vis_param->coordinate_axis_scale_ = 0.2f;


    // ==== Fill object model database ==== ( assumes each object is in a seperate folder named after the object and contains and "views" folder with the training views of the object)
    model_database_.reset ( new Source<PointT> (models_dir_) );

    normal_estimator_ = v4r::initNormalEstimator<PointT> ( param_.normal_computation_method_, to_pass_further );


    // ====== SETUP MULTI PIPELINE RECOGNIZER ======
    typename v4r::MultiRecognitionPipeline<PointT>::Ptr multipipeline (new v4r::MultiRecognitionPipeline<PointT> );
    local_recognition_pipeline_.reset(new LocalRecognitionPipeline<PointT>);
    {
        // ====== SETUP LOCAL RECOGNITION PIPELINE =====
        if(param_.do_sift_ || param_.do_shot_)
        {
            local_recognition_pipeline_->setModelDatabase( model_database_ );

            if(param_.use_graph_based_gc_grouping_)
            {
                GraphGeometricConsistencyGroupingParameter gcparam;
                gcparam.gc_size_ = param_.cg_size_;
                gcparam.gc_threshold_ = param_.cg_thresh_;
                to_pass_further = gcparam.init(to_pass_further);
                GraphGeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ>::Ptr gc_clusterer
                        (new GraphGeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ>);
                local_recognition_pipeline_->setCGAlgorithm( gc_clusterer );
            }
            else
            {
                boost::shared_ptr< pcl::GeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> > gc_clusterer
                        (new pcl::GeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ>);
                gc_clusterer->setGCSize( param_.cg_size_ );
                gc_clusterer->setGCThreshold( param_.cg_thresh_ );
                local_recognition_pipeline_->setCGAlgorithm( gc_clusterer );
            }

            if(param_.do_sift_)
            {
                LocalRecognizerParameter sift_param(param_.sift_config_xml_);
                typename LocalFeatureMatcher<PointT>::Ptr sift_rec (new LocalFeatureMatcher<PointT>(sift_param));
                typename SIFTLocalEstimation<PointT>::Ptr sift_est (new SIFTLocalEstimation<PointT>);
                sift_est->setMaxDistance(std::numeric_limits<float>::max());
                sift_rec->addFeatureEstimator( sift_est );
                local_recognition_pipeline_->addLocalFeatureMatcher(sift_rec);
            }
            if(param_.do_shot_)
            {

                LocalRecognizerParameter shot_pipeline_param(param_.shot_config_xml_);
                typename LocalFeatureMatcher<PointT>::Ptr shot_rec (new LocalFeatureMatcher<PointT>(shot_pipeline_param));
                std::vector<typename v4r::KeypointExtractor<PointT>::Ptr > keypoint_extractor = initKeypointExtractors<PointT>( param_.shot_keypoint_extractor_method_, to_pass_further );

                for( typename v4r::KeypointExtractor<PointT>::Ptr ke : keypoint_extractor)
                    shot_rec->addKeypointExtractor( ke );

                for(float support_radius : param_.keypoint_support_radii_)
                {
                    SHOTLocalEstimationParameter shot_param;
                    shot_param.support_radius_ = support_radius;
//                    shot_param.init( to_pass_further );
                    typename SHOTLocalEstimation<PointT>::Ptr shot_est (new SHOTLocalEstimation<PointT> (shot_param) );

    //                ROPSLocalEstimationParameter rops_param;
    //                rops_param.init( to_pass_further );
    //                typename ROPSLocalEstimation<PointT>::Ptr rops_est (new ROPSLocalEstimation<PointT> (rops_param) );


                    shot_rec->addFeatureEstimator( shot_est );
                }
                shot_rec->setVisualizeKeypoints(visualize_keypoints);
                local_recognition_pipeline_->addLocalFeatureMatcher(shot_rec);
            }

            typename RecognitionPipeline<PointT>::Ptr rec_pipeline_tmp = boost::static_pointer_cast<RecognitionPipeline<PointT> > (local_recognition_pipeline_);
            multipipeline->addRecognitionPipeline(rec_pipeline_tmp);
        }

        // ====== SETUP GLOBAL RECOGNITION PIPELINE =====

        if( !param_.global_feature_types_.empty() )
        {
            typename GlobalRecognitionPipeline<PointT>::Ptr global_recognition_pipeline (new GlobalRecognitionPipeline<PointT>);
            typename v4r::Segmenter<PointT>::Ptr segmenter = v4r::initSegmenter<PointT>( param_.segmentation_method_, to_pass_further);
            global_recognition_pipeline->setSegmentationAlgorithm( segmenter );

            for(size_t global_pipeline_id = 0; global_pipeline_id < param_.global_feature_types_.size(); global_pipeline_id++)
            {
                    GlobalConcatEstimatorParameter p;
                    p.feature_type = param_.global_feature_types_[global_pipeline_id];
                    typename GlobalConcatEstimator<PointT>::Ptr global_concat_estimator (new GlobalConcatEstimator<PointT>(to_pass_further, p));

//                    typename OURCVFHEstimator<PointT>::Ptr ourcvfh_estimator (new OURCVFHEstimator<PointT>);
                    Classifier::Ptr classifier = initClassifier( param_.classification_methods_[global_pipeline_id], to_pass_further);

                    GlobalRecognizerParameter global_rec_param ( param_.global_recognition_pipeline_config_[global_pipeline_id] );
                    typename GlobalRecognizer<PointT>::Ptr global_r (new GlobalRecognizer<PointT>( global_rec_param ));
                    global_r->setFeatureEstimator( global_concat_estimator );
                    global_r->setClassifier( classifier );
                    global_recognition_pipeline->addRecognizer( global_r );
            }

            global_recognition_pipeline->setVisualizeClusters( visualize_global_results );

            typename RecognitionPipeline<PointT>::Ptr rec_pipeline_tmp = boost::static_pointer_cast<RecognitionPipeline<PointT> > (global_recognition_pipeline);
            multipipeline->addRecognitionPipeline( rec_pipeline_tmp );
        }

        multipipeline->setModelDatabase( model_database_ );
        multipipeline->setNormalEstimator( normal_estimator_ );
        multipipeline->setVisualizationParameter(vis_param);
        multipipeline->initialize( models_dir_, retrain );
    }



    if( param_.use_multiview_ )
    {
        typename RecognitionPipeline<PointT>::Ptr rec_pipeline = boost::static_pointer_cast<RecognitionPipeline<PointT> > (multipipeline);
        typename v4r::MultiviewRecognizer<PointT>::Ptr mv_rec ( new v4r::MultiviewRecognizer<PointT> );
        mv_rec->setSingleViewRecognitionPipeline( rec_pipeline );
        mrec_ = mv_rec;
    }
    else
        mrec_ = multipipeline;


    if(!skip_verification_)
    {
        // ====== SETUP HYPOTHESES VERIFICATION =====
        HV_Parameter paramHV;
        paramHV.load (param_.hv_config_xml_);
        hv_.reset (new HypothesisVerification<PointT, PointT> (camera_, paramHV) );

        if( visualize_hv_go_cues )
            hv_->visualizeCues(vis_param);
        if( visualize_hv_model_cues )
            hv_->visualizeModelCues(vis_param);
        if( visualize_hv_pairwise_cues )
            hv_->visualizePairwiseCues(vis_param);

        hv_->setModelDatabase(model_database_);
    }

    if (param_.remove_planes_)
    {
        // --plane_extraction_method 8 -z 2 --remove_points_below_selected_plane 1 --remove_planes 0 --plane_extractor_maxStepSize 0.1 --use_highest_plane 1 --min_plane_inliers 10000
        std::vector<std::string> additional_cs_arguments = {"--skip_segmentation", "1",
                                                            "--remove_selected_plane", "1",
                                                            "--remove_points_below_selected_plane", "1",
                                                            "--use_highest_plane", "1"};
        to_pass_further.insert(to_pass_further.end(), additional_cs_arguments.begin(), additional_cs_arguments.end());
        v4r::apps::CloudSegmenterParameter cs_param;
        to_pass_further = cs_param.init( to_pass_further );
        cloud_segmenter_.reset( new v4r::apps::CloudSegmenter<PointT> (cs_param) );
        cloud_segmenter_->initialize(to_pass_further);
    }

    if(visualize_)
    {
        rec_vis_.reset( new v4r::ObjectRecognitionVisualizer<PointT>);
        rec_vis_->setModelDatabase(model_database_);
    }
}

template<typename PointT>
void
ObjectRecognizer<PointT>::detectChanges(View &v)
{
    v.removed_points_.reset( new pcl::PointCloud<PointT>);

    typename pcl::PointCloud<PointT>::Ptr new_observation_aligned(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*v.processed_cloud_, *new_observation_aligned, v.camera_pose_);

    // downsample
    float resolution = 0.005f;
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud (new_observation_aligned);
    vg.setLeafSize (resolution, resolution, resolution);
    typename pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
    vg.filter (*cloud_filtered);
    new_observation_aligned = cloud_filtered;

    if( registered_scene_cloud_ && !registered_scene_cloud_->points.empty() )
    {
        v4r::ChangeDetector<PointT> detector;
        detector.detect(registered_scene_cloud_, new_observation_aligned, Eigen::Affine3f(v.camera_pose_), param_.tolerance_for_cloud_diff_);
//        v4r::ChangeDetector<PointT>::removePointsFrom(registered_scene_cloud_, detector.getRemoved());
        *v.removed_points_ += *(detector.getRemoved());
//        *changing_scene += *(detector.getAdded());
    }
}

template<typename PointT>
std::vector<typename ObjectHypothesis<PointT>::Ptr >
ObjectRecognizer<PointT>::recognize(const typename pcl::PointCloud<PointT>::ConstPtr &cloud)
{
    //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
//    cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
//    cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);

    const Eigen::Matrix4f camera_pose = v4r::RotTrans2Mat4f( cloud->sensor_orientation_, cloud->sensor_origin_ );

    typename pcl::PointCloud<PointT>::Ptr processed_cloud (new pcl::PointCloud<PointT>(*cloud));

    verified_hypotheses_.clear();

    std::vector<double> elapsed_time;

    pcl::PointCloud<pcl::Normal>::Ptr normals;
    if( mrec_->needNormals() || hv_ )
    {
        pcl::ScopeTime t("Computing normals");
        normal_estimator_->setInputCloud( processed_cloud );
        normals = normal_estimator_->compute();
        mrec_->setSceneNormals( normals );
        elapsed_time.push_back( t.getTime() );
    }

    if(param_.remove_planes_)
    {
        cloud_segmenter_->setNormals( normals );
        cloud_segmenter_->segment( processed_cloud );
        processed_cloud = cloud_segmenter_->getProcessedCloud();
        const Eigen::Vector4f chosen_plane = cloud_segmenter_->getSelectedPlane();
        mrec_->setTablePlane( chosen_plane );
    }

    // ==== FILTER POINTS BASED ON DISTANCE =====
    for(PointT &p : processed_cloud->points)
    {
        if (pcl::isFinite(p) && p.getVector3fMap().norm() > param_.chop_z_)
            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
    }

    {
        pcl::ScopeTime t("Generation of object hypotheses");
        mrec_->setInputCloud ( processed_cloud );
        mrec_->recognize();
        generated_object_hypotheses_ = mrec_->getObjectHypothesis();
        elapsed_time.push_back( t.getTime() );
    }


//    if(param_.icp_iterations_)
//    {
//        refinePose(processed_cloud);
//    }

    if(!skip_verification_)
    {
        pcl::ScopeTime t("Verification of object hypotheses");
        hv_->setHypotheses( generated_object_hypotheses_ );

        if( param_.use_multiview_ && param_.use_multiview_hv_ )
        {
            NMBasedCloudIntegrationParameter nm_int_param;
            nm_int_param.min_points_per_voxel_ = 1;
            nm_int_param.octree_resolution_ = 0.002f;

            NguyenNoiseModelParameter nm_param;

            View v;
            v.cloud_ = cloud;
            v.processed_cloud_ = processed_cloud;
            v.camera_pose_ = camera_pose;
            v.cloud_normals_ = normals;

            {
                pcl::StopWatch tt;
                NguyenNoiseModel<PointT> nm (nm_param);
                nm.setInputCloud( processed_cloud );
                nm.setInputNormals( normals );
                nm.compute();
                v.pt_properties_ = nm.getPointProperties();
                VLOG(1) << "Computing noise model parameter for cloud took " << tt.getTime() << " ms.";
            }


            if (param_.use_change_detection_ )
            {
                pcl::StopWatch tt;
                detectChanges(v);

                typename pcl::PointCloud<PointT>::Ptr removed_points_cumulative(new pcl::PointCloud<PointT>(*v.removed_points_));

                for(int v_id=views_.size()-1; v_id>=0; v_id--)
                {
                    View &vv = views_[v_id];

                    typename pcl::PointCloud<PointT>::Ptr view_aligned(new pcl::PointCloud<PointT>);
                    pcl::transformPointCloud(*vv.processed_cloud_, *view_aligned, vv.camera_pose_);

                    typename pcl::PointCloud<PointT>::Ptr cloud_tmp(new pcl::PointCloud<PointT>);

                    if(vv.removed_points_)
                        *removed_points_cumulative += *vv.removed_points_;

                    if( !removed_points_cumulative->points.empty() )
                    {
                        std::vector<int> preserved_indices;
                        v4r::ChangeDetector<PointT>::difference(
                                    *view_aligned,
                                    removed_points_cumulative,
                                    *cloud_tmp,
                                    preserved_indices,
                                    param_.tolerance_for_cloud_diff_);

                        /* Visualization of changes removal for reconstruction:
                        Cloud rec_changes;
                        rec_changes += *cloud_transformed;
                        v4r::VisualResultsStorage::copyCloudColored(*removed_points_cumulated_history_[view_id], rec_changes, 255, 0, 0);
                        v4r::VisualResultsStorage::copyCloudColored(*cloud_tmp, rec_changes, 200, 0, 200);
                        stringstream ss;
                        ss << view_id;
                        visResStore.savePcd("reconstruction_changes_" + ss.str(), rec_changes);*/

                        boost::dynamic_bitset<> preserved_mask( view_aligned->points.size(), 0 );
                        for (int idx : preserved_indices)
                            preserved_mask.set(idx);

                        for (size_t j = 0; j < preserved_mask.size(); j++)
                        {
                            if ( !preserved_mask[j] )
                            {
                                PointT &p = vv.processed_cloud_->points[j];
                                p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                            }
                        }
                        LOG(INFO) << "Points by change detection removed: " << vv.processed_cloud_->points.size() - preserved_indices.size() << ".";

//                        NguyenNoiseModel<PointT> nm_again (nm_param);
//                        nm_again.setInputCloud( vv.processed_cloud_after_removal_ );
//                        nm_again.setInputNormals( vv.cloud_normals_ );
//                        nm_again.compute();
//                        vv.pt_properties_ = nm_again.getPointProperties();
                    }
                }

                VLOG(1) << "Change detection took " << tt.getTime() << " ms.";
            }


            views_.push_back(v);

            std::vector<typename pcl::PointCloud<PointT>::ConstPtr> views (views_.size());  ///< all views in multi-view sequence
            std::vector<typename pcl::PointCloud<PointT>::ConstPtr> processed_views (views_.size());  ///< all processed views in multi-view sequence
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > camera_poses (views_.size());   ///< all absolute camera poses in multi-view sequence
            std::vector< pcl::PointCloud<pcl::Normal>::ConstPtr > views_normals (views_.size());  ///< all view normals in multi-view sequence
            std::vector< std::vector<std::vector<float> > > views_pt_properties (views_.size());  ///< all Nguyens noise model point properties in multi-view sequence

            for(size_t v_id=0; v_id<views_.size(); v_id++)
            {
                const View &vv = views_[v_id];
                views[v_id] = vv.cloud_;
                processed_views[v_id] = vv.processed_cloud_;
                camera_poses[v_id] = vv.camera_pose_;
                views_normals[v_id] = vv.cloud_normals_;
                views_pt_properties[v_id] = vv.pt_properties_;
            }

            registered_scene_cloud_.reset(new pcl::PointCloud<PointT>);
            NMBasedCloudIntegration<PointT> nmIntegration (nm_int_param);
            nmIntegration.setInputClouds( processed_views );
            nmIntegration.setPointProperties( views_pt_properties );
            nmIntegration.setTransformations( camera_poses );
            nmIntegration.setInputNormals( views_normals );
//            nmIntegration.setIndices(obj_indices);
            nmIntegration.compute(registered_scene_cloud_);
            std::vector< typename pcl::PointCloud<PointT>::Ptr > clouds_used;
            nmIntegration.getInputCloudsUsed(clouds_used);
            nmIntegration.getOutputNormals( normals );

            const Eigen::Matrix4f tf_global2camera = camera_pose.inverse();

            typename pcl::PointCloud<PointT>::Ptr registered_scene_cloud_aligned (new pcl::PointCloud<PointT>);
            pcl::transformPointCloud(*registered_scene_cloud_, *registered_scene_cloud_aligned, tf_global2camera );
            v4r::transformNormals(*normals, *normals, tf_global2camera );

//            static pcl::visualization::PCLVisualizer vis ("final registration");
//            int vp1, vp2, vp3;
//            vis.createViewPort(0,0,0.33,1,vp1);
//            vis.createViewPort(0.33,0,0.66,1,vp2);
//            vis.createViewPort(0.66,0,1,1,vp3);
//            vis.removeAllPointClouds();

//            typename pcl::PointCloud<PointT>::Ptr registered_scene_cloud_aligned_vis(new pcl::PointCloud<PointT> (*registered_scene_cloud_aligned));
//            typename pcl::PointCloud<PointT>::Ptr registered_scene_cloud_vis(new pcl::PointCloud<PointT> (*registered_scene_cloud_));
//            typename pcl::PointCloud<PointT>::Ptr removed_points_vis(new pcl::PointCloud<PointT> (*v.processed_cloud_));


//            registered_scene_cloud_aligned_vis->sensor_origin_ = Eigen::Vector4f::Zero();
//            registered_scene_cloud_aligned_vis->sensor_orientation_ = Eigen::Quaternionf::Identity();
//            registered_scene_cloud_vis->sensor_origin_ = Eigen::Vector4f::Zero();
//            registered_scene_cloud_vis->sensor_orientation_ = Eigen::Quaternionf::Identity();
//            removed_points_vis->sensor_origin_ = Eigen::Vector4f::Zero();
//            removed_points_vis->sensor_orientation_ = Eigen::Quaternionf::Identity();

//            vis.addPointCloud(registered_scene_cloud_aligned_vis, "registered_clouda",vp1);
//            vis.addPointCloud(registered_scene_cloud_vis, "registered_cloudb",vp2);
//            vis.addPointCloud(removed_points_vis, "registered_cloudc",vp3);
//            vis.spin();

            hv_->setSceneCloud( registered_scene_cloud_aligned );
            hv_->setOcclusionCloudsAndAbsoluteCameraPoses(views, camera_poses);
        }
        else
            hv_->setSceneCloud( cloud );

        hv_->setNormals( normals );

        hv_->verify();
//        verified_hypotheses_ = hv_->getVerifiedHypotheses();
        elapsed_time.push_back( t.getTime() );
    }

    for ( const typename ObjectHypothesis<PointT>::Ptr &voh : verified_hypotheses_ )
    {
        const std::string &model_id = voh->model_id_;
        const Eigen::Matrix4f &tf = voh->transform_;
        float confidence = voh->confidence_;
        LOG(INFO) << "********************" << model_id << " (confidence: " << confidence << ") " << std::endl << tf << std::endl << std::endl;
    }

    if ( visualize_ )
    {
        const std::map<std::string, typename LocalObjectModel::ConstPtr> lomdb = local_recognition_pipeline_->getLocalObjectModelDatabase();
        rec_vis_->setCloud( cloud );

        if( param_.use_multiview_ && param_.use_multiview_hv_ )
        {
            const Eigen::Matrix4f tf_global2camera = camera_pose.inverse();
            typename pcl::PointCloud<PointT>::Ptr registered_scene_cloud_aligned (new pcl::PointCloud<PointT>);
            pcl::transformPointCloud(*registered_scene_cloud_, *registered_scene_cloud_aligned, tf_global2camera );
            rec_vis_->setProcessedCloud( registered_scene_cloud_aligned );
        }
        else
            rec_vis_->setProcessedCloud( processed_cloud );

        rec_vis_->setNormals(normals);

        rec_vis_->setGeneratedObjectHypotheses( generated_object_hypotheses_ );
//        rec_vis_->setRefinedGeneratedObjectHypotheses( generated_object_hypotheses_refined_ );
        rec_vis_->setLocalModelDatabase(lomdb);
//        rec_vis_->setVerifiedObjectHypotheses( verified_hypotheses_ );
        rec_vis_->visualize();
    }

    return verified_hypotheses_;
}

template <typename PointT>
void
ObjectRecognizer<PointT>::resetMultiView()
{
    if(param_.use_multiview_)
    {
        views_.clear();

        typename v4r::MultiviewRecognizer<PointT>::Ptr mv_rec =
                boost::dynamic_pointer_cast<  v4r::MultiviewRecognizer<PointT> > (mrec_);
        if( mrec_ )
            mv_rec->clear();
        else
            LOG(ERROR) << "Cannot reset multi-view recognizer because given recognizer is not a multi-view recognizer!";
    }

}

template class V4R_EXPORTS ObjectRecognizer<pcl::PointXYZRGB>;

}

}
