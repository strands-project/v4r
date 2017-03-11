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

#include <v4r/common/camera.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/normals.h>
#include <v4r/common/graph_geometric_consistency.h>
#include <v4r/features/esf_estimator.h>
#include <v4r/features/shot_local_estimator.h>
#include <v4r/features/sift_local_estimator.h>
#include <v4r/keypoints/uniform_sampling_extractor.h>
#include <v4r/io/filesystem.h>
#include <v4r/ml/all_headers.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/recognition/global_recognition_pipeline.h>
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
            ("retrain", po::bool_switch(&retrain), "If set, retrains the object models no matter if they already exists.")
            ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(command_line_arguments).options(desc).allow_unregistered().run();
    std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; to_pass_further.push_back("-h"); }
    try { po::notify(vm); }
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;  }

    // ====== DEFINE CAMERA =======
    Camera::Ptr xtion (new Camera(param_.camera_config_xml_) );

    cv::Mat_<uchar> img_mask = cv::imread(param_.depth_img_mask_, CV_LOAD_IMAGE_GRAYSCALE);
    if( img_mask.data )
        xtion->setCameraDepthRegistrationMask( img_mask );
    else
        std::cout << "No camera depth registration mask provided. Assuming all pixels have valid depth." << std::endl;


    // ==== Fill object model database ==== ( assumes each object is in a seperate folder named after the object and contains and "views" folder with the training views of the object)
    typename Source<PointT>::Ptr model_database (new Source<PointT> (models_dir_));

    normal_estimator_ = v4r::initNormalEstimator<PointT> ( param_.normal_computation_method_, to_pass_further );


    // ====== SETUP MULTI PIPELINE RECOGNIZER ======
    mrec_.reset( new v4r::MultiRecognitionPipeline<PointT> );
    local_recognition_pipeline_.reset(new LocalRecognitionPipeline<PointT>);
    {
        // ====== SETUP LOCAL RECOGNITION PIPELINE =====
        if(param_.do_sift_ || param_.do_shot_)
        {
            local_recognition_pipeline_->setModelDatabase( model_database );

            if(param_.use_graph_based_gc_grouping_)
            {
                v4r::GraphGeometricConsistencyGroupingParameter gcparam;
                gcparam.gc_size_ = param_.cg_size_;
                gcparam.gc_threshold_ = param_.cg_thresh_;
                to_pass_further = gcparam.init(to_pass_further);
                v4r::GraphGeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ>::Ptr gc_clusterer
                        (new v4r::GraphGeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ>);
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
                sift_rec->setFeatureEstimator( sift_est );
                local_recognition_pipeline_->addLocalFeatureMatcher(sift_rec);
            }
            if(param_.do_shot_)
            {
                typename SHOTLocalEstimation<PointT>::Ptr shot_est (new SHOTLocalEstimation<PointT>);
                typename UniformSamplingExtractor<PointT>::Ptr extr (new UniformSamplingExtractor<PointT>(0.02f));
                typename KeypointExtractor<PointT>::Ptr keypoint_extractor = boost::static_pointer_cast<KeypointExtractor<PointT> > (extr);

                LocalRecognizerParameter shot_param(param_.shot_config_xml_);
                typename LocalFeatureMatcher<PointT>::Ptr shot_rec (new LocalFeatureMatcher<PointT>(shot_param));
                shot_rec->addKeypointExtractor( keypoint_extractor );
                shot_rec->setFeatureEstimator( shot_est );
                local_recognition_pipeline_->addLocalFeatureMatcher(shot_rec);
            }

            typename RecognitionPipeline<PointT>::Ptr rec_pipeline_tmp = boost::static_pointer_cast<RecognitionPipeline<PointT> > (local_recognition_pipeline_);
            mrec_->addRecognitionPipeline(rec_pipeline_tmp);
        }

        // ====== SETUP GLOBAL RECOGNITION PIPELINE =====

        if(param_.do_esf_ || param_.do_alexnet_)
        {
            typename GlobalRecognitionPipeline<PointT>::Ptr global_recognition_pipeline (new GlobalRecognitionPipeline<PointT>);
            typename v4r::Segmenter<PointT>::Ptr segmenter = v4r::initSegmenter<PointT>( param_.segmentation_method_, to_pass_further);
            global_recognition_pipeline->setSegmentationAlgorithm( segmenter );

            if(param_.do_esf_)
            {
                typename ESFEstimation<PointT>::Ptr esf_estimator (new ESFEstimation<PointT>);
                Classifier::Ptr classifier = initClassifier( param_.esf_classification_method_, to_pass_further);

                GlobalRecognizerParameter esf_param (param_.esf_config_xml_);
                typename GlobalRecognizer<PointT>::Ptr global_r (new GlobalRecognizer<PointT>( esf_param ));
                global_r->setFeatureEstimator( esf_estimator );
                global_r->setClassifier( classifier );
                global_recognition_pipeline->addRecognizer( global_r );
            }

            if (param_.do_alexnet_)
            {
                std::cerr << "Not implemented right now!" << std::endl;
            }

            typename RecognitionPipeline<PointT>::Ptr rec_pipeline_tmp = boost::static_pointer_cast<RecognitionPipeline<PointT> > (global_recognition_pipeline);
            mrec_->addRecognitionPipeline( rec_pipeline_tmp );
        }

        mrec_->setModelDatabase( model_database );
        mrec_->setNormalEstimator( normal_estimator_ );
        mrec_->initialize( models_dir_, retrain );
    }


    if(!skip_verification_)
    {
        // ====== SETUP HYPOTHESES VERIFICATION =====
        HV_Parameter paramHV (param_.hv_config_xml_);
        hv_.reset (new HypothesisVerification<PointT, PointT> (xtion, paramHV) );

        if( visualize_hv_go_cues )
            hv_->visualizeCues();
        if( visualize_hv_model_cues )
            hv_->visualizeModelCues();
        if( visualize_hv_pairwise_cues )
            hv_->visualizePairwiseCues();

        hv_->setModelDatabase(model_database);
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
        rec_vis_->setModelDatabase(model_database);
    }
}

template<typename PointT>
std::vector<typename ObjectHypothesis<PointT>::Ptr >
ObjectRecognizer<PointT>::recognize(const typename pcl::PointCloud<PointT>::ConstPtr &cloud)
{
    typename pcl::PointCloud<PointT>::Ptr processed_cloud (new pcl::PointCloud<PointT>(*cloud));
    //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
    processed_cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
    processed_cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);

    verified_hypotheses_.clear();

    std::vector<double> elapsed_time;

    pcl::PointCloud<pcl::Normal>::Ptr normals;
    if( mrec_->needNormals() || hv_ )
    {
        pcl::ScopeTime t("Computing normals");
        normal_estimator_->setInputCloud( cloud );
        normals.reset(new pcl::PointCloud<pcl::Normal>);
        normals = normal_estimator_->compute();
        mrec_->setSceneNormals( normals );
        elapsed_time.push_back( t.getTime() );
    }

    if(param_.remove_planes_)
    {
        cloud_segmenter_->setNormals( normals );
        cloud_segmenter_->segment( processed_cloud );
        processed_cloud = cloud_segmenter_->getProcessedCloud();
    }

    // ==== FILTER POINTS BASED ON DISTANCE =====
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud (processed_cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, param_.chop_z_);
    pass.setKeepOrganized(true);
    pass.filter (*processed_cloud);

    {
        pcl::ScopeTime t("Generation of object hypotheses");
        mrec_->setInputCloud ( processed_cloud );
        mrec_->recognize();
        generated_object_hypotheses_ = mrec_->getObjectHypothesis();
        elapsed_time.push_back( t.getTime() );
    }

    if(!skip_verification_)
    {
        pcl::ScopeTime t("Verification of object hypotheses");
        hv_->setSceneCloud( cloud );
        hv_->setNormals( normals );
        hv_->setHypotheses( generated_object_hypotheses_ );
        hv_->verify();
        verified_hypotheses_ = hv_->getVerifiedHypotheses();
        elapsed_time.push_back( t.getTime() );
    }

    for ( const typename ObjectHypothesis<PointT>::Ptr &voh : verified_hypotheses_ )
    {
        const std::string &model_id = voh->model_id_;
        const Eigen::Matrix4f &tf = voh->transform_;
        LOG(INFO) << "********************" << model_id << std::endl << tf << std::endl << std::endl;
    }

    if ( visualize_ )
    {
        LocalObjectModelDatabase::ConstPtr lomdb = local_recognition_pipeline_->getLocalObjectModelDatabase();
        rec_vis_->setCloud( cloud );
        rec_vis_->setProcessedCloud( processed_cloud );
        rec_vis_->setGeneratedObjectHypotheses( generated_object_hypotheses_ );
        rec_vis_->setLocalModelDatabase(lomdb);
        rec_vis_->setVerifiedObjectHypotheses( verified_hypotheses_ );
        rec_vis_->visualize();
    }

    return verified_hypotheses_;
}

template class V4R_EXPORTS ObjectRecognizer<pcl::PointXYZRGB>;

}

}
