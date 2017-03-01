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
#include <v4r/common/normals.h>
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
    std::string models_dir;
    std::string hv_config_xml = "cfg/hv_config.xml";
    std::string shot_config_xml = "cfg/shot_config.xml";
    std::string alexnet_config_xml  = "cfg/alexnet_config.xml";
    std::string esf_config_xml = "cfg/esf_config.xml";
    std::string camera_config_xml = "cfg/camera.xml";
    std::string depth_img_mask = "cfg/xtion_depth_mask.png";
    std::string sift_config_xml = "cfg/sift_config.xml";

    float cg_size = 0.01f; // Size for correspondence grouping.
    int cg_thresh = 7; // Threshold for correspondence grouping. The lower the more hypotheses are generated, the higher the more confident and accurate. Minimum 3.

    // pipeline setup
    bool do_sift = true;
    bool do_shot = false;
    bool do_esf = false;
    bool do_alexnet = false;
    int segmentation_method = SegmentationType::OrganizedConnectedComponents;
    int esf_classification_method = ClassifierType::SVM;

    bool visualize_hv_go_cues = false;
    bool visualize_hv_model_cues = false;
    bool visualize_hv_pairwise_cues = false;

    po::options_description desc("Single-View Object Instance Recognizer\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("model_dir,m", po::value<std::string>(&models_dir)->default_value(models_dir), "Models directory")
            ("chop_z,z", po::value<double>(&chop_z_)->default_value(chop_z_, boost::str(boost::format("%.2e") % chop_z_) ), "points with z-component higher than chop_z_ will be ignored (low chop_z reduces computation time and false positives (noise increase with z)")
            ("cg_thresh,c", po::value<int>(&cg_thresh)->default_value(cg_thresh), "Threshold for correspondence grouping. The lower the more hypotheses are generated, the higher the more confident and accurate. Minimum 3.")
            ("cg_size,g", po::value<float>(&cg_size)->default_value(cg_size, boost::str(boost::format("%.2e") % cg_size) ), "Size for correspondence grouping.")
            ("do_sift", po::value<bool>(&do_sift)->default_value(do_sift), "if true, enables SIFT feature matching")
            ("do_shot", po::value<bool>(&do_shot)->default_value(do_shot), "if true, enables SHOT feature matching")
            ("do_esf", po::value<bool>(&do_esf)->default_value(do_esf), "if true, enables ESF global matching")
            ("do_alexnet", po::value<bool>(&do_alexnet)->default_value(do_alexnet), "if true, enables AlexNet global matching")
            ("remove_planes", po::value<bool>(&remove_planes_)->default_value(remove_planes_), "if true, removes all planar surfaces with having at least min_plane_points points.")
            ("min_plane_points", po::value<size_t>(&min_plane_points_)->default_value(min_plane_points_), "Minimum number of plane points for plane to be removed (only if remove_planes is enabled).")
            ("segmentation_method", po::value<int>(&segmentation_method)->default_value(segmentation_method), "segmentation method (as stated in the V4R library (modules segmentation/types.h) ")
            ("esf_classification_method", po::value<int>(&esf_classification_method)->default_value(esf_classification_method), "ESF classification method (as stated in the V4R library (modules ml/types.h) ")
            ("depth_img_mask", po::value<std::string>(&depth_img_mask)->default_value(depth_img_mask), "filename for image registration mask. This mask tells which pixels in the RGB image can have valid depth pixels and which ones are not seen due to the phsysical displacement between RGB and depth sensor.")
            ("hv_config_xml", po::value<std::string>(&hv_config_xml)->default_value(hv_config_xml), "Filename of Hypotheses Verification XML configuration file.")
            ("sift_config_xml", po::value<std::string>(&sift_config_xml)->default_value(sift_config_xml), "Filename of SIFT XML configuration file.")
            ("shot_config_xml", po::value<std::string>(&shot_config_xml)->default_value(shot_config_xml), "Filename of SHOT XML configuration file.")
            ("alexnet_config_xml", po::value<std::string>(&alexnet_config_xml)->default_value(alexnet_config_xml), "Filename of Alexnet XML configuration file.")
            ("esf_config_xml", po::value<std::string>(&esf_config_xml)->default_value(esf_config_xml), "Filename of ESF XML configuration file.")
            ("camera_xml", po::value<std::string>(&camera_config_xml)->default_value(camera_config_xml), "Filename of camera parameter XML file.")
            ("visualize,v", po::bool_switch(&visualize_), "visualize recognition results")
            ("skip_verification", po::bool_switch(&skip_verification_), "if true, skips verification (only hypotheses generation)")
            ("hv_vis_cues", po::bool_switch(&visualize_hv_go_cues), "If set, visualizes cues computated at the hypothesis verification stage such as inlier, outlier points. Mainly used for debugging.")
            ("hv_vis_model_cues", po::bool_switch(&visualize_hv_model_cues), "If set, visualizes the model cues. Useful for debugging")
            ("hv_vis_pairwise_cues", po::bool_switch(&visualize_hv_pairwise_cues), "If set, visualizes the pairwise cues. Useful for debugging")
            ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(command_line_arguments).options(desc).allow_unregistered().run();
    std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; to_pass_further.push_back("-h"); }
    try { po::notify(vm); }
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;  }

    // ====== DEFINE CAMERA =======
    Camera::Ptr xtion (new Camera(camera_config_xml) );

    cv::Mat_<uchar> img_mask = cv::imread(depth_img_mask, CV_LOAD_IMAGE_GRAYSCALE);
    if( img_mask.data )
        xtion->setCameraDepthRegistrationMask( img_mask );
    else
        std::cout << "No camera depth registration mask provided. Assuming all pixels have valid depth." << std::endl;


    // ==== Fill object model database ==== ( assumes each object is in a seperate folder named after the object and contains and "views" folder with the training views of the object)
    typename Source<PointT>::Ptr model_database (new Source<PointT> (models_dir));


    // ====== SETUP MULTI PIPELINE RECOGNIZER ======
    mrec_.reset( new v4r::MultiRecognitionPipeline<PointT> );
    local_recognition_pipeline_.reset(new LocalRecognitionPipeline<PointT>);
    {
        // ====== SETUP LOCAL RECOGNITION PIPELINE =====
        if(do_sift || do_shot)
        {
            local_recognition_pipeline_->setModelDatabase( model_database );
            boost::shared_ptr< pcl::GeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> > gc_clusterer
                    (new pcl::GeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ>);
            gc_clusterer->setGCSize( cg_size );
            gc_clusterer->setGCThreshold( cg_thresh );
            local_recognition_pipeline_->setCGAlgorithm( gc_clusterer );

            if(do_sift)
            {
                LocalRecognizerParameter sift_param(sift_config_xml);
                typename LocalFeatureMatcher<PointT>::Ptr sift_rec (new LocalFeatureMatcher<PointT>(sift_param));
                typename SIFTLocalEstimation<PointT>::Ptr sift_est (new SIFTLocalEstimation<PointT>);
                sift_est->setMaxDistance(std::numeric_limits<float>::max());
                sift_rec->setFeatureEstimator( sift_est );
                local_recognition_pipeline_->addLocalFeatureMatcher(sift_rec);
            }
            if(do_shot)
            {
                typename SHOTLocalEstimation<PointT>::Ptr shot_est (new SHOTLocalEstimation<PointT>);
                typename UniformSamplingExtractor<PointT>::Ptr extr (new UniformSamplingExtractor<PointT>(0.02f));
                typename KeypointExtractor<PointT>::Ptr keypoint_extractor = boost::static_pointer_cast<KeypointExtractor<PointT> > (extr);

                LocalRecognizerParameter shot_param(shot_config_xml);
                typename LocalFeatureMatcher<PointT>::Ptr shot_rec (new LocalFeatureMatcher<PointT>(shot_param));
                shot_rec->addKeypointExtractor( keypoint_extractor );
                shot_rec->setFeatureEstimator( shot_est );
                local_recognition_pipeline_->addLocalFeatureMatcher(shot_rec);
            }

            typename RecognitionPipeline<PointT>::Ptr rec_pipeline_tmp = boost::static_pointer_cast<RecognitionPipeline<PointT> > (local_recognition_pipeline_);
            mrec_->addRecognitionPipeline(rec_pipeline_tmp);
        }

        // ====== SETUP GLOBAL RECOGNITION PIPELINE =====
        if(do_esf || do_alexnet)
        {
            typename GlobalRecognitionPipeline<PointT>::Ptr global_recognition_pipeline (new GlobalRecognitionPipeline<PointT>);
            typename v4r::Segmenter<PointT>::Ptr segmenter = v4r::initSegmenter<PointT>( segmentation_method, to_pass_further);
            global_recognition_pipeline->setSegmentationAlgorithm( segmenter );

            if(do_esf)
            {
                typename ESFEstimation<PointT>::Ptr esf_estimator (new ESFEstimation<PointT>);
                Classifier::Ptr classifier = initClassifier( esf_classification_method, to_pass_further);

                GlobalRecognizerParameter esf_param (esf_config_xml);
                typename GlobalRecognizer<PointT>::Ptr global_r (new GlobalRecognizer<PointT>( esf_param ));
                global_r->setFeatureEstimator( esf_estimator );
                global_r->setClassifier( classifier );
                global_recognition_pipeline->addRecognizer( global_r );
            }

            if (do_alexnet)
            {
                std::cerr << "Not implemented right now!" << std::endl;
            }

            typename RecognitionPipeline<PointT>::Ptr rec_pipeline_tmp = boost::static_pointer_cast<RecognitionPipeline<PointT> > (global_recognition_pipeline);
            mrec_->addRecognitionPipeline( rec_pipeline_tmp );
        }

        mrec_->setModelDatabase( model_database );
        mrec_->initialize( models_dir, false );
    }


    if(!skip_verification_)
    {
        // ====== SETUP HYPOTHESES VERIFICATION =====
        HV_Parameter paramHV (hv_config_xml);
        hv_.reset (new HypothesisVerification<PointT, PointT> (xtion, paramHV) );

        if( visualize_hv_go_cues )
            hv_->visualizeCues();
        if( visualize_hv_model_cues )
            hv_->visualizeModelCues();
        if( visualize_hv_pairwise_cues )
            hv_->visualizePairwiseCues();

        hv_->setModelDatabase(model_database);
    }

    if(visualize_)
    {
        rec_vis_.reset( new v4r::ObjectRecognitionVisualizer<PointT>);
        rec_vis_->setModelDatabase(model_database);
    }
}

template<typename PointT>
std::vector<typename ObjectHypothesis<PointT>::Ptr >
ObjectRecognizer<PointT>::recognize(typename pcl::PointCloud<PointT>::Ptr &cloud)
{
    //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
    cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
    cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);
    verified_hypotheses_.clear();

    std::vector<double> elapsed_time;

    pcl::PointCloud<pcl::Normal>::Ptr normals;
    if( mrec_->needNormals() || hv_)
    {
        pcl::ScopeTime t("Computing normals");
        normals.reset (new pcl::PointCloud<pcl::Normal>);
        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
        ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
        ne.setMaxDepthChangeFactor(0.02f);
        ne.setNormalSmoothingSize(10.0f);
        ne.setDepthDependentSmoothing(true);
        ne.setInputCloud(cloud);
        ne.compute(*normals);
        mrec_->setSceneNormals( normals );
        elapsed_time.push_back( t.getTime() );
    }


    if( remove_planes_ )
    {
        std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
        {
            pcl::ScopeTime t("Removing plane from input cloud.");
            pcl::OrganizedMultiPlaneSegmentation< PointT, pcl::Normal, pcl::Label > mps;
            mps.setMinInliers (min_plane_points_);
            mps.setAngularThreshold ( 2 * M_PI/180.f ); // 2 degrees
            mps.setDistanceThreshold ( 0.02 ); // 2cm
            mps.setInputNormals (normals);
            mps.setInputCloud (cloud);
            typename pcl::PlaneRefinementComparator<PointT, pcl::Normal, pcl::Label>::Ptr ref_comp (
            new pcl::PlaneRefinementComparator<PointT, pcl::Normal, pcl::Label> ());
            ref_comp->setDistanceThreshold ( 0.02, false);
            ref_comp->setAngularThreshold (2 * M_PI/180.f);
            mps.setRefinementComparator (ref_comp);
            mps.segment (regions);
        }

        pcl::visualization::PCLVisualizer vis;
        int vp1, vp2;
        vis.createViewPort(0, 0, 0.5, 1, vp1);
        vis.createViewPort(0.5, 0, 1, 1, vp2);
        vis.addPointCloud(cloud, "input", vp1);
        for (size_t i = 0; i < regions.size (); i++)
        {
          Eigen::Vector3f centroid = regions[i].getCentroid ();
          Eigen::Vector4f model = regions[i].getCoefficients ();

          std::vector<int> plane_inliers = get_all_plane_inliers( *cloud, model, 0.02 );

          typename pcl::PointCloud<PointT>::Ptr plane_cloud (new pcl::PointCloud<PointT>);
          pcl::copyPointCloud( *cloud, plane_inliers, *plane_cloud );

          vis.removeAllPointClouds(vp2);
          vis.addPointCloud(plane_cloud, "plane_cloud", vp2);
          vis.spin();

//          pcl::PointCloud boundary_cloud;
//          boundary_cloud.points = regions[i].getContour ();
//          printf ("Centroid: (%f, %f, %f)\n  Coefficients: (%f, %f, %f, %f)\n Inliers: %d\n",
//                  centroid[0], centroid[1], centroid[2],
//                  model[0], model[1], model[2], model[3],
//                  boundary_cloud.points.size ());
         }



//        boost::dynamic_bitset<> pt_is_accepted ( cloud->points.size() );
//        pt_is_accepted.set();

    }

    // ==== FILTER POINTS BASED ON DISTANCE =====
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, chop_z_);
    pass.setKeepOrganized(true);
    pass.filter (*cloud);

    {
        pcl::ScopeTime t("Generation of object hypotheses");
        mrec_->setInputCloud ( cloud );
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
