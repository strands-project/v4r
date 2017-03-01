#include <iostream>
#include <sstream>

#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <pcl/common/time.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/filters/passthrough.h>

#include <v4r/common/camera.h>
#include <v4r/common/normals.h>
#include <v4r/io/filesystem.h>
#include <v4r/features/esf_estimator.h>
#include <v4r/features/shot_local_estimator.h>
#include <v4r/features/sift_local_estimator.h>
#include <v4r/keypoints/uniform_sampling_extractor.h>
#include <v4r/ml/nearestNeighbor.h>
#include <v4r/ml/svmWrapper.h>
#include <v4r/recognition/local_recognition_pipeline.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/recognition/global_recognition_pipeline.h>
#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/recognition/multiview_recognizer.h>
#include <v4r/segmentation/all_headers.h>

#include "visualization.h"

#include <sys/time.h>
#include <sys/resource.h>

namespace po = boost::program_options;
using namespace v4r;

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;

    std::string test_dir;
    std::string out_dir = "/tmp/object_recognition_results/";
    std::string debug_dir = "";
    bool visualize = false;

    // model database folder structure
    std::string models_dir;
    std::string cloud_fn_prefix;
    std::string indices_fn_prefix;
    std::string pose_fn_prefix;
    std::string transformation_fn;

    // pipeline setup
    bool do_sift = true;
    bool do_shot = true;
    bool do_esf = true;
    bool do_alexnet = false;
    double chop_z = std::numeric_limits<double>::max();
    std::string hv_config_xml = "cfg/hv_config.xml";
    std::string sift_config_xml = "cfg/sift_config.xml";
    std::string shot_config_xml = "cfg/shot_config.xml";
    std::string alexnet_config_xml  = "cfg/alexnet_config.xml";
    std::string esf_config_xml = "cfg/esf_config.xml";
    std::string camera_config_xml = "cfg/camera.xml";
    std::string depth_img_mask = "cfg/xtion_depth_mask.png";
    int segmentation_method = SegmentationType::OrganizedConnectedComponents;

    // Correspondence grouping parameters for local recognition pipeline
    float cg_size = 0.01f; // Size for correspondence grouping.
    int cg_thresh = 7; // Threshold for correspondence grouping. The lower the more hypotheses are generated, the higher the more confident and accurate. Minimum 3.

    google::InitGoogleLogging(argv[0]);

    HV_Parameter paramHV (hv_config_xml);   ///TODO: this does not allow to read hv_config_xml filename from console

    po::options_description desc("Single-View Object Instance Recognizer\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("model_dir,m", po::value<std::string>(&models_dir)->default_value(models_dir), "Models directory")
            ("test_dir,t", po::value<std::string>(&test_dir)->required(), "Directory with test scenes stored as point clouds (.pcd). The camera pose is taken directly from the pcd header fields \"sensor_orientation_\" and \"sensor_origin_\" (if the test directory contains subdirectories, each subdirectory is considered as seperate sequence for multiview recognition)")
            ("cloud_fn_prefix", po::value<std::string>(&cloud_fn_prefix)->default_value("cloud_"), "Prefix of cloud filename")
            ("indices_fn_prefix", po::value<std::string>(&indices_fn_prefix)->default_value("object_indices_"), "Prefix of object indices filename")
            ("transformation_fn", po::value<std::string>(&transformation_fn)->default_value(""), "Transformation to apply to each pose")
            ("pose_fn_prefix", po::value<std::string>(&pose_fn_prefix)->default_value("pose_"), "Prefix for the output pose filename")
            ("chop_z,z", po::value<double>(&chop_z)->default_value(chop_z, boost::str(boost::format("%.2e") % chop_z) ), "points with z-component higher than chop_z_ will be ignored (low chop_z reduces computation time and false positives (noise increase with z)")
            ("cg_thresh,c", po::value<int>(&cg_thresh)->default_value(cg_thresh), "Threshold for correspondence grouping. The lower the more hypotheses are generated, the higher the more confident and accurate. Minimum 3.")
            ("cg_size,g", po::value<float>(&cg_size)->default_value(cg_size, boost::str(boost::format("%.2e") % cg_size) ), "Size for correspondence grouping.")
            ("do_sift", po::value<bool>(&do_sift)->default_value(do_sift), "if true, enables SIFT feature matching")
            ("do_shot", po::value<bool>(&do_shot)->default_value(do_shot), "if true, enables SHOT feature matching")
            ("do_esf", po::value<bool>(&do_esf)->default_value(do_esf), "if true, enables ESF global matching")
            ("do_alexnet", po::value<bool>(&do_alexnet)->default_value(do_alexnet), "if true, enables AlexNet global matching")
            ("segmentation_method", po::value<int>(&segmentation_method)->default_value(segmentation_method), "segmentation method (as stated in the V4R library (modules segmentation/types.h) ")
            ("depth_img_mask", po::value<std::string>(&depth_img_mask)->default_value(depth_img_mask), "filename for image registration mask. This mask tells which pixels in the RGB image can have valid depth pixels and which ones are not seen due to the phsysical displacement between RGB and depth sensor.")
//            ("hv_config_xml", po::value<std::string>(&hv_config_xml)->default_value(hv_config_xml), "Filename of Hypotheses Verification XML configuration file.")
            ("sift_config_xml", po::value<std::string>(&sift_config_xml)->default_value(sift_config_xml), "Filename of SIFT XML configuration file.")
            ("shot_config_xml", po::value<std::string>(&shot_config_xml)->default_value(shot_config_xml), "Filename of SHOT XML configuration file.")
            ("alexnet_config_xml", po::value<std::string>(&alexnet_config_xml)->default_value(alexnet_config_xml), "Filename of Alexnet XML configuration file.")
            ("esf_config_xml", po::value<std::string>(&esf_config_xml)->default_value(esf_config_xml), "Filename of ESF XML configuration file.")
            ("camera_xml", po::value<std::string>(&camera_config_xml)->default_value(camera_config_xml), "Filename of camera parameter XML file.")
            ("visualize,v", po::bool_switch(&visualize), "visualize recognition results")
            ("out_dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "Output directory where recognition results will be stored.")
            ("dbg_dir", po::value<std::string>(&debug_dir)->default_value(debug_dir), "Output directory where debug information (generated object hypotheses) will be stored (skipped if empty)")
            ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
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
    Source<PointT>::Ptr model_database (new Source<PointT> (models_dir));


    // ====== SETUP MULTI-VIEW OBJECT RECOGNIZER =======
    MultiviewRecognizer<PointT>::Ptr mv_rec (new MultiviewRecognizer<PointT>);
    {
        // ====== SETUP MULTI PIPELINE RECOGNIZER ======
        MultiRecognitionPipeline<PointT>::Ptr mrec (new MultiRecognitionPipeline<PointT>);
        LocalRecognitionPipeline<PointT>::Ptr local_recognition_pipeline (new LocalRecognitionPipeline<PointT>);
        {
            // ====== SETUP LOCAL RECOGNITION PIPELINE =====
            if(do_sift || do_shot)
            {
                local_recognition_pipeline->setModelDatabase( model_database );
                boost::shared_ptr< pcl::GeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> > gc_clusterer
                        (new pcl::GeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ>);
                gc_clusterer->setGCSize( cg_size );
                gc_clusterer->setGCThreshold( cg_thresh );
                local_recognition_pipeline->setCGAlgorithm( gc_clusterer );

                if(do_sift)
                {
                    LocalRecognizerParameter sift_param(sift_config_xml);
                    LocalFeatureMatcher<PointT>::Ptr sift_rec (new LocalFeatureMatcher<PointT>(sift_param));
                    SIFTLocalEstimation<PointT>::Ptr sift_est (new SIFTLocalEstimation<PointT>);
                    sift_est->setMaxDistance(std::numeric_limits<float>::max());
                    sift_rec->setFeatureEstimator( sift_est );
                    local_recognition_pipeline->addLocalFeatureMatcher(sift_rec);
                }
                if(do_shot)
                {
                    SHOTLocalEstimation<PointT>::Ptr shot_est (new SHOTLocalEstimation<PointT>);
                    UniformSamplingExtractor<PointT>::Ptr extr (new UniformSamplingExtractor<PointT>(0.02f));
                    KeypointExtractor<PointT>::Ptr keypoint_extractor = boost::static_pointer_cast<KeypointExtractor<PointT> > (extr);

                    LocalRecognizerParameter shot_param(shot_config_xml);
                    LocalFeatureMatcher<PointT>::Ptr shot_rec (new LocalFeatureMatcher<PointT>(shot_param));
                    shot_rec->addKeypointExtractor( keypoint_extractor );
                    shot_rec->setFeatureEstimator( shot_est );
                    local_recognition_pipeline->addLocalFeatureMatcher(shot_rec);
                }

                RecognitionPipeline<PointT>::Ptr rec_pipeline_tmp = boost::static_pointer_cast<RecognitionPipeline<PointT> > (local_recognition_pipeline);
                mrec->addRecognitionPipeline(rec_pipeline_tmp);
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

                    // choose classifier
    //                 NearestNeighborClassifier::Ptr classifier (new NearestNeighborClassifier);
                    svmClassifier::Ptr classifier (new svmClassifier);
    //                classifier->setInFilename(esf_svm_model_fn);

                    GlobalRecognizerParameter esf_param (esf_config_xml);
                    typename GlobalRecognizer<PointT>::Ptr global_r (new GlobalRecognizer<PointT>(esf_param));
                    global_r->setFeatureEstimator(esf_estimator);
                    global_r->setClassifier(classifier);
                    global_recognition_pipeline->addRecognizer(global_r);
                }

                if (do_alexnet)
                {
                    std::cerr << "Not implemented right now!" << std::endl;
                }

                RecognitionPipeline<PointT>::Ptr rec_pipeline_tmp = boost::static_pointer_cast<RecognitionPipeline<PointT> > (global_recognition_pipeline);
                mrec->addRecognitionPipeline( rec_pipeline_tmp );
            }

            mrec->setModelDatabase( model_database );
            mrec->initialize( models_dir, false );
        }
        RecognitionPipeline<PointT>::Ptr rec_pipeline_tmp = boost::static_pointer_cast<RecognitionPipeline<PointT> > (mrec);
        mv_rec->setSingleViewRecognitionPipeline( rec_pipeline_tmp );
    }


    // ====== SETUP HYPOTHESES VERIFICATION =====
    HypothesisVerification<PointT, PointT>::Ptr hv (new HypothesisVerification<PointT, PointT> (xtion, paramHV) );
    hv->setModelDatabase(model_database);

    // ====== TEST RECOGNIZER ===================
    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( test_dir );
    if(sub_folder_names.empty()) sub_folder_names.push_back("");

    ObjectRecognitionVisualizer<PointT> rec_vis;
    rec_vis.setModelDatabase(model_database);

    for (const std::string &sub_folder_name : sub_folder_names)
    {
        std::vector< std::string > views = v4r::io::getFilesInDirectory( test_dir+"/"+sub_folder_name, ".*.pcd", false );
        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            bf::path test_path = test_dir;
            test_path /= sub_folder_name;
            test_path /= views[v_id];

            std::vector<double> elapsed_time;

            LOG(INFO) << "Recognizing file " << test_path.string();
            pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
            pcl::io::loadPCDFile( test_path.string(), *cloud);

            //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
            cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
            cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);
            pcl::PointCloud<pcl::Normal>::Ptr normals;

            if( mv_rec->needNormals() || hv)
            {
                pcl::ScopeTime t("Computing normals");
                normals.reset (new pcl::PointCloud<pcl::Normal>);
                pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
                ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
                ne.setMaxDepthChangeFactor(0.02f);
                ne.setNormalSmoothingSize(10.0f);
                ne.setInputCloud(cloud);
                ne.compute(*normals);
                mv_rec->setSceneNormals( normals );
                elapsed_time.push_back( t.getTime() );
            }

            // ==== FILTER POINTS BASED ON DISTANCE =====
            pcl::PassThrough<PointT> pass;
            pass.setInputCloud (cloud);
            pass.setFilterFieldName ("z");
            pass.setFilterLimits (0, chop_z);
            pass.setKeepOrganized(true);
            pass.filter (*cloud);

            std::vector<ObjectHypothesesGroup<PointT> > generated_object_hypotheses;
            std::vector<typename ObjectHypothesis<PointT>::Ptr > verified_hypotheses;
            {
                pcl::ScopeTime t("Generation of object hypotheses");
                mv_rec->addView( cloud );
                mv_rec->recognize();
                generated_object_hypotheses = mv_rec->getObjectHypothesis();
                elapsed_time.push_back( t.getTime() );
            }

            {
                pcl::ScopeTime t("Verification of object hypotheses");
                hv->setSceneCloud( cloud );
                hv->setNormals( normals );
                hv->setHypotheses( generated_object_hypotheses );
                hv->verify();
                verified_hypotheses = hv->getVerifiedHypotheses();
                elapsed_time.push_back( t.getTime() );
            }

            for ( const ObjectHypothesis<PointT>::Ptr &voh : verified_hypotheses )
            {
                const std::string &model_id = voh->model_id_;
                const Eigen::Matrix4f &tf = voh->transform_;
                LOG(INFO) << "********************" << model_id << std::endl << tf << std::endl << std::endl;
            }

            if ( !out_dir.empty() )  // write results to disk (for each verified hypothesis add a row in the text file with object name, dummy confidence value and object pose in row-major order)
            {
                std::string out_basename = views[v_id];
                boost::replace_last(out_basename, ".pcd", ".anno");
                bf::path out_path = out_dir;
                out_path /= sub_folder_name;
                out_path /= out_basename;

                v4r::io::createDirForFileIfNotExist(out_path.string());

                std::ofstream f ( out_path.string().c_str() );
                for ( const ObjectHypothesis<PointT>::Ptr &voh : verified_hypotheses )
                {
                    f << voh->model_id_ << " (-1.): ";
                    for (size_t row=0; row <4; row++)
                        for(size_t col=0; col<4; col++)
                            f << voh->transform_(row, col) << " ";
                    f << std::endl;
                }
                f.close();
            }


            if ( visualize )
            {
//                LocalObjectModelDatabase::ConstPtr lomdb = local_recognition_pipeline->getLocalObjectModelDatabase();
                rec_vis.setCloud( cloud );
                rec_vis.setGeneratedObjectHypotheses( generated_object_hypotheses );
//                rec_vis.setLocalModelDatabase(lomdb);
                rec_vis.setVerifiedObjectHypotheses( verified_hypotheses );
                rec_vis.visualize();
            }
        }
    }
}

