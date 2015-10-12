
#include <v4r/common/miscellaneous.h>
#include <v4r/features/opencv_sift_local_estimator.h>
#include <v4r/features/shot_local_estimator_omp.h>
#include <v4r/features/sift_local_estimator.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/ghv.h>
#include <v4r/recognition/local_recognizer.h>
#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/recognition/multiview_object_recognizer.h>
#include <v4r/recognition/recognizer.h>
#include <v4r/recognition/registered_views_source.h>

#include <pcl/common/centroid.h>
#include <pcl/console/parse.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/cloud_viewer.h>

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdlib.h>

#define USE_SIFT_GPU

class Rec
{
private:
    typedef pcl::PointXYZRGB PointT;
    typedef v4r::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;
    typedef pcl::Histogram<128> FeatureT;

    boost::shared_ptr<v4r::MultiRecognitionPipeline<PointT> > rr_;
    boost::shared_ptr<v4r::MultiviewRecognizer<PointT> > mv_r_;

    std::string test_dir_;
    bool visualize_;
    pcl::visualization::PCLVisualizer::Ptr vis_;

    cv::Ptr<SiftGPU> sift_;

public:

    Rec()
    {
        visualize_ = true;
    }

    bool initialize(int argc, char ** argv)
    {
        bool do_sift = true;
        bool do_shot = false;
        bool do_ourcvfh = false;

        float resolution = 0.005f;
        std::string models_dir, training_dir;

        v4r::GHV<PointT, PointT>::Parameter paramGHV;
        v4r::GraphGeometricConsistencyGrouping<PointT, PointT>::Parameter paramGgcg;
        v4r::LocalRecognitionPipeline<flann::L1, PointT, FeatureT >::Parameter paramLocalRecSift;
        v4r::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> >::Parameter paramLocalRecShot;
        v4r::MultiRecognitionPipeline<PointT>::Parameter paramMultiPipeRec;
        v4r::SHOTLocalEstimationOMP<PointT, pcl::Histogram<352> >::Parameter paramLocalEstimator;
        v4r::MultiviewRecognizer<PointT>::Parameter paramMultiView;

        paramGgcg.gc_size_ = 0.015f;
        paramGgcg.thres_dot_distance_ = 0.2f;
        paramGgcg.dist_for_cluster_factor_ = 0;
        paramGgcg.max_taken_correspondence_ = 2;
        paramGgcg.max_time_allowed_cliques_comptutation_ = 100;

        paramGHV.eps_angle_threshold_ = 0.1f;
        paramGHV.min_points_ = 100;
        paramGHV.cluster_tolerance_ = 0.01f;
        paramGHV.use_histogram_specification_ = true;
        paramGHV.w_occupied_multiple_cm_ = 0.f;
        paramGHV.opt_type_ = 0;
//        paramGHV.active_hyp_penalty_ = 0.f;
        paramGHV.regularizer_ = 3;
        paramGHV.color_sigma_ab_ = 0.5f;
        paramGHV.radius_normals_ = 0.02f;
        paramGHV.occlusion_thres_ = 0.01f;
        paramGHV.inliers_threshold_ = 0.015f;

        paramLocalRecSift.use_cache_ = paramLocalRecShot.use_cache_ = true;
        paramLocalRecSift.icp_iterations_ = paramLocalRecShot.icp_iterations_ = 10;
        paramLocalRecSift.save_hypotheses_ = paramLocalRecShot.save_hypotheses_ = true;
        paramLocalRecShot.kdtree_splits_ = 128;

        pcl::console::parse_argument (argc, argv,  "-visualize", visualize_);
        pcl::console::parse_argument (argc, argv,  "-test_dir", test_dir_);
        pcl::console::parse_argument (argc, argv,  "-models_dir", models_dir);
        pcl::console::parse_argument (argc, argv,  "-training_dir", training_dir);
        pcl::console::parse_argument (argc, argv,  "-do_sift", do_sift);
        pcl::console::parse_argument (argc, argv,  "-do_shot", do_shot);
        pcl::console::parse_argument (argc, argv,  "-do_ourcvfh", do_ourcvfh);
        pcl::console::parse_argument (argc, argv,  "-knn_sift", paramLocalRecSift.knn_);
        pcl::console::parse_argument (argc, argv,  "-knn_shot", paramLocalRecShot.knn_);

        int normal_computation_method;
        if(pcl::console::parse_argument (argc, argv,  "-normal_method", normal_computation_method) != -1)
        {
            paramLocalRecSift.normal_computation_method_ =
                    paramLocalRecShot.normal_computation_method_ =
                    paramMultiPipeRec.normal_computation_method_ =
                    paramLocalEstimator.normal_computation_method_ =
                    normal_computation_method;
        }

        int icp_iterations;
        if(pcl::console::parse_argument (argc, argv,  "-icp_iterations", icp_iterations) != -1)
            paramLocalRecSift.icp_iterations_ = paramLocalRecShot.icp_iterations_ = paramMultiPipeRec.icp_iterations_ = icp_iterations;

        pcl::console::parse_argument (argc, argv,  "-chop_z", paramMultiView.chop_z_ );
        pcl::console::parse_argument (argc, argv,  "-max_vertices_in_graph", paramMultiView.max_vertices_in_graph_ );

        pcl::console::parse_argument (argc, argv,  "-cg_size_thresh", paramGgcg.gc_threshold_);
        pcl::console::parse_argument (argc, argv,  "-cg_size", paramGgcg.gc_size_);
        pcl::console::parse_argument (argc, argv,  "-cg_ransac_threshold", paramGgcg.ransac_threshold_);
        pcl::console::parse_argument (argc, argv,  "-cg_dist_for_clutter_factor", paramGgcg.dist_for_cluster_factor_);
        pcl::console::parse_argument (argc, argv,  "-cg_max_taken", paramGgcg.max_taken_correspondence_);
        pcl::console::parse_argument (argc, argv,  "-cg_max_time_for_cliques_computation", paramGgcg.max_time_allowed_cliques_comptutation_);
        pcl::console::parse_argument (argc, argv,  "-cg_dot_distance", paramGgcg.thres_dot_distance_);
        pcl::console::parse_argument (argc, argv,  "-use_cg_graph", paramGgcg.use_graph_);
        pcl::console::parse_argument (argc, argv,  "-hv_clutter_regularizer", paramGHV.clutter_regularizer_);
        pcl::console::parse_argument (argc, argv,  "-hv_color_sigma_ab", paramGHV.color_sigma_ab_);
        pcl::console::parse_argument (argc, argv,  "-hv_color_sigma_l", paramGHV.color_sigma_l_);
        pcl::console::parse_argument (argc, argv,  "-hv_detect_clutter", paramGHV.detect_clutter_);
        pcl::console::parse_argument (argc, argv,  "-hv_duplicity_cm_weight", paramGHV.w_occupied_multiple_cm_);
        pcl::console::parse_argument (argc, argv,  "-hv_histogram_specification", paramGHV.use_histogram_specification_);
        pcl::console::parse_argument (argc, argv,  "-hv_hyp_penalty", paramGHV.active_hyp_penalty_);
        pcl::console::parse_argument (argc, argv,  "-hv_ignore_color", paramGHV.ignore_color_even_if_exists_);
        pcl::console::parse_argument (argc, argv,  "-hv_initial_status", paramGHV.initial_status_);
        pcl::console::parse_argument (argc, argv,  "-hv_inlier_threshold", paramGHV.inliers_threshold_);
        pcl::console::parse_argument (argc, argv,  "-hv_occlusion_threshold", paramGHV.occlusion_thres_);
        pcl::console::parse_argument (argc, argv,  "-hv_optimizer_type", paramGHV.opt_type_);
        pcl::console::parse_argument (argc, argv,  "-hv_radius_clutter", paramGHV.radius_neighborhood_clutter_);
        pcl::console::parse_argument (argc, argv,  "-hv_radius_normals", paramGHV.radius_normals_);
        pcl::console::parse_argument (argc, argv,  "-hv_regularizer", paramGHV.regularizer_);
//        pcl::console::parse_argument (argc, argv,  "-hv_requires_normals", r_.hv_params_.requires_normals_);

        rr_.reset(new v4r::MultiRecognitionPipeline<PointT>(paramMultiPipeRec));

        boost::shared_ptr < v4r::GraphGeometricConsistencyGrouping<PointT, PointT> > gcg_alg (
                    new v4r::GraphGeometricConsistencyGrouping<PointT, PointT> (paramGgcg));

        boost::shared_ptr <v4r::Source<PointT> > cast_source;
        if (do_sift || do_shot ) // for local recognizers we need this source type / training data
        {
            boost::shared_ptr < v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > src
                    (new v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT>);
            src->setPath (models_dir);
            src->setModelStructureDir (training_dir);
            src->generate ();
            src->createVoxelGridAndDistanceTransform(resolution);
            cast_source = boost::static_pointer_cast<v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > (src);
        }

        if (do_sift)
        {
#ifdef USE_SIFT_GPU
        static char kw[][16] = {"-m", "-fo", "-1", "-s", "-v", "1", "-pack"};
        char * argvv[] = {kw[0], kw[1], kw[2], kw[3],kw[4],kw[5],kw[6], NULL};

        int argcc = sizeof(argvv) / sizeof(char*);
        sift_ = new SiftGPU ();
        sift_->ParseParam (argcc, argvv);

        //create an OpenGL context for computation
        if (sift_->CreateContextGL () != SiftGPU::SIFTGPU_FULL_SUPPORTED)
          throw std::runtime_error ("PSiftGPU::PSiftGPU: No GL support!");

      boost::shared_ptr < v4r::SIFTLocalEstimation<PointT, FeatureT > > estimator (new v4r::SIFTLocalEstimation<PointT, FeatureT >(sift_));
      boost::shared_ptr < v4r::LocalEstimator<PointT, FeatureT > > cast_estimator = boost::dynamic_pointer_cast<v4r::SIFTLocalEstimation<PointT, FeatureT > > (estimator);
#else
      boost::shared_ptr < v4r::OpenCVSIFTLocalEstimation<PointT, FeatureT > > estimator (new v4r::OpenCVSIFTLocalEstimation<PointT, FeatureT >);
      boost::shared_ptr < v4r::LocalEstimator<PointT, FeatureT > > cast_estimator = boost::dynamic_pointer_cast<v4r::OpenCVSIFTLocalEstimation<PointT, FeatureT > > (estimator);
#endif

            boost::shared_ptr<v4r::LocalRecognitionPipeline<flann::L1, PointT, FeatureT > > sift_r;
            sift_r.reset (new v4r::LocalRecognitionPipeline<flann::L1, PointT, FeatureT > (paramLocalRecSift));
            sift_r->setDataSource (cast_source);
            sift_r->setTrainingDir (training_dir);
            sift_r->setFeatureEstimator (cast_estimator);
            sift_r->initialize (false);

            boost::shared_ptr < v4r::Recognizer<PointT> > cast_recog;
            cast_recog = boost::static_pointer_cast<v4r::LocalRecognitionPipeline<flann::L1, PointT, FeatureT > > (sift_r);
            std::cout << "Feature Type: " << cast_recog->getFeatureType() << std::endl;
            rr_->addRecognizer (cast_recog);
        }
        if (do_shot)
        {
            boost::shared_ptr<v4r::UniformSamplingExtractor<PointT> > uniform_kp_extractor ( new v4r::UniformSamplingExtractor<PointT>);
            uniform_kp_extractor->setSamplingDensity (0.01f);
            uniform_kp_extractor->setFilterPlanar (true);
            uniform_kp_extractor->setThresholdPlanar(0.1);
            uniform_kp_extractor->setMaxDistance( 1000.0 ); // for training we want to consider all points (except nan values)

            boost::shared_ptr<v4r::KeypointExtractor<PointT> > keypoint_extractor = boost::static_pointer_cast<v4r::KeypointExtractor<PointT> > (uniform_kp_extractor);
            boost::shared_ptr<v4r::SHOTLocalEstimationOMP<PointT, pcl::Histogram<352> > > estimator (new v4r::SHOTLocalEstimationOMP<PointT, pcl::Histogram<352> >(paramLocalEstimator));
            estimator->addKeypointExtractor (keypoint_extractor);

            boost::shared_ptr<v4r::LocalEstimator<PointT, pcl::Histogram<352> > > cast_estimator;
            cast_estimator = boost::dynamic_pointer_cast<v4r::LocalEstimator<PointT, pcl::Histogram<352> > > (estimator);

            boost::shared_ptr<v4r::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> > > local;
            local.reset(new v4r::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> > (paramLocalRecShot));
            local->setDataSource (cast_source);
            local->setTrainingDir(training_dir);
            local->setFeatureEstimator (cast_estimator);
            local->initialize (false);

            uniform_kp_extractor->setMaxDistance( paramMultiView.chop_z_ ); // for training we do not want this restriction

            boost::shared_ptr<v4r::Recognizer<PointT> > cast_recog;
            cast_recog = boost::static_pointer_cast<v4r::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> > > (local);
            std::cout << "Feature Type: " << cast_recog->getFeatureType() << std::endl;
            rr_->addRecognizer(cast_recog);
        }


        boost::shared_ptr<v4r::GHV<PointT, PointT> > hyp_verification_method (new v4r::GHV<PointT, PointT>(paramGHV));
        boost::shared_ptr<v4r::HypothesisVerification<PointT,PointT> > cast_hyp_pointer = boost::static_pointer_cast<v4r::GHV<PointT, PointT> > (hyp_verification_method);
        rr_->setHVAlgorithm( cast_hyp_pointer );
        rr_->setCGAlgorithm( gcg_alg );

        boost::shared_ptr<v4r::Recognizer<PointT> > cast_recog  = boost::static_pointer_cast<v4r::MultiRecognitionPipeline<PointT> > (rr_);

        mv_r_.reset(new v4r::MultiviewRecognizer<PointT>(paramMultiView));
        mv_r_->setSingleViewRecognizer(cast_recog);
        mv_r_->set_sift(sift_);

        return true;
    }

    bool test()
    {
        std::vector< std::string> sub_folder_names;
        if(!v4r::io::getFoldersInDirectory( test_dir_, "", sub_folder_names) )
        {
            std::cerr << "No subfolders in directory " << test_dir_ << ". " << std::endl;
            sub_folder_names.push_back("");
        }

        std::sort(sub_folder_names.begin(), sub_folder_names.end());
        for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
        {
            const std::string sequence_path = test_dir_ + "/" + sub_folder_names[ sub_folder_id ];

            std::vector< std::string > views;
            v4r::io::getFilesInDirectory(sequence_path, views, "", ".*.pcd", false);
            std::sort(views.begin(), views.end());
            for (size_t v_id=0; v_id<views.size(); v_id++)
            {
                const std::string fn = test_dir_ + "/" + sub_folder_names[sub_folder_id] + "/" + views[ v_id ];

                std::cout << "Recognizing file " << fn << std::endl;
                pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
                pcl::io::loadPCDFile(fn, *cloud);

                pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>());

                mv_r_->setInputCloud (cloud);
                mv_r_->setCameraPose(Eigen::Matrix4f::Identity());
                mv_r_->recognize();

                std::vector<ModelTPtr> verified_models = rr_->getVerifiedModels();
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_verified;
                transforms_verified = rr_->getVerifiedTransforms();

//                if (visualize_)
//                    rr_->visualize();

                for(size_t m_id=0; m_id<verified_models.size(); m_id++)
                {
                    const std::string &model_id = verified_models[m_id]->id_;
                    const Eigen::Matrix4f &tf = transforms_verified[m_id];
                    std::cout << "******" << model_id << std::endl << tf << std::endl << std::endl;
                }
            }
        }
        return true;
    }
};

int
main (int argc, char ** argv)
{
    srand (time(NULL));
    Rec r_eval;
    r_eval.initialize(argc,argv);
    r_eval.test();
    return 0;
}
