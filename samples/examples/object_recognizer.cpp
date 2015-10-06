#include <v4r/common/miscellaneous.h>
#include <v4r/features/opencv_sift_local_estimator.h>
#include <v4r/features/shot_local_estimator_omp.h>
#include <v4r/features/sift_local_estimator.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/ghv.h>
#include <v4r/recognition/local_recognizer.h>
#include <v4r/recognition/multi_pipeline_recognizer.h>
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

class Recognizer
{
private:
    typedef pcl::PointXYZRGB PointT;
    typedef v4r::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;
    typedef pcl::Histogram<128> FeatureT;

    v4r::MultiRecognitionPipeline<PointT> rr_;
    std::string test_dir_;
    bool visualize_;
    pcl::visualization::PCLVisualizer::Ptr vis_;
    cv::Ptr<SiftGPU> sift_;

public:
    class Parameter
    {
    public:
        bool do_sift_;
        bool do_shot_;
        bool do_ourcvfh_;
        int knn_sift_;
        int knn_shot_;
        float chop_z_;
        int normal_computation_method_;

        Parameter (
                bool do_sift = true,
                bool do_shot = false,
                bool do_ourcvfh = false,
                int knn_sift = 5,
                int knn_shot = 1,
                float chop_z = std::numeric_limits<float>::max(),
                int normal_computation_method = 2)
            : do_sift_ (do_sift),
              do_shot_ (do_shot),
              do_ourcvfh_ (do_ourcvfh),
              knn_sift_ (knn_sift),
              knn_shot_ (knn_shot),
              chop_z_ (chop_z),
              normal_computation_method_ (normal_computation_method)
        {}
    };

    Recognizer( const Parameter &p = Parameter())
    {
        param_ = p;
        visualize_ = true;
    }


    Parameter param_;

    void visualize_result(const pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector<ModelTPtr> &models, const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms)
    {        
//        r_.visualize();

        if(!vis_)
            vis_.reset ( new pcl::visualization::PCLVisualizer("Recognition Results") );
        vis_->removeAllPointClouds();
        vis_->removeAllShapes();
        vis_->addPointCloud(cloud, "cloud");

        for(size_t m_id=0; m_id<models.size(); m_id++)
        {
            const std::string model_id = models[m_id]->id_.substr(0, models[m_id]->id_.length() - 4);
            std::stringstream model_text;
            model_text << model_id << "_" << m_id;
            pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
            pcl::PointCloud<PointT>::ConstPtr model_cloud = models[m_id]->getAssembled( 0.003f );
            pcl::transformPointCloud( *model_cloud, *model_aligned, transforms[m_id]);

            //PointT centroid;
            //pcl::computeCentroid(*model_aligned, centroid);
            //centroid.x += cloud->sensor_origin_[0];
            //centroid.y += cloud->sensor_origin_[1];
            //centroid.z += cloud->sensor_origin_[2];
            //const float r=50+rand()%205;
            //const float g=50+rand()%205;
            //const float b=50+rand()%205;
            //vis_->addText3D(model_text.str(), centroid, 0.01, r/255, g/255, b/255);

            model_aligned->sensor_orientation_ = cloud->sensor_orientation_;
            model_aligned->sensor_origin_ = cloud->sensor_origin_;
            vis_->addPointCloud(model_aligned, model_text.str());
        }
        vis_->spin();
    }

    bool initialize(int argc, char ** argv)
    {
        int icp_iterations = 10;
        float resolution = 0.005f;
        std::string models_dir, training_dir;

        v4r::GHV<PointT, PointT>::Parameter paramGHV;
        v4r::GraphGeometricConsistencyGrouping<PointT, PointT>::Parameter paramGgcg;

        paramGgcg.gc_size_ = 0.015f;
        paramGgcg.thres_dot_distance_ = 0.2f;
        paramGgcg.dist_for_cluster_factor_ = 0;
        paramGgcg.max_taken_correspondence_ = 2;
        paramGgcg.max_time_allowed_cliques_comptutation_ = 100;

        pcl::console::parse_argument (argc, argv,  "-visualize", visualize_);
        pcl::console::parse_argument (argc, argv,  "-test_dir", test_dir_);

        pcl::console::parse_argument (argc, argv,  "-models_dir", models_dir);
        pcl::console::parse_argument (argc, argv,  "-training_dir", training_dir);

        pcl::console::parse_argument (argc, argv,  "-chop_z", param_.chop_z_ );
        pcl::console::parse_argument (argc, argv,  "-icp_iterations", icp_iterations);
        pcl::console::parse_argument (argc, argv,  "-do_sift", param_.do_sift_);
        pcl::console::parse_argument (argc, argv,  "-do_shot", param_.do_shot_);
        pcl::console::parse_argument (argc, argv,  "-do_ourcvfh", param_.do_ourcvfh_);
        pcl::console::parse_argument (argc, argv,  "-knn_sift", param_.knn_sift_);
        pcl::console::parse_argument (argc, argv,  "-knn_shot", param_.knn_shot_);

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
//        pcl::console::parse_argument (argc, argv,  "-hv_initial_status", r_.hv_params_.initial_status_);
        pcl::console::parse_argument (argc, argv,  "-hv_inlier_threshold", paramGHV.inliers_threshold_);
        pcl::console::parse_argument (argc, argv,  "-hv_occlusion_threshold", paramGHV.occlusion_thres_);
        pcl::console::parse_argument (argc, argv,  "-hv_optimizer_type", paramGHV.opt_type_);
        pcl::console::parse_argument (argc, argv,  "-hv_radius_clutter", paramGHV.radius_neighborhood_clutter_);
        pcl::console::parse_argument (argc, argv,  "-hv_radius_normals", paramGHV.radius_normals_);
        pcl::console::parse_argument (argc, argv,  "-hv_regularizer", paramGHV.regularizer_);
//        pcl::console::parse_argument (argc, argv,  "-hv_requires_normals", r_.hv_params_.requires_normals_);

        boost::shared_ptr < v4r::GraphGeometricConsistencyGrouping<PointT, PointT> > gcg_alg (
                    new v4r::GraphGeometricConsistencyGrouping<PointT, PointT> (paramGgcg));

        boost::shared_ptr < v4r::CorrespondenceGrouping<PointT, PointT> > cast_cg_alg_;
        cast_cg_alg_ = boost::static_pointer_cast<v4r::CorrespondenceGrouping<PointT, PointT> > (gcg_alg);

        boost::shared_ptr <v4r::Source<PointT> > cast_source;
        if (param_.do_sift_ || param_.do_shot_ ) // for local recognizers we need this source type / training data
        {
            boost::shared_ptr < v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > src
                    (new v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT>);
            src->setPath (models_dir);
            src->setModelStructureDir (training_dir);
            std::string foo;
            src->generate (foo);
            src->createVoxelGridAndDistanceTransform(resolution);
            cast_source = boost::static_pointer_cast<v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > (src);
        }

        if (param_.do_sift_)
        {
#ifdef USE_SIFT_GPU

      if(!sift_) { //--create a new SIFT-GPU context
          static char kw[][16] = {"-m", "-fo", "-1", "-s", "-v", "1", "-pack"};
          char * argvv[] = {kw[0], kw[1], kw[2], kw[3],kw[4],kw[5],kw[6], NULL};

          int argcc = sizeof(argvv) / sizeof(char*);
          sift_ = new SiftGPU ();
          sift_->ParseParam (argcc, argvv);

          //create an OpenGL context for computation
          if (sift_->CreateContextGL () != SiftGPU::SIFTGPU_FULL_SUPPORTED)
            throw std::runtime_error ("PSiftGPU::PSiftGPU: No GL support!");
      }

      boost::shared_ptr < v4r::SIFTLocalEstimation<PointT, FeatureT > > estimator;
      estimator.reset (new v4r::SIFTLocalEstimation<PointT, FeatureT >(sift_));

      boost::shared_ptr < v4r::LocalEstimator<PointT, FeatureT > > cast_estimator;
      cast_estimator = boost::dynamic_pointer_cast<v4r::SIFTLocalEstimation<PointT, FeatureT > > (estimator);
#else
      boost::shared_ptr < v4r::OpenCVSIFTLocalEstimation<PointT, FeatureT > > estimator;
      estimator.reset (new v4r::OpenCVSIFTLocalEstimation<PointT, FeatureT >);

      boost::shared_ptr < v4r::LocalEstimator<PointT, FeatureT > > cast_estimator;
      cast_estimator = boost::dynamic_pointer_cast<v4r::OpenCVSIFTLocalEstimation<PointT, FeatureT > > (estimator);
#endif

            boost::shared_ptr<v4r::LocalRecognitionPipeline<flann::L1, PointT, FeatureT > > sift_r;
            sift_r.reset (new v4r::LocalRecognitionPipeline<flann::L1, PointT, FeatureT > ());
            sift_r->setDataSource (cast_source);
            sift_r->setTrainingDir (training_dir);
            sift_r->setDescriptorName ("sift");
            sift_r->setICPIterations (icp_iterations);
            sift_r->setFeatureEstimator (cast_estimator);
            sift_r->setUseCache (true);
            sift_r->setCGAlgorithm (cast_cg_alg_);
            sift_r->setKnn (param_.knn_sift_);
            sift_r->setSaveHypotheses(true);
            sift_r->initialize (false);

            boost::shared_ptr < v4r::Recognizer<PointT> > cast_recog;
            cast_recog = boost::static_pointer_cast<v4r::LocalRecognitionPipeline<flann::L1, PointT, FeatureT > > (sift_r);
            std::cout << "Feature Type: " << cast_recog->getFeatureType() << std::endl;
            rr_.addRecognizer (cast_recog);
        }
        if (param_.do_shot_)
        {
            boost::shared_ptr<v4r::UniformSamplingExtractor<PointT> > uniform_kp_extractor ( new v4r::UniformSamplingExtractor<PointT>);
            uniform_kp_extractor->setSamplingDensity (0.01f);
            uniform_kp_extractor->setFilterPlanar (true);
            uniform_kp_extractor->setMaxDistance( param_.chop_z_ );
            uniform_kp_extractor->setThresholdPlanar(0.1);

            boost::shared_ptr<v4r::KeypointExtractor<PointT> > keypoint_extractor = boost::static_pointer_cast<v4r::KeypointExtractor<PointT> > (uniform_kp_extractor);

            boost::shared_ptr<v4r::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator
                    (new v4r::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
            normal_estimator->setCMR (false);
            normal_estimator->setDoVoxelGrid (true);
            normal_estimator->setRemoveOutliers (false);
            normal_estimator->setValuesForCMRFalse (0.003f, 0.02f);

            boost::shared_ptr<v4r::SHOTLocalEstimationOMP<PointT, pcl::Histogram<352> > > estimator
                    (new v4r::SHOTLocalEstimationOMP<PointT, pcl::Histogram<352> >);
            estimator->setNormalEstimator (normal_estimator);
            estimator->addKeypointExtractor (keypoint_extractor);
            estimator->setSupportRadius (0.04f);
            estimator->setAdaptativeMLS (false);

            boost::shared_ptr<v4r::LocalEstimator<PointT, pcl::Histogram<352> > > cast_estimator;
            cast_estimator = boost::dynamic_pointer_cast<v4r::LocalEstimator<PointT, pcl::Histogram<352> > > (estimator);

            boost::shared_ptr<v4r::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> > > local;
            local.reset(new v4r::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> > ());
            local->setDataSource (cast_source);
            local->setTrainingDir(training_dir);
            local->setDescriptorName ("shot");
            local->setFeatureEstimator (cast_estimator);
            local->setCGAlgorithm (cast_cg_alg_);
            local->setKnn(param_.knn_shot_);
            local->setUseCache (true);
            local->setThresholdAcceptHyp (1);
            local->setICPIterations ( icp_iterations );
            local->setKdtreeSplits (128);
            local->setSaveHypotheses(true);
            local->initialize (false);
            local->setMaxDescriptorDistance(std::numeric_limits<float>::infinity());

            boost::shared_ptr<v4r::Recognizer<PointT> > cast_recog;
            cast_recog = boost::static_pointer_cast<v4r::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> > > (local);
            std::cout << "Feature Type: " << cast_recog->getFeatureType() << std::endl;
            rr_.addRecognizer(cast_recog);
            rr_.setSaveHypotheses(false);
        }


        boost::shared_ptr<v4r::GHV<PointT, PointT> > hyp_verification_method (new v4r::GHV<PointT, PointT>(paramGHV));
        boost::shared_ptr<v4r::HypothesisVerification<PointT,PointT> > cast_hyp_pointer =
                boost::static_pointer_cast<v4r::GHV<PointT, PointT> > (hyp_verification_method);
        rr_.setHVAlgorithm( cast_hyp_pointer );
        rr_.setCGAlgorithm( gcg_alg );

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
                v4r::computeNormals<PointT>(cloud, normals, param_.normal_computation_method_);

                if( param_.chop_z_ > 0) {
                    pcl::PassThrough<PointT> pass;
                    pass.setFilterLimits ( 0.f, param_.chop_z_ );
                    pass.setFilterFieldName ("z");
                    pass.setInputCloud (cloud);
                    pass.setKeepOrganized (true);
                    pass.filter (*cloud);
                    pcl::copyPointCloud(*normals, *pass.getIndices(), *normals);
                }

                rr_.setSceneNormals(normals);
                rr_.setInputCloud (cloud);
                rr_.recognize();

                std::vector<ModelTPtr> verified_models = rr_.getVerifiedModels();
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_verified;
                transforms_verified = rr_.getVerifiedTransforms();

                if (visualize_)
                {
                    visualize_result(cloud, verified_models, transforms_verified);
                    rr_.visualize();
                }

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
    Recognizer r_eval;
    r_eval.initialize(argc,argv);
    r_eval.test();
    return 0;
}
