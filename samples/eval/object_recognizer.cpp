#include <v4r_config.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/features/sift_local_estimator.h>

#ifndef HAVE_SIFTGPU
#include <v4r/features/opencv_sift_local_estimator.h>
#endif

#include <v4r/features/shot_local_estimator_omp.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/ghv.h>
#include <v4r/recognition/local_recognizer.h>
#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/recognition/recognizer.h>
#include <v4r/recognition/registered_views_source.h>

#include <pcl/common/centroid.h>
#include <pcl/common/time.h>
#include <pcl/filters/passthrough.h>

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdlib.h>

#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

class Rec
{
private:
    typedef pcl::PointXYZRGB PointT;
    typedef v4r::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;
    typedef pcl::Histogram<128> FeatureT;

    boost::shared_ptr<v4r::MultiRecognitionPipeline<PointT> > rr_;
    std::string test_dir_, out_dir_;
    bool visualize_;
    double chop_z_;

    std::map<std::string, size_t> rec_models_per_id_;

public:
    Rec()
    {
        chop_z_ = std::numeric_limits<float>::max();
    }

    bool initialize(int argc, char ** argv)
    {
        bool do_sift;
        bool do_shot;
        bool do_ourcvfh;
        float resolution = 0.005f;
        std::string models_dir, training_dir;

        v4r::GHV<PointT, PointT>::Parameter paramGHV;
        v4r::GraphGeometricConsistencyGrouping<PointT, PointT>::Parameter paramGgcg;
        v4r::LocalRecognitionPipeline<flann::L1, PointT, FeatureT >::Parameter paramLocalRecSift;
        v4r::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> >::Parameter paramLocalRecShot;
        v4r::MultiRecognitionPipeline<PointT>::Parameter paramMultiPipeRec;
        v4r::SHOTLocalEstimationOMP<PointT, pcl::Histogram<352> >::Parameter paramLocalEstimator;

        paramGgcg.max_time_allowed_cliques_comptutation_ = 100;
        paramLocalRecSift.use_cache_ = paramLocalRecShot.use_cache_ = true;
        paramLocalRecSift.save_hypotheses_ = paramLocalRecShot.save_hypotheses_ = true;
        paramLocalRecShot.kdtree_splits_ = 128;

        int normal_computation_method = paramLocalRecSift.normal_computation_method_;

        po::options_description desc("Single-View Object Instance Recognizer\n======================================\n**Allowed options");
        desc.add_options()
                ("help,h", "produce help message")
                ("models_dir,m", po::value<std::string>(&models_dir)->required(), "directory containing the model .pcd files")
                ("training_dir", po::value<std::string>(&training_dir)->required(), "directory containing the training data (for each model there should be a folder with the same name as the model and inside this folder there must be training views of the model with pose and segmented indices)")
                ("test_dir", po::value<std::string>(&test_dir_)->required(), "Directory with test scenes stored as point clouds (.pcd). The camera pose is taken directly from the pcd header fields \"sensor_orientation_\" and \"sensor_origin_\" (if the test directory contains subdirectories, each subdirectory is considered as seperate sequence for multiview recognition)")
                ("out_dir,o", po::value<std::string>(&out_dir_)->default_value("/tmp/sv_recognition_out/"), "Output directory where recognition results will be stored.")
                ("visualize,v", po::value<bool>(&visualize_)->default_value(true), "If true, turns visualization on")
                ("do_sift", po::value<bool>(&do_sift)->default_value(true), "if true, generates hypotheses using SIFT (visual texture information)")
                ("do_shot", po::value<bool>(&do_shot)->default_value(false), "if true, generates hypotheses using SHOT (local geometrical properties)")
                ("do_ourcvfh", po::value<bool>(&do_ourcvfh)->default_value(false), "if true, generates hypotheses using OurCVFH (global geometrical properties, requires segmentation!)")
                ("knn_sift", po::value<int>(&paramLocalRecSift.knn_)->default_value(paramLocalRecSift.knn_), "sets the number k of matches for each extracted SIFT feature to its k nearest neighbors")
                ("knn_shot", po::value<int>(&paramLocalRecShot.knn_)->default_value(paramLocalRecShot.knn_), "sets the number k of matches for each extracted SHOT feature to its k nearest neighbors")
                ("icp_iterations", po::value<int>(&paramMultiPipeRec.icp_iterations_)->default_value(paramMultiPipeRec.icp_iterations_), "number of icp iterations. If 0, no pose refinement will be done")
                ("icp_type", po::value<int>(&paramMultiPipeRec.icp_type_)->default_value(paramMultiPipeRec.icp_type_), "defines the icp method being used for pose refinement (0... regular ICP with CorrespondenceRejectorSampleConsensus, 1... crops point cloud of the scene to the bounding box of the model that is going to be refined)")
                ("max_corr_distance", po::value<double>(&paramMultiPipeRec.max_corr_distance_)->default_value(paramMultiPipeRec.max_corr_distance_,  boost::str(boost::format("%.2e") % paramMultiPipeRec.max_corr_distance_)), "defines the margin for the bounding box used when doing pose refinement with ICP of the cropped scene to the model")
                ("merge_close_hypotheses", po::value<bool>(&paramMultiPipeRec.merge_close_hypotheses_)->default_value(paramMultiPipeRec.merge_close_hypotheses_), "if true, close correspondence clusters (object hypotheses) of the same object model are merged together and this big cluster is refined")
                ("merge_close_hypotheses_dist", po::value<double>(&paramMultiPipeRec.merge_close_hypotheses_dist_)->default_value(paramMultiPipeRec.merge_close_hypotheses_dist_, boost::str(boost::format("%.2e") % paramMultiPipeRec.merge_close_hypotheses_dist_)), "defines the maximum distance of the centroids in meter for clusters to be merged together")
                ("merge_close_hypotheses_angle", po::value<double>(&paramMultiPipeRec.merge_close_hypotheses_angle_)->default_value(paramMultiPipeRec.merge_close_hypotheses_angle_, boost::str(boost::format("%.2e") % paramMultiPipeRec.merge_close_hypotheses_angle_) ), "defines the maximum angle in degrees for clusters to be merged together")
                ("chop_z,z", po::value<double>(&chop_z_)->default_value(chop_z_, boost::str(boost::format("%.2e") % chop_z_) ), "points with z-component higher than chop_z_ will be ignored (low chop_z reduces computation time and false positives (noise increase with z)")
                ("cg_size_thresh,c", po::value<size_t>(&paramGgcg.gc_threshold_)->default_value(paramGgcg.gc_threshold_), "Minimum cluster size. At least 3 correspondences are needed to compute the 6DOF pose ")
                ("cg_size", po::value<double>(&paramGgcg.gc_size_)->default_value(paramGgcg.gc_size_, boost::str(boost::format("%.2e") % paramGgcg.gc_size_) ), "Resolution of the consensus set used to cluster correspondences together ")
                ("cg_ransac_threshold", po::value<double>(&paramGgcg.ransac_threshold_)->default_value(paramGgcg.ransac_threshold_, boost::str(boost::format("%.2e") % paramGgcg.ransac_threshold_) ), " ")
                ("cg_dist_for_clutter_factor", po::value<double>(&paramGgcg.dist_for_cluster_factor_)->default_value(paramGgcg.dist_for_cluster_factor_, boost::str(boost::format("%.2e") % paramGgcg.dist_for_cluster_factor_) ), " ")
                ("cg_max_taken", po::value<size_t>(&paramGgcg.max_taken_correspondence_)->default_value(paramGgcg.max_taken_correspondence_), " ")
                ("cg_max_time_for_cliques_computation", po::value<double>(&paramGgcg.max_time_allowed_cliques_comptutation_)->default_value(100.d, "100.0"), " if grouping correspondences takes more processing time in milliseconds than this defined value, correspondences will be no longer computed by this graph based approach but by the simpler greedy correspondence grouping algorithm")
                ("cg_dot_distance", po::value<double>(&paramGgcg.thres_dot_distance_)->default_value(paramGgcg.thres_dot_distance_, boost::str(boost::format("%.2e") % paramGgcg.thres_dot_distance_) ) ,"")
                ("cg_use_graph", po::value<bool>(&paramGgcg.use_graph_)->default_value(paramGgcg.use_graph_), " ")
                ("hv_clutter_regularizer", po::value<double>(&paramGHV.clutter_regularizer_)->default_value(paramGHV.clutter_regularizer_, boost::str(boost::format("%.2e") % paramGHV.clutter_regularizer_) ), "The penalty multiplier used to penalize unexplained scene points within the clutter influence radius <i>radius_neighborhood_clutter_</i> of an explained scene point when they belong to the same smooth segment.")
                ("hv_color_sigma_ab", po::value<double>(&paramGHV.color_sigma_ab_)->default_value(paramGHV.color_sigma_ab_, boost::str(boost::format("%.2e") % paramGHV.color_sigma_ab_) ), "allowed chrominance (AB channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)")
                ("hv_color_sigma_l", po::value<double>(&paramGHV.color_sigma_l_)->default_value(paramGHV.color_sigma_l_, boost::str(boost::format("%.2e") % paramGHV.color_sigma_l_) ), "allowed illumination (L channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)")
                ("hv_detect_clutter", po::value<bool>(&paramGHV.detect_clutter_)->default_value(paramGHV.detect_clutter_), " ")
                ("hv_duplicity_cm_weight", po::value<double>(&paramGHV.w_occupied_multiple_cm_)->default_value(paramGHV.w_occupied_multiple_cm_, boost::str(boost::format("%.2e") % paramGHV.w_occupied_multiple_cm_) ), " ")
                ("hv_histogram_specification", po::value<bool>(&paramGHV.use_histogram_specification_)->default_value(paramGHV.use_histogram_specification_), " ")
                ("hv_hyp_penalty", po::value<double>(&paramGHV.active_hyp_penalty_)->default_value(paramGHV.active_hyp_penalty_, boost::str(boost::format("%.2e") % paramGHV.active_hyp_penalty_) ), " ")
                ("hv_ignore_color", po::value<bool>(&paramGHV.ignore_color_even_if_exists_)->default_value(paramGHV.ignore_color_even_if_exists_), " ")
                ("hv_initial_status", po::value<bool>(&paramGHV.initial_status_)->default_value(paramGHV.initial_status_), "sets the initial activation status of each hypothesis to this value before starting optimization. E.g. If true, all hypotheses will be active and the cost will be optimized from that initial status.")
                ("hv_color_space", po::value<int>(&paramGHV.color_space_)->default_value(paramGHV.color_space_), "specifies the color space being used for verification (0... LAB, 1... RGB, 2... Grayscale,  3,4,5,6... ???)")
                ("hv_inlier_threshold", po::value<double>(&paramGHV.inliers_threshold_)->default_value(paramGHV.inliers_threshold_, boost::str(boost::format("%.2e") % paramGHV.inliers_threshold_) ), "Represents the maximum distance between model and scene points in order to state that a scene point is explained by a model point. Valid model points that do not have any corresponding scene point within this threshold are considered model outliers")
                ("hv_occlusion_threshold", po::value<double>(&paramGHV.occlusion_thres_)->default_value(paramGHV.occlusion_thres_, boost::str(boost::format("%.2e") % paramGHV.occlusion_thres_) ), "Threshold for a point to be considered occluded when model points are back-projected to the scene ( depends e.g. on sensor noise)")
                ("hv_optimizer_type", po::value<int>(&paramGHV.opt_type_)->default_value(paramGHV.opt_type_), "defines the optimization methdod. 0: Local search (converges quickly, but can easily get trapped in local minima), 1: Tabu Search, 4; Tabu Search + Local Search (Replace active hypotheses moves), else: Simulated Annealing")
                ("hv_radius_clutter", po::value<double>(&paramGHV.radius_neighborhood_clutter_)->default_value(paramGHV.radius_neighborhood_clutter_, boost::str(boost::format("%.2e") % paramGHV.radius_neighborhood_clutter_) ), "defines the maximum distance between two points to be checked for label consistency")
                ("hv_radius_normals", po::value<double>(&paramGHV.radius_normals_)->default_value(paramGHV.radius_normals_, boost::str(boost::format("%.2e") % paramGHV.radius_normals_) ), " ")
                ("hv_regularizer,r", po::value<double>(&paramGHV.regularizer_)->default_value(paramGHV.regularizer_, boost::str(boost::format("%.2e") % paramGHV.regularizer_) ), "represents a penalty multiplier for model outliers. In particular, each model outlier associated with an active hypothesis increases the global cost function.")
                ("hv_plane_method", po::value<int>(&paramGHV.plane_method_)->default_value(paramGHV.plane_method_), "defines which method to use for plane extraction (if add_planes_ is true). 0... Multiplane Segmentation, 1... ClusterNormalsForPlane segmentation")
                ("hv_add_planes", po::value<bool>(&paramGHV.add_planes_)->default_value(paramGHV.add_planes_), "if true, adds planes as possible hypotheses (slower but decreases false positives especially for planes detected as flat objects like books)")
                ("hv_plane_inlier_distance", po::value<double>(&paramGHV.plane_inlier_distance_)->default_value(paramGHV.plane_inlier_distance_, boost::str(boost::format("%.2e") % paramGHV.plane_inlier_distance_) ), "Maximum inlier distance for plane clustering")
                ("hv_plane_thrAngle", po::value<double>(&paramGHV.plane_thrAngle_)->default_value(paramGHV.plane_thrAngle_, boost::str(boost::format("%.2e") % paramGHV.plane_thrAngle_) ), "Threshold of normal angle in degree for plane clustering")
                ("knn_plane_clustering_search", po::value<int>(&paramGHV.knn_plane_clustering_search_)->default_value(paramGHV.knn_plane_clustering_search_), "sets the number of points used for searching nearest neighbors in unorganized point clouds (used in plane segmentation)")
                ("hv_use_supervoxels", po::value<bool>(&paramGHV.use_super_voxels_)->default_value(paramGHV.use_super_voxels_), "If true, uses supervoxel clustering to detect smoothness violations")
                ("hv_min_plane_inliers", po::value<size_t>(&paramGHV.min_plane_inliers_)->default_value(paramGHV.min_plane_inliers_), "a planar cluster is only added as plane if it has at least min_plane_inliers_ points")
                ("normal_method,n", po::value<int>(&normal_computation_method)->default_value(normal_computation_method), "chosen normal computation method of the V4R library")
       ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return false;
        }

        try
        {
            po::notify(vm);
        }
        catch(std::exception& e)
        {
            std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
            return false;
        }

        paramLocalRecSift.normal_computation_method_ = paramLocalRecShot.normal_computation_method_ =
                paramMultiPipeRec.normal_computation_method_ = paramLocalEstimator.normal_computation_method_ = normal_computation_method;

        rr_.reset(new v4r::MultiRecognitionPipeline<PointT>(paramMultiPipeRec));

        boost::shared_ptr < v4r::GraphGeometricConsistencyGrouping<PointT, PointT> > gcg_alg (
                    new v4r::GraphGeometricConsistencyGrouping<PointT, PointT> (paramGgcg));

        boost::shared_ptr <v4r::Source<PointT> > cast_source;
        if (do_sift || do_shot ) // for local recognizers we need this source type / training data
        {
            boost::shared_ptr < v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > src
                    (new v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT>(resolution));
            src->setPath (models_dir);
            src->setModelStructureDir (training_dir);
            src->generate ();
//            src->createVoxelGridAndDistanceTransform(resolution);
            cast_source = boost::static_pointer_cast<v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > (src);
        }

        if (do_sift)
        {
#ifdef HAVE_SIFTGPU
      boost::shared_ptr < v4r::SIFTLocalEstimation<PointT, FeatureT > > estimator (new v4r::SIFTLocalEstimation<PointT, FeatureT >());
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
            LOG(INFO) << "Feature Type: " << cast_recog->getFeatureType();
            rr_->addRecognizer (cast_recog);
        }
        if (do_shot)
        {
            boost::shared_ptr<v4r::UniformSamplingExtractor<PointT> > uniform_kp_extractor ( new v4r::UniformSamplingExtractor<PointT>);
            uniform_kp_extractor->setSamplingDensity (0.01f);
            uniform_kp_extractor->setFilterPlanar (true);
            uniform_kp_extractor->setThresholdPlanar(0.1);
            uniform_kp_extractor->setMaxDistance( 100.0 ); // for training we want to consider all points (except nan values)

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

            uniform_kp_extractor->setMaxDistance( chop_z_ ); // for training we do not want this restriction

            boost::shared_ptr<v4r::Recognizer<PointT> > cast_recog;
            cast_recog = boost::static_pointer_cast<v4r::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> > > (local);
            LOG(INFO) << "Feature Type: " << cast_recog->getFeatureType();
            rr_->addRecognizer(cast_recog);
        }


        boost::shared_ptr<v4r::GHV<PointT, PointT> > hyp_verification_method (new v4r::GHV<PointT, PointT>(paramGHV));
        boost::shared_ptr<v4r::HypothesisVerification<PointT,PointT> > cast_hyp_pointer = boost::static_pointer_cast<v4r::GHV<PointT, PointT> > (hyp_verification_method);
        rr_->setHVAlgorithm( cast_hyp_pointer );
        rr_->setCGAlgorithm( gcg_alg );
        
        v4r::io::createDirIfNotExist(out_dir_);

        // writing parameters to file
        ofstream param_file;
        param_file.open ((out_dir_ + "/param.nfo").c_str());
        for(const auto& it : vm)
        {
          param_file << "--" << it.first << " ";

          auto& value = it.second.value();
          if (auto v = boost::any_cast<double>(&value))
            param_file << std::setprecision(3) << *v;
          else if (auto v = boost::any_cast<std::string>(&value))
            param_file << *v;
          else if (auto v = boost::any_cast<bool>(&value))
            param_file << *v;
          else if (auto v = boost::any_cast<int>(&value))
            param_file << *v;
          else if (auto v = boost::any_cast<size_t>(&value))
            param_file << *v;
          else
            param_file << "error";

          param_file << " ";
        }
        param_file.close();

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
            const std::string out_path = out_dir_ + "/" + sub_folder_names[ sub_folder_id ];
            v4r::io::createDirIfNotExist(out_path);

            rec_models_per_id_.clear();

            std::vector< std::string > views;
            v4r::io::getFilesInDirectory(sequence_path, views, "", ".*.pcd", false);
            std::sort(views.begin(), views.end());
            for (size_t v_id=0; v_id<views.size(); v_id++)
            {
                const std::string fn = test_dir_ + "/" + sub_folder_names[sub_folder_id] + "/" + views[ v_id ];

                LOG(INFO) << "Recognizing file " << fn;
                pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
                pcl::io::loadPCDFile(fn, *cloud);

                if( chop_z_ > 0)
                {
                    pcl::PassThrough<PointT> pass;
                    pass.setFilterLimits ( 0.f, chop_z_ );
                    pass.setFilterFieldName ("z");
                    pass.setInputCloud (cloud);
                    pass.setKeepOrganized (true);
                    pass.filter (*cloud);
                }

                rr_->setInputCloud (cloud);
                pcl::StopWatch watch;
                rr_->recognize();
                double time = watch.getTimeSeconds();

                std::stringstream out_fn;
                out_fn << out_path << "/" << views[v_id].substr(0, views[v_id].length()-4) << "_time.nfo";
                ofstream or_file (out_fn.str().c_str());
                or_file << time;
                or_file.close();

                std::vector<ModelTPtr> verified_models = rr_->getVerifiedModels();
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_verified;
                transforms_verified = rr_->getVerifiedTransforms();

                if (visualize_)
                    rr_->visualize();

                for(size_t m_id=0; m_id<verified_models.size(); m_id++)
                {
                    LOG(INFO) << "********************" << verified_models[m_id]->id_ << std::endl;

                    const std::string model_id = verified_models[m_id]->id_;
                    const Eigen::Matrix4f tf = transforms_verified[m_id];

                    size_t num_models_per_model_id;

                    std::map<std::string, size_t>::iterator it_rec_mod;
                    it_rec_mod = rec_models_per_id_.find(model_id);
                    if(it_rec_mod == rec_models_per_id_.end())
                    {
                        rec_models_per_id_.insert(std::pair<std::string, size_t>(model_id, 1));
                        num_models_per_model_id = 0;
                    }
                    else
                    {
                        num_models_per_model_id = it_rec_mod->second;
                        it_rec_mod->second++;
                    }

                    out_fn.str("");
                    out_fn << out_path << "/" << views[v_id].substr(0, views[v_id].length()-4) << "_"
                           << model_id.substr(0, model_id.length() - 4) << "_" << num_models_per_model_id << ".txt";

                    ofstream or_file;
                    or_file.open (out_fn.str().c_str());
                    for (size_t row=0; row <4; row++)
                    {
                        for(size_t col=0; col<4; col++)
                        {
                            or_file << tf(row, col) << " ";
                        }
                    }
                    or_file.close();
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
    google::InitGoogleLogging(argv[0]);
    Rec r_eval;
    if(r_eval.initialize(argc,argv))
        r_eval.test();
    return 0;
}
