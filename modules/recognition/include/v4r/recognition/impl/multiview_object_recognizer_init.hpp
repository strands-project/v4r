#include <v4r/recognition/multiview_object_recognizer.h>

#include <omp.h>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <glog/logging.h>

#include <v4r/recognition/hv_go_3D.h>
#include <v4r/recognition/registered_views_source.h>
#include <v4r/features/shot_local_estimator_omp.h>

#ifdef HAVE_SIFTGPU
#include <v4r/features/sift_local_estimator.h>
#else
#include <v4r/features/opencv_sift_local_estimator.h>
#endif


namespace po = boost::program_options;

namespace v4r
{

template<typename PointT>
MultiviewRecognizer<PointT>::MultiviewRecognizer(int argc, char **argv)
{
    id_ = 0;
    pose_ = Eigen::Matrix4f::Identity();

    bool do_sift;
    bool do_shot;
    bool do_ourcvfh;
    bool use_go3d;
    float resolution = 0.005f;
    std::string models_dir;

    // Parameter classes
    typename GO3D<PointT, PointT>::Parameter paramGO3D;
    typename GraphGeometricConsistencyGrouping<PointT, PointT>::Parameter paramGgcg;
    typename LocalRecognitionPipeline<PointT>::Parameter paramLocalRecSift, paramLocalRecShot;
    typename MultiRecognitionPipeline<PointT>::Parameter paramMultiPipeRec;
    typename SHOTLocalEstimationOMP<PointT>::Parameter paramLocalEstimator;
    nmInt_param_.octree_resolution_ = 0.004f;

    paramLocalRecSift.use_cache_ = paramLocalRecShot.use_cache_ = true;
    paramLocalRecSift.save_hypotheses_ = paramLocalRecShot.save_hypotheses_ = true;
    paramLocalRecShot.kdtree_splits_ = 128;

    int normal_computation_method = paramLocalRecSift.normal_computation_method_;

    po::options_description desc("");
    desc.add_options()
            ("help,h", "produce help message")
            ("models_dir,m", po::value<std::string>(&models_dir)->required(), "directory containing the object models")
//            ("test_dir,t", po::value<std::string>(&test_dir_)->required(), "Directory with test scenes stored as point clouds (.pcd). The camera pose is taken directly from the pcd header fields \"sensor_orientation_\" and \"sensor_origin_\" (if the test directory contains subdirectories, each subdirectory is considered as seperate sequence for multiview recognition)")
//            ("visualize,v", po::bool_switch(&visualize_), "visualize recognition results")
            ("do_sift", po::value<bool>(&do_sift)->default_value(true), "if true, generates hypotheses using SIFT (visual texture information)")
            ("do_shot", po::value<bool>(&do_shot)->default_value(false), "if true, generates hypotheses using SHOT (local geometrical properties)")
            ("do_ourcvfh", po::value<bool>(&do_ourcvfh)->default_value(false), "if true, generates hypotheses using OurCVFH (global geometrical properties, requires segmentation!)")
            ("use_go3d", po::value<bool>(&use_go3d)->default_value(false), "if true, verifies against a reconstructed scene from multiple viewpoints. Otherwise only against the current viewpoint.")
            ("knn_sift", po::value<size_t>(&paramLocalRecSift.knn_)->default_value(paramLocalRecSift.knn_), "sets the number k of matches for each extracted SIFT feature to its k nearest neighbors")
            ("knn_shot", po::value<size_t>(&paramLocalRecShot.knn_)->default_value(paramLocalRecShot.knn_), "sets the number k of matches for each extracted SHOT feature to its k nearest neighbors")
            ("transfer_feature_matches", po::value<bool>(&paramMultiPipeRec.save_hypotheses_)->default_value(paramMultiPipeRec.save_hypotheses_), "if true, transfers feature matches between views [Faeulhammer ea., ICRA 2015]. Otherwise generated hypotheses [Faeulhammer ea., MVA 2015].")
            ("icp_iterations", po::value<int>(&paramGO3D.icp_iterations_)->default_value(paramGO3D.icp_iterations_), "number of icp iterations. If 0, no pose refinement will be done")
            ("max_corr_distance", po::value<double>(&param_.max_corr_distance_)->default_value(param_.max_corr_distance_,  boost::str(boost::format("%.2e") % param_.max_corr_distance_)), "defines the margin for the bounding box used when doing pose refinement with ICP of the cropped scene to the model")
            ("merge_close_hypotheses", po::value<bool>(&param_.merge_close_hypotheses_)->default_value(param_.merge_close_hypotheses_), "if true, close correspondence clusters (object hypotheses) of the same object model are merged together and this big cluster is refined")
            ("merge_close_hypotheses_dist", po::value<double>(&param_.merge_close_hypotheses_dist_)->default_value(param_.merge_close_hypotheses_dist_, boost::str(boost::format("%.2e") % param_.merge_close_hypotheses_dist_)), "defines the maximum distance of the centroids in meter for clusters to be merged together")
            ("merge_close_hypotheses_angle", po::value<double>(&param_.merge_close_hypotheses_angle_)->default_value(param_.merge_close_hypotheses_angle_, boost::str(boost::format("%.2e") % param_.merge_close_hypotheses_angle_) ), "defines the maximum angle in degrees for clusters to be merged together")
            ("chop_z,z", po::value<double>(&param_.chop_z_)->default_value(param_.chop_z_, boost::str(boost::format("%.2e") % param_.chop_z_) ), "points with z-component higher than chop_z_ will be ignored (low chop_z reduces computation time and false positives (noise increase with z)")
            ("max_vertices_in_graph", po::value<int>(&param_.max_vertices_in_graph_)->default_value(param_.max_vertices_in_graph_), "maximum number of views taken into account (views selected in order of latest recognition calls)")
            ("compute_mst", po::value<bool>(&param_.compute_mst_)->default_value(param_.compute_mst_), "if true, does point cloud registration by SIFT background matching (given scene_to_scene_ == true), by using given pose (if use_robot_pose_ == true) and by common object hypotheses (if hyp_to_hyp_ == true) from all the possible connection a Mimimum Spanning Tree is computed. If false, it only uses the given pose for each point cloud ")
            ("cg_size_thresh", po::value<size_t>(&paramGgcg.gc_threshold_)->default_value(paramGgcg.gc_threshold_), "Minimum cluster size. At least 3 correspondences are needed to compute the 6DOF pose ")
            ("cg_size,c", po::value<double>(&paramGgcg.gc_size_)->default_value(paramGgcg.gc_size_, boost::str(boost::format("%.2e") % paramGgcg.gc_size_) ), "Resolution of the consensus set used to cluster correspondences together ")
            ("cg_ransac_threshold", po::value<double>(&paramGgcg.ransac_threshold_)->default_value(paramGgcg.ransac_threshold_, boost::str(boost::format("%.2e") % paramGgcg.ransac_threshold_) ), " ")
            ("cg_dist_for_clutter_factor", po::value<double>(&paramGgcg.dist_for_cluster_factor_)->default_value(paramGgcg.dist_for_cluster_factor_, boost::str(boost::format("%.2e") % paramGgcg.dist_for_cluster_factor_) ), " ")
            ("cg_max_taken", po::value<size_t>(&paramGgcg.max_taken_correspondence_)->default_value(paramGgcg.max_taken_correspondence_), " ")
            ("cg_max_time_for_cliques_computation", po::value<double>(&paramGgcg.max_time_allowed_cliques_comptutation_)->default_value(100.0, "100.0"), " if grouping correspondences takes more processing time in milliseconds than this defined value, correspondences will be no longer computed by this graph based approach but by the simpler greedy correspondence grouping algorithm")
            ("cg_dot_distance", po::value<double>(&paramGgcg.thres_dot_distance_)->default_value(paramGgcg.thres_dot_distance_, boost::str(boost::format("%.2e") % paramGgcg.thres_dot_distance_) ) ,"")
            ("cg_use_graph", po::value<bool>(&paramGgcg.use_graph_)->default_value(paramGgcg.use_graph_), " ")
            ("hv_clutter_regularizer", po::value<double>(&paramGO3D.clutter_regularizer_)->default_value(paramGO3D.clutter_regularizer_, boost::str(boost::format("%.2e") % paramGO3D.clutter_regularizer_) ), "The penalty multiplier used to penalize unexplained scene points within the clutter influence radius <i>radius_neighborhood_clutter_</i> of an explained scene point when they belong to the same smooth segment.")
            ("hv_color_sigma_ab", po::value<double>(&paramGO3D.color_sigma_ab_)->default_value(paramGO3D.color_sigma_ab_, boost::str(boost::format("%.2e") % paramGO3D.color_sigma_ab_) ), "allowed chrominance (AB channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)")
            ("hv_color_sigma_l", po::value<double>(&paramGO3D.color_sigma_l_)->default_value(paramGO3D.color_sigma_l_, boost::str(boost::format("%.2e") % paramGO3D.color_sigma_l_) ), "allowed illumination (L channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)")
            ("hv_detect_clutter", po::value<bool>(&paramGO3D.detect_clutter_)->default_value(paramGO3D.detect_clutter_), " ")
            ("hv_duplicity_cm_weight", po::value<double>(&paramGO3D.w_occupied_multiple_cm_)->default_value(paramGO3D.w_occupied_multiple_cm_, boost::str(boost::format("%.2e") % paramGO3D.w_occupied_multiple_cm_) ), " ")
            ("hv_histogram_specification", po::value<bool>(&paramGO3D.use_histogram_specification_)->default_value(paramGO3D.use_histogram_specification_), " ")
            ("hv_hyp_penalty", po::value<double>(&paramGO3D.active_hyp_penalty_)->default_value(paramGO3D.active_hyp_penalty_, boost::str(boost::format("%.2e") % paramGO3D.active_hyp_penalty_) ), " ")
            ("hv_ignore_color", po::value<bool>(&paramGO3D.ignore_color_even_if_exists_)->default_value(paramGO3D.ignore_color_even_if_exists_), " ")
            ("hv_initial_status", po::value<bool>(&paramGO3D.initial_status_)->default_value(paramGO3D.initial_status_), "sets the initial activation status of each hypothesis to this value before starting optimization. E.g. If true, all hypotheses will be active and the cost will be optimized from that initial status.")
            ("hv_color_space", po::value<int>(&paramGO3D.color_space_)->default_value(paramGO3D.color_space_), "specifies the color space being used for verification (0... LAB, 1... RGB, 2... Grayscale,  3,4,5,6... ?)")
            ("hv_inlier_threshold", po::value<double>(&paramGO3D.inliers_threshold_)->default_value(paramGO3D.inliers_threshold_, boost::str(boost::format("%.2e") % paramGO3D.inliers_threshold_) ), "Represents the maximum distance between model and scene points in order to state that a scene point is explained by a model point. Valid model points that do not have any corresponding scene point within this threshold are considered model outliers")
            ("hv_occlusion_threshold", po::value<double>(&paramGO3D.occlusion_thres_)->default_value(paramGO3D.occlusion_thres_, boost::str(boost::format("%.2e") % paramGO3D.occlusion_thres_) ), "Threshold for a point to be considered occluded when model points are back-projected to the scene ( depends e.g. on sensor noise)")
            ("hv_optimizer_type", po::value<int>(&paramGO3D.opt_type_)->default_value(paramGO3D.opt_type_), "defines the optimization methdod. 0: Local search (converges quickly, but can easily get trapped in local minima), 1: Tabu Search, 4; Tabu Search + Local Search (Replace active hypotheses moves), else: Simulated Annealing")
            ("hv_radius_clutter", po::value<double>(&paramGO3D.radius_neighborhood_clutter_)->default_value(paramGO3D.radius_neighborhood_clutter_, boost::str(boost::format("%.2e") % paramGO3D.radius_neighborhood_clutter_) ), "defines the maximum distance between two points to be checked for label consistency")
            ("hv_regularizer,r", po::value<double>(&paramGO3D.regularizer_)->default_value(paramGO3D.regularizer_, boost::str(boost::format("%.2e") % paramGO3D.regularizer_) ), "represents a penalty multiplier for model outliers. In particular, each model outlier associated with an active hypothesis increases the global cost function.")
            ("hv_plane_method", po::value<int>(&paramGO3D.plane_method_)->default_value(paramGO3D.plane_method_), "defines which method to use for plane extraction (if add_planes_ is true). 0... Multiplane Segmentation, 1... ClusterNormalsForPlane segmentation")
            ("hv_add_planes", po::value<bool>(&paramGO3D.add_planes_)->default_value(paramGO3D.add_planes_), "if true, adds planes as possible hypotheses (slower but decreases false positives especially for planes detected as flat objects like books)")
            ("hv_plane_inlier_distance", po::value<double>(&paramGO3D.plane_inlier_distance_)->default_value(paramGO3D.plane_inlier_distance_, boost::str(boost::format("%.2e") % paramGO3D.plane_inlier_distance_) ), "Maximum inlier distance for plane clustering")
            ("hv_plane_thrAngle", po::value<double>(&paramGO3D.plane_thrAngle_)->default_value(paramGO3D.plane_thrAngle_, boost::str(boost::format("%.2e") % paramGO3D.plane_thrAngle_) ), "Threshold of normal angle in degree for plane clustering")
            ("hv_use_supervoxels", po::value<bool>(&paramGO3D.use_super_voxels_)->default_value(paramGO3D.use_super_voxels_), "If true, uses supervoxel clustering to detect smoothness violations")
            ("knn_plane_clustering_search", po::value<int>(&paramGO3D.knn_plane_clustering_search_)->default_value(paramGO3D.knn_plane_clustering_search_), "sets the number of points used for searching nearest neighbors in unorganized point clouds (used in plane segmentation)")
            ("hv_min_plane_inliers", po::value<size_t>(&paramGO3D.min_plane_inliers_)->default_value(paramGO3D.min_plane_inliers_), "a planar cluster is only added as plane if it has at least min_plane_inliers_ points")
            ("hv_min_visible_ratio", po::value<double>(&paramGO3D.min_visible_ratio_)->default_value(paramGO3D.min_visible_ratio_, boost::str(boost::format("%.2e") % paramGO3D.min_visible_ratio_) ), "defines how much of the object has to be visible in order to be included in the verification stage")
            ("visualize_go3d_cues", po::value<bool>(&paramGO3D.visualize_cues_)->default_value(paramGO3D.visualize_cues_), "If true, visualizes cues computated at the go3d verification stage such as inlier, outlier points. Mainly used for debugging.")
            ("visualize_go_cues", po::value<bool>(&paramGO3D.visualize_go_cues_)->default_value(paramGO3D.visualize_go_cues_), "If true, visualizes cues computated at the hypothesis verification stage such as inlier, outlier points. Mainly used for debugging.")
            ("normal_method,n", po::value<int>(&normal_computation_method)->default_value(normal_computation_method), "chosen normal computation method of the V4R library")
            ("octree_radius", po::value<float>(&nmInt_param_.octree_resolution_)->default_value(nmInt_param_.octree_resolution_, boost::str(boost::format("%.2e") % nmInt_param_.octree_resolution_)), "resolution of the octree in the noise model based cloud registration used for hypothesis verification")
            ("edge_radius_px", po::value<float>(&nmInt_param_.edge_radius_px_)->default_value(nmInt_param_.edge_radius_px_, boost::str(boost::format("%.2e") % nmInt_param_.edge_radius_px_)), "points of the input cloud within this distance (in pixel) to its closest depth discontinuity pixel will be removed in the noise model based cloud registration used for hypothesis verification")
   ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; exit(0); }
    try { po::notify(vm); }
    catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }

    paramLocalRecSift.normal_computation_method_ = paramLocalRecShot.normal_computation_method_ =
            paramMultiPipeRec.normal_computation_method_ = paramLocalEstimator.normal_computation_method_ =
            param_.normal_computation_method_ = normal_computation_method;

    paramMultiPipeRec.merge_close_hypotheses_ = param_.merge_close_hypotheses_;

    rr_.reset(new MultiRecognitionPipeline<PointT>(paramMultiPipeRec));

    boost::shared_ptr < GraphGeometricConsistencyGrouping<PointT, PointT> > gcg_alg (
                new GraphGeometricConsistencyGrouping<PointT, PointT> (paramGgcg));

    boost::shared_ptr <Source<PointT> > cast_source;
    if (do_sift || do_shot ) // for local recognizers we need this source type / training data
    {
        boost::shared_ptr < RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > src
                (new RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT>(resolution));
        src->setPath (models_dir);
        src->generate ();
//            src->createVoxelGridAndDistanceTransform(resolution);
        cast_source = boost::static_pointer_cast<RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > (src);
    }

    if (do_sift)
    {
#ifdef HAVE_SIFTGPU
    static char kw[][16] = {"-m", "-fo", "-1", "-s", "-v", "1", "-pack"};
    char * argvv[] = {kw[0], kw[1], kw[2], kw[3],kw[4],kw[5],kw[6], NULL};

    int argcc = sizeof(argvv) / sizeof(char*);
    sift_.reset( new SiftGPU () );
    sift_->ParseParam (argcc, argvv);

    //create an OpenGL context for computation
    if (sift_->CreateContextGL () != SiftGPU::SIFTGPU_FULL_SUPPORTED)
      throw std::runtime_error ("PSiftGPU::PSiftGPU: No GL support!");

    boost::shared_ptr < SIFTLocalEstimation<PointT> > estimator (new SIFTLocalEstimation<PointT>(sift_));
    boost::shared_ptr < LocalEstimator<PointT> > cast_estimator = boost::dynamic_pointer_cast<SIFTLocalEstimation<PointT > > (estimator);
#else
  boost::shared_ptr < OpenCVSIFTLocalEstimation<PointT> > estimator (new OpenCVSIFTLocalEstimation<PointT >);
  boost::shared_ptr < LocalEstimator<PointT> > cast_estimator = boost::dynamic_pointer_cast<OpenCVSIFTLocalEstimation<PointT> > (estimator);
#endif

        boost::shared_ptr<LocalRecognitionPipeline<PointT> > sift_r;
        sift_r.reset (new LocalRecognitionPipeline<PointT> (paramLocalRecSift));
        sift_r->setDataSource (cast_source);
        sift_r->setModelsDir (models_dir);
        sift_r->setFeatureEstimator (cast_estimator);

        boost::shared_ptr < Recognizer<PointT> > cast_recog;
        cast_recog = boost::static_pointer_cast<LocalRecognitionPipeline<PointT > > (sift_r);
        rr_->addRecognizer (cast_recog);
    }
    if (do_shot)
    {
        boost::shared_ptr<UniformSamplingExtractor<PointT> > uniform_kp_extractor ( new UniformSamplingExtractor<PointT>);
        uniform_kp_extractor->setSamplingDensity (0.01f);
        uniform_kp_extractor->setFilterPlanar (true);
        uniform_kp_extractor->setThresholdPlanar(0.1);
        uniform_kp_extractor->setMaxDistance( 1000.0 ); // for training we want to consider all points (except nan values)

        boost::shared_ptr<KeypointExtractor<PointT> > keypoint_extractor = boost::static_pointer_cast<KeypointExtractor<PointT> > (uniform_kp_extractor);
        boost::shared_ptr<SHOTLocalEstimationOMP<PointT> > estimator (new SHOTLocalEstimationOMP<PointT>(paramLocalEstimator));
        estimator->addKeypointExtractor (keypoint_extractor);

        boost::shared_ptr<LocalEstimator<PointT> > cast_estimator;
        cast_estimator = boost::dynamic_pointer_cast<LocalEstimator<PointT> > (estimator);

        boost::shared_ptr<LocalRecognitionPipeline<PointT> > shot_r;
        shot_r.reset(new LocalRecognitionPipeline<PointT> (paramLocalRecShot));
        shot_r->setDataSource (cast_source);
        shot_r->setModelsDir(models_dir);
        shot_r->setFeatureEstimator (cast_estimator);

        uniform_kp_extractor->setMaxDistance( param_.chop_z_ ); // for training we do not want this restriction

        boost::shared_ptr<Recognizer<PointT> > cast_recog;
        cast_recog = boost::static_pointer_cast<LocalRecognitionPipeline<PointT> > (shot_r);
        rr_->addRecognizer(cast_recog);
    }

    if(!paramMultiPipeRec.save_hypotheses_)
        rr_->setCGAlgorithm( gcg_alg );

    rr_->initialize(false);

    if(use_go3d) {
        boost::shared_ptr<GO3D<PointT, PointT> > hyp_verification_method (new GO3D<PointT, PointT>(paramGO3D));
        hv_algorithm_ = boost::static_pointer_cast<GO3D<PointT, PointT> > (hyp_verification_method);
    }
    else {
        typename GHV<PointT, PointT>::Parameter paramGHV2 = paramGO3D;
        boost::shared_ptr<GHV<PointT, PointT> > hyp_verification_method (new GHV<PointT, PointT>(paramGHV2));
        hv_algorithm_ = boost::static_pointer_cast<GHV<PointT, PointT> > (hyp_verification_method);
    }

    setNoiseModelIntegrationParameters(nmInt_param_);
    setSingleViewRecognizer(rr_);
    setCGAlgorithm( gcg_alg );
#ifdef HAVE_SIFTGPU
    setSift(sift_);
#endif
}


}
