#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r_config.h>
#include <omp.h>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <glog/logging.h>

#include <v4r/recognition/ghv.h>
#include <v4r/recognition/global_recognizer.h>
#include <v4r/recognition/registered_views_source.h>
#include <v4r/features/esf_estimator.h>
#include <v4r/features/global_alexnet_cnn_estimator.h>
#include <v4r/features/ourcvfh_estimator.h>
#include <v4r/features/shot_local_estimator.h>
//#include <v4r/features/shot_color_local_estimator.h>
#include <v4r/segmentation/dominant_plane_segmenter.h>
#include <v4r/ml/nearestNeighbor.h>
#include <v4r/ml/svmWrapper.h>
#include <v4r/keypoints/iss_keypoint_extractor.h>
#include <v4r/keypoints/narf_keypoint_extractor.h>
#include <v4r/keypoints/harris3d_keypoint_extractor.h>
#include <v4r/keypoints/uniform_sampling_extractor.h>

#ifdef HAVE_SIFTGPU
#include <v4r/features/sift_local_estimator.h>
#else
#include <v4r/features/opencv_sift_local_estimator.h>
#endif

namespace po = boost::program_options;

namespace v4r
{
//template<>
//MultiRecognitionPipeline<pcl::PointXYZ>::MultiRecognitionPipeline(int argc, char **argv)
//{
//    (void) argc;
//    (void) argv;
//   std::cerr << "This initialization function is only available for XYZRGB point type!" << std::endl;
//}

template<typename PointT>
MultiRecognitionPipeline<PointT>::MultiRecognitionPipeline(int argc, char **argv)
{
    bool do_sift;
    bool do_shot;
    bool do_esf;
    bool do_alexnet;
    float resolution = 0.005f;
    float shot_support_radius = 0.02f;
    std::string models_dir;
    int keypoint_type = KeypointType::UniformSampling;
    bool shot_use_color;
    float sift_z;

    bool sift_make_dense = false;
    int sift_kp_stride = 20;

    typename GHV<PointT, PointT>::Parameter paramGHV;
    typename GraphGeometricConsistencyGrouping<PointT, PointT>::Parameter paramGgcg;
    typename LocalRecognitionPipeline<PointT>::Parameter paramSIFT, paramSHOT;
    typename GlobalRecognizer<PointT>::Parameter paramESF;
    svmClassifier::Parameter paramSVM;
    std::string svm_model_fn;

    paramGgcg.max_time_allowed_cliques_comptutation_ = 100;
    paramSHOT.kdtree_splits_ = 128;

    int normal_computation_method = paramSIFT.normal_computation_method_;

    po::options_description desc("");
    desc.add_options()
            ("help,h", "produce help message")
            ("models_dir,m", po::value<std::string>(&models_dir)->required(), "directory containing the object models")

            ("do_sift", po::value<bool>(&do_sift)->default_value(true), "if true, generates hypotheses using SIFT (visual texture information)")
            ("sift_knn", po::value<size_t>(&paramSIFT.knn_)->default_value(paramSIFT.knn_), "sets the number k of matches for each extracted SIFT feature to its k nearest neighbors")
            ("sift_use_codebook", po::value<bool>(&paramSIFT.use_codebook_)->default_value(false), "if true, performs K-Means clustering on all signatures being trained.")
            ("sift_codebook_size", po::value<size_t>(&paramSIFT.codebook_size_)->default_value(paramSIFT.codebook_size_), "number of clusters being computed for the codebook (K-Means)")
            ("sift_codebook_filter_ratio", po::value<float>(&paramSIFT.codebook_filter_ratio_)->default_value(paramSIFT.codebook_filter_ratio_), "signatures clustered into a cluster which occures more often (w.r.t the total number of signatures) than this threshold, will be rejected.")
            ("sift_filter_planar", po::value<bool>(&paramSIFT.filter_planar_)->default_value(false), "If true, filters keypoints which are on a planar surface..")
            ("sift_filter_points_above_plane", po::value<bool>(&paramSIFT.filter_points_above_plane_)->default_value(false), "If true, only recognizes points above dominant plane.")
            ("sift_filter_border_pts", po::value<bool>(&paramSIFT.filter_border_pts_)->default_value(paramSIFT.filter_border_pts_), "If true, filters keypoints at the boundary.")
            ("sift_border_width", po::value<int>(&paramSIFT.boundary_width_)->default_value(paramSIFT.boundary_width_), "Width in pixel of the depth discontinuity.")
            ("sift_z,z", po::value<float>(&sift_z)->default_value(3.0f), "points with z-component higher than chop_z_ will be ignored for SIFT (low chop_z reduces computation time and false positives (noise increase with z)")
            ("sift_make_dense", po::value<bool>(&sift_make_dense)->default_value(sift_make_dense), "if true, uses dense SIFT feature extraction")
            ("sift_stride", po::value<int>(&sift_kp_stride)->default_value(sift_kp_stride), "if dense SIFT is on, uses this stride for extracting SIFT keypoints")

            ("do_shot", po::value<bool>(&do_shot)->default_value(false), "if true, generates hypotheses using SHOT (local geometrical properties)")
            ("shot_use_color", po::value<bool>(&shot_use_color)->default_value(false), "if true, uses the color SHOT descriptor")
            ("shot_use_3d_model_for_training", po::value<bool>(&paramSHOT.use_3d_model_)->default_value(paramSHOT.use_3d_model_), "if true, it learns features directly from the reconstructed 3D model instead of on the individual training views.")
            ("shot_use_codebook", po::value<bool>(&paramSHOT.use_codebook_)->default_value(paramSHOT.use_codebook_), "if true, performs K-Means clustering on all signatures being trained.")
            ("shot_filter_planar", po::value<bool>(&paramSHOT.filter_planar_)->default_value(true), "If true, filters keypoints which are on a planar surface.")
            ("shot_filter_points_above_plane", po::value<bool>(&paramSHOT.filter_points_above_plane_)->default_value(true), "If true, only recognizes points above dominant plane.")
            ("shot_plane_threshold", po::value<float>(&paramSHOT.threshold_planar_)->default_value(paramSHOT.threshold_planar_), "Threshold ratio for planarity definition")
            ("shot_plane_support_radius", po::value<float>(&paramSHOT.planar_support_radius_)->default_value(paramSHOT.planar_support_radius_), "Patch size used to check keypoints for planarity")
            ("shot_keypoint_type", po::value<int>(&keypoint_type)->default_value(keypoint_type), "Define keypoint extraction type (0... Uniform Sampling, 1... Intrinsic Shape Signature(ISS), 2... Narf, 3... Harris3D. To enable multiple keypoint extraction, just enter the sum of the desired combinations.")
            ("shot_filter_border_pts", po::value<bool>(&paramSHOT.filter_border_pts_)->default_value(paramSHOT.filter_border_pts_), "If true, filters keypoints at the boundary.")
            ("shot_border_width", po::value<int>(&paramSHOT.boundary_width_)->default_value(paramSHOT.boundary_width_), "Width in pixel of the depth discontinuity.")
            ("shot_visualize_keypoints", po::bool_switch(&paramSHOT.visualize_keypoints_), "If set, visualizes the extracted 3D keypoints")
            ("shot_z", po::value<double>(&paramSHOT.max_distance_)->default_value(1.5f), "points with z-component higher than chop_z_ will be ignored for SHOT (low chop_z reduces computation time and false positives (noise increase with z)")
            ("shot_knn", po::value<size_t>(&paramSHOT.knn_)->default_value(paramSHOT.knn_), "sets the number k of matches for each extracted SHOT feature to its k nearest neighbors")
            ("shot_support_radius", po::value<float>(&shot_support_radius)->default_value(shot_support_radius), "Support radius for SHOT feature description")
            ("shot_codebook_size", po::value<size_t>(&paramSHOT.codebook_size_)->default_value(paramSHOT.codebook_size_), "number of clusters being computed for the codebook (K-Means)")
            ("shot_codebook_filter_ratio", po::value<float>(&paramSHOT.codebook_filter_ratio_)->default_value(paramSHOT.codebook_filter_ratio_), "signatures clustered into a cluster which occures more often (w.r.t the total number of signatures) than this threshold, will be rejected.")
            ("shot_kernel_sigma", po::value<float>(&paramSHOT.kernel_sigma_)->default_value(paramSHOT.kernel_sigma_), "signatures clustered into a cluster which occures more often (w.r.t the total number of signatures) than this threshold, will be rejected.")

            ("do_esf", po::value<bool>(&do_esf)->default_value(false), "if true, generates hypotheses using ESF (global geometrical properties, requires segmentation!)")
            ("esf_vis", po::bool_switch(&paramESF.visualize_clusters_), "If set, visualizes the cluster and displays recognition information for each.")
            ("esf_knn", po::value<size_t>(&paramESF.knn_)->default_value(paramESF.knn_), "sets the number k of matches for each extracted global feature to its k nearest neighbors")
            ("esf_check_elongation", po::value<bool>(&paramESF.check_elongations_)->default_value(paramESF.check_elongations_), "if true, checks each segment for its elongation along the two eigenvectors (with greatest eigenvalue) if they fit the matched model")
            ("esf_use_table_plane_for_alignment", po::value<bool>(&paramESF.use_table_plane_for_alignment_)->default_value(paramESF.use_table_plane_for_alignment_), "if true, aligns the object model such that the centroid is equal to the centroid of the segmented cluster projected onto the found table plane. The z-axis is aligned with the normal vector of the plane and the other axis are on the table plane")

            ("do_alexnet", po::value<bool>(&do_alexnet)->default_value(false), "if true, generates hypotheses using AlexNet features ")
            ("do_svm_cross_validation", po::value<bool>(&paramSVM.do_cross_validation_)->default_value(paramSVM.do_cross_validation_), "if true, does k cross validation on the svm training data")
            ("svm_model", po::value<std::string>(&svm_model_fn)->default_value(""), "filename to read svm model. If set (file exists), training is skipped and this model loaded instead.")

            ("max_corr_distance", po::value<double>(&param_.max_corr_distance_)->default_value(param_.max_corr_distance_,  boost::str(boost::format("%.2e") % param_.max_corr_distance_)), "defines the margin for the bounding box used when doing pose refinement with ICP of the cropped scene to the model")
            ("merge_close_hypotheses", po::value<bool>(&param_.merge_close_hypotheses_)->default_value(param_.merge_close_hypotheses_), "if true, close correspondence clusters (object hypotheses) of the same object model are merged together and this big cluster is refined")
            ("merge_close_hypotheses_dist", po::value<double>(&param_.merge_close_hypotheses_dist_)->default_value(param_.merge_close_hypotheses_dist_, boost::str(boost::format("%.2e") % param_.merge_close_hypotheses_dist_)), "defines the maximum distance of the centroids in meter for clusters to be merged together")
            ("merge_close_hypotheses_angle", po::value<double>(&param_.merge_close_hypotheses_angle_)->default_value(param_.merge_close_hypotheses_angle_, boost::str(boost::format("%.2e") % param_.merge_close_hypotheses_angle_) ), "defines the maximum angle in degrees for clusters to be merged together")

            ("cg_size_thresh,c", po::value<size_t>(&paramGgcg.gc_threshold_)->default_value(paramGgcg.gc_threshold_), "Minimum cluster size. At least 3 correspondences are needed to compute the 6DOF pose ")
            ("cg_size", po::value<double>(&paramGgcg.gc_size_)->default_value(paramGgcg.gc_size_, boost::str(boost::format("%.2e") % paramGgcg.gc_size_) ), "Resolution of the consensus set used to cluster correspondences together ")
            ("cg_ransac_threshold", po::value<double>(&paramGgcg.ransac_threshold_)->default_value(paramGgcg.ransac_threshold_, boost::str(boost::format("%.2e") % paramGgcg.ransac_threshold_) ), " ")
            ("cg_dist_for_clutter_factor", po::value<double>(&paramGgcg.dist_for_cluster_factor_)->default_value(paramGgcg.dist_for_cluster_factor_, boost::str(boost::format("%.2e") % paramGgcg.dist_for_cluster_factor_) ), " ")
            ("cg_max_taken", po::value<size_t>(&paramGgcg.max_taken_correspondence_)->default_value(paramGgcg.max_taken_correspondence_), " ")
            ("cg_max_time_for_cliques_computation", po::value<double>(&paramGgcg.max_time_allowed_cliques_comptutation_)->default_value(100.0, "100.0"), " if grouping correspondences takes more processing time in milliseconds than this defined value, correspondences will be no longer computed by this graph based approach but by the simpler greedy correspondence grouping algorithm")
            ("cg_dot_distance", po::value<double>(&paramGgcg.thres_dot_distance_)->default_value(paramGgcg.thres_dot_distance_, boost::str(boost::format("%.2e") % paramGgcg.thres_dot_distance_) ) ,"")
            ("cg_use_graph", po::value<bool>(&paramGgcg.use_graph_)->default_value(paramGgcg.use_graph_), " ")

            ("hv_icp_iterations", po::value<int>(&paramGHV.icp_iterations_)->default_value(paramGHV.icp_iterations_), "number of icp iterations. If 0, no pose refinement will be done")
            ("hv_clutter_regularizer", po::value<double>(&paramGHV.clutter_regularizer_)->default_value(paramGHV.clutter_regularizer_, boost::str(boost::format("%.2e") % paramGHV.clutter_regularizer_) ), "The penalty multiplier used to penalize unexplained scene points within the clutter influence radius <i>radius_neighborhood_clutter_</i> of an explained scene point when they belong to the same smooth segment.")
            ("hv_color_sigma_ab", po::value<double>(&paramGHV.color_sigma_ab_)->default_value(paramGHV.color_sigma_ab_, boost::str(boost::format("%.2e") % paramGHV.color_sigma_ab_) ), "allowed chrominance (AB channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)")
            ("hv_color_sigma_l", po::value<double>(&paramGHV.color_sigma_l_)->default_value(paramGHV.color_sigma_l_, boost::str(boost::format("%.2e") % paramGHV.color_sigma_l_) ), "allowed illumination (L channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)")
            ("hv_sigma_normals_deg", po::value<double>(&paramGHV.sigma_normals_deg_)->default_value(paramGHV.sigma_normals_deg_, boost::str(boost::format("%.2e") % paramGHV.sigma_normals_deg_) ), "variance for surface normals")
            ("hv_histogram_specification", po::value<bool>(&paramGHV.use_histogram_specification_)->default_value(paramGHV.use_histogram_specification_), " ")
            ("hv_ignore_color", po::value<bool>(&paramGHV.ignore_color_even_if_exists_)->default_value(paramGHV.ignore_color_even_if_exists_), " ")
            ("hv_initial_status", po::value<bool>(&paramGHV.initial_status_)->default_value(paramGHV.initial_status_), "sets the initial activation status of each hypothesis to this value before starting optimization. E.g. If true, all hypotheses will be active and the cost will be optimized from that initial status.")
            ("hv_color_space", po::value<int>(&paramGHV.color_space_)->default_value(paramGHV.color_space_), "specifies the color space being used for verification (0... LAB, 1... RGB, 2... Grayscale,  3,4,5,6... ?)")
            ("hv_color_stddev_mul", po::value<float>(&paramGHV.color_std_dev_multiplier_threshold_)->default_value(paramGHV.color_std_dev_multiplier_threshold_), "standard deviation multiplier threshold for the local color description for each color channel")
            ("hv_inlier_threshold", po::value<double>(&paramGHV.inliers_threshold_)->default_value(paramGHV.inliers_threshold_, boost::str(boost::format("%.2e") % paramGHV.inliers_threshold_) ), "Represents the maximum distance between model and scene points in order to state that a scene point is explained by a model point. Valid model points that do not have any corresponding scene point within this threshold are considered model outliers")
            ("hv_occlusion_threshold", po::value<double>(&paramGHV.occlusion_thres_)->default_value(paramGHV.occlusion_thres_, boost::str(boost::format("%.2e") % paramGHV.occlusion_thres_) ), "Threshold for a point to be considered occluded when model points are back-projected to the scene ( depends e.g. on sensor noise)")
            ("hv_optimizer_type", po::value<int>(&paramGHV.opt_type_)->default_value(paramGHV.opt_type_), "defines the optimization methdod. 0: Local search (converges quickly, but can easily get trapped in local minima), 1: Tabu Search, 4; Tabu Search + Local Search (Replace active hypotheses moves), else: Simulated Annealing")
            ("hv_radius_clutter", po::value<double>(&paramGHV.radius_neighborhood_clutter_)->default_value(paramGHV.radius_neighborhood_clutter_, boost::str(boost::format("%.2e") % paramGHV.radius_neighborhood_clutter_) ), "defines the maximum distance between two points to be checked for label consistency")
            ("hv_regularizer,r", po::value<double>(&paramGHV.regularizer_)->default_value(paramGHV.regularizer_, boost::str(boost::format("%.2e") % paramGHV.regularizer_) ), "represents a penalty multiplier for model outliers. In particular, each model outlier associated with an active hypothesis increases the global cost function.")
            ("hv_min_model_fitness_lower_bound", po::value<float>(&paramGHV.min_model_fitness_lower_bound_)->default_value(paramGHV.min_model_fitness_lower_bound_, boost::str(boost::format("%.2e") % paramGHV.min_model_fitness_lower_bound_) ), "defines the fitness threshold for a hypothesis to be kept for optimization (0... no threshold, 1... everything gets rejected)")
            ("hv_min_model_fitness_upper_bound", po::value<float>(&paramGHV.min_model_fitness_upper_bound_)->default_value(paramGHV.min_model_fitness_upper_bound_, boost::str(boost::format("%.2e") % paramGHV.min_model_fitness_upper_bound_) ), "defines the fitness threshold for a hypothesis to be kept for optimization (0... no threshold, 1... everything gets rejected)")
            ("hv_min_visible_ratio", po::value<float>(&paramGHV.min_visible_ratio_)->default_value(paramGHV.min_visible_ratio_, boost::str(boost::format("%.2e") % paramGHV.min_visible_ratio_) ), "defines how much of the object has to be visible in order to be included in the verification stage")
            ("hv_min_ratio_smooth_cluster_explained", po::value<float>(&paramGHV.min_ratio_cluster_explained_)->default_value(paramGHV.min_ratio_cluster_explained_, boost::str(boost::format("%.2e") % paramGHV.min_ratio_cluster_explained_) ), " defines the minimum ratio a smooth cluster has to be explained by the visible points (given there are at least 100 points)")
            ("hv_vis_cues", po::bool_switch(&paramGHV.visualize_go_cues_), "If set, visualizes cues computated at the hypothesis verification stage such as inlier, outlier points. Mainly used for debugging.")
            ("hv_vis_model_cues", po::bool_switch(&paramGHV.visualize_model_cues_), "If set, visualizes the model cues. Useful for debugging")
            ("hv_vis_pairwise_cues", po::bool_switch(&paramGHV.visualize_pairwise_cues_), "If set, visualizes the pairwise cues. Useful for debugging")

            ("normal_method,n", po::value<int>(&normal_computation_method)->default_value(normal_computation_method), "chosen normal computation method of the V4R library")
;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; exit(0); }
    try { po::notify(vm); }
    catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }

    paramSIFT.normal_computation_method_ = paramSHOT.normal_computation_method_ =
            param_.normal_computation_method_ = normal_computation_method;

    typename Source<PointT>::Ptr cast_source;

    typename RegisteredViewsSource<PointT>::Ptr src (new RegisteredViewsSource<PointT>(resolution));
    src->setPath (models_dir);
    src->generate ();
//            src->createVoxelGridAndDistanceTransform(resolution);
    cast_source = boost::static_pointer_cast<RegisteredViewsSource<PointT> > (src);

    if (do_sift)
    {
        typename SIFTLocalEstimation<PointT>::Ptr estimator (new SIFTLocalEstimation<PointT>());
        estimator->setMaxDistance(sift_z);
        estimator->param_.dense_extraction_ = sift_make_dense;
        estimator->param_.stride_ = sift_kp_stride;
        typename LocalEstimator<PointT>::Ptr cast_estimator = boost::dynamic_pointer_cast<LocalEstimator<PointT> > (estimator);

        typename LocalRecognitionPipeline<PointT>::Ptr sift_r (new LocalRecognitionPipeline<PointT> (paramSIFT));
        sift_r->setDataSource (cast_source);
        sift_r->setModelsDir (models_dir);
        sift_r->setFeatureEstimator (cast_estimator);

        typename Recognizer<PointT>::Ptr cast_recog = boost::static_pointer_cast<Recognizer<PointT > > (sift_r);
        LOG(INFO) << "Feature Type: " << cast_recog->getFeatureType();
        addRecognizer (cast_recog);
    }

    if (do_shot)
    {
        typename LocalEstimator<PointT>::Ptr cast_estimator;

        typename LocalRecognitionPipeline<PointT>::Ptr local (new LocalRecognitionPipeline<PointT> (paramSHOT));
        local->setDataSource (cast_source);
        local->setModelsDir (models_dir);

//        if (shot_use_color)
//        {
//            typename SHOTColorLocalEstimation<PointT>::Ptr shot_estimator (new SHOTColorLocalEstimation<PointT>(shot_support_radius));
//            cast_estimator = boost::dynamic_pointer_cast<LocalEstimator<PointT> > (shot_estimator);
//        }
//        else
        {
            typename SHOTLocalEstimation<PointT>::Ptr shot_estimator (new SHOTLocalEstimation<PointT>(shot_support_radius));
            cast_estimator = boost::dynamic_pointer_cast<LocalEstimator<PointT> > (shot_estimator);
        }
        local->setFeatureEstimator (cast_estimator);


        if ( keypoint_type == KeypointType::UniformSampling )
        {
            typename UniformSamplingExtractor<PointT>::Ptr extr ( new UniformSamplingExtractor<PointT>(0.01f));
            typename KeypointExtractor<PointT>::Ptr keypoint_extractor = boost::static_pointer_cast<KeypointExtractor<PointT> > (extr);
            local->addKeypointExtractor (keypoint_extractor);
        }

        if ( keypoint_type == KeypointType::ISS )
        {
            typename IssKeypointExtractor<PointT>::Ptr extr (new IssKeypointExtractor<PointT>);
            typename KeypointExtractor<PointT>::Ptr keypoint_extractor = boost::static_pointer_cast<KeypointExtractor<PointT> > (extr);
            local->addKeypointExtractor (keypoint_extractor);
        }

        if ( keypoint_type == KeypointType::NARF )
        {
            typename NarfKeypointExtractor<PointT>::Ptr extr (new NarfKeypointExtractor<PointT>);
            typename KeypointExtractor<PointT>::Ptr keypoint_extractor = boost::static_pointer_cast<KeypointExtractor<PointT> > (extr);
            local->addKeypointExtractor (keypoint_extractor);
        }

        if ( keypoint_type == KeypointType::HARRIS3D )
        {
            typename Harris3DKeypointExtractor<PointT>::Ptr extr (new Harris3DKeypointExtractor<PointT>);
            typename KeypointExtractor<PointT>::Ptr keypoint_extractor = boost::static_pointer_cast<KeypointExtractor<PointT> > (extr);
            local->addKeypointExtractor (keypoint_extractor);
        }

        typename Recognizer<PointT>::Ptr cast_recog = boost::static_pointer_cast<Recognizer<PointT> > (local);
        LOG(INFO) << "Feature Type: " << cast_recog->getFeatureType();
        addRecognizer(cast_recog);
    }

    if (do_esf)
    {
        // feature type
        typename GlobalEstimator<PointT>::Ptr cast_estimator;
        typename ESFEstimation<PointT>::Ptr est (new ESFEstimation<PointT>);
//        typename OURCVFHEstimator<PointT>::Ptr est (new OURCVFHEstimator<PointT>);
        cast_estimator = boost::dynamic_pointer_cast<GlobalEstimator<PointT> > (est);

        // segmentation type
        typename Segmenter<PointT>::Ptr cast_segmenter;
        typename DominantPlaneSegmenter<PointT>::Ptr dp_seg (new DominantPlaneSegmenter<PointT> (argc, argv));
        cast_segmenter = boost::dynamic_pointer_cast<Segmenter<PointT> > (dp_seg);

        // classifier
        typename Classifier::Ptr cast_classifier;
        NearestNeighborClassifier::Ptr classifier (new NearestNeighborClassifier);
        cast_classifier = boost::dynamic_pointer_cast<Classifier > (classifier);

        typename GlobalRecognizer<PointT>::Ptr global_r (new GlobalRecognizer<PointT>(paramESF));
        global_r->setSegmentationAlgorithm(cast_segmenter);
        global_r->setFeatureEstimator(cast_estimator);
        global_r->setDataSource(cast_source);
        global_r->setModelsDir(models_dir);
        global_r->setClassifier(cast_classifier);

        typename Recognizer<PointT>::Ptr cast_recog = boost::static_pointer_cast<Recognizer<PointT> > (global_r);
        LOG(INFO) << "Feature Type: " << cast_recog->getFeatureType();
        addRecognizer(cast_recog);
    }


    if (do_alexnet)
    {
        // feature type
        typename GlobalEstimator<PointT>::Ptr cast_estimator;
        typename CNN_Feat_Extractor<PointT>::Ptr est (new CNN_Feat_Extractor<PointT>(argc, argv));
        cast_estimator = boost::dynamic_pointer_cast<GlobalEstimator<PointT> > (est);

        // segmentation type
        typename Segmenter<PointT>::Ptr cast_segmenter;
        typename DominantPlaneSegmenter<PointT>::Ptr dp_seg (new DominantPlaneSegmenter<PointT> (argc, argv));
        cast_segmenter = boost::dynamic_pointer_cast<Segmenter<PointT> > (dp_seg);

        // classifier
        typename Classifier::Ptr cast_classifier;
//        NearestNeighborClassifier::Ptr classifier (new NearestNeighborClassifier);
        svmClassifier::Ptr classifier (new svmClassifier (paramSVM));
        classifier->setInFilename(svm_model_fn);
        cast_classifier = boost::dynamic_pointer_cast<Classifier > (classifier);

        typename GlobalRecognizer<PointT>::Ptr global_r (new GlobalRecognizer<PointT>(paramESF));
        global_r->setSegmentationAlgorithm(cast_segmenter);
        global_r->setFeatureEstimator(cast_estimator);
        global_r->setDataSource(cast_source);
        global_r->setModelsDir(models_dir);
        global_r->setClassifier(cast_classifier);

        typename Recognizer<PointT>::Ptr cast_recog = boost::static_pointer_cast<Recognizer<PointT> > (global_r);
        LOG(INFO) << "Feature Type: " << cast_recog->getFeatureType();
        addRecognizer(cast_recog);
    }

    this->setDataSource(cast_source);
    initialize(false);
    typename GHV<PointT, PointT>::Ptr hyp_verification_method (new GHV<PointT, PointT>(paramGHV));
    hv_algorithm_ = boost::static_pointer_cast<GHV<PointT, PointT> > (hyp_verification_method);
    cg_algorithm_.reset( new GraphGeometricConsistencyGrouping<PointT, PointT> (paramGgcg));
}

}
