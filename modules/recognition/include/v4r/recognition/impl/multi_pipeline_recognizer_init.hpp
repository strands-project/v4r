#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r_config.h>
#include <omp.h>
#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/recognition/global_recognizer.h>
#include <v4r/recognition/registered_views_source.h>
#include <v4r/features/all_headers.h>
#include <v4r/keypoints/all_headers.h>
#include <v4r/segmentation/all_headers.h>
#include <v4r/ml/nearestNeighbor.h>
#include <v4r/ml/svmWrapper.h>

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
MultiRecognitionPipeline<PointT>::MultiRecognitionPipeline(std::vector<std::string> &arguments)
{
    bool do_sift;
    bool do_shot;
    bool do_esf;
    bool do_alexnet;
    int resolution_mm = 5;
    float shot_support_radius = 0.02f;
    std::string models_dir;
    int keypoint_type = KeypointType::UniformSampling;
    bool shot_use_color;
    float sift_z;

    bool sift_make_dense = false;
    int sift_kp_stride = 20;

    typename LocalRecognitionPipeline<PointT>::Parameter paramSIFT, paramSHOT;
    typename GlobalRecognizer<PointT>::Parameter paramESF;
    svmClassifier::Parameter paramSVM;
    std::string alexnet_svm_model_fn;
    std::string esf_svm_model_fn;
    paramSHOT.kdtree_splits_ = 128;

    int normal_computation_method = paramSIFT.normal_computation_method_;

    po::options_description desc("");
    desc.add_options()
            ("help,h", "produce help message")
            ("models_dir,m", po::value<std::string>(&models_dir)->required(), "directory containing the object models")

            ("do_sift", po::value<bool>(&do_sift)->default_value(true), "if true, generates hypotheses using SIFT (visual texture information)")
            ("sift_knn", po::value<size_t>(&paramSIFT.knn_)->default_value(7), "sets the number k of matches for each extracted SIFT feature to its k nearest neighbors")
            ("sift_use_codebook", po::value<bool>(&paramSIFT.use_codebook_)->default_value(false), "if true, performs K-Means clustering on all signatures being trained.")
            ("sift_codebook_size", po::value<size_t>(&paramSIFT.codebook_size_)->default_value(paramSIFT.codebook_size_), "number of clusters being computed for the codebook (K-Means)")
            ("sift_codebook_filter_ratio", po::value<float>(&paramSIFT.codebook_filter_ratio_)->default_value(paramSIFT.codebook_filter_ratio_), "signatures clustered into a cluster which occures more often (w.r.t the total number of signatures) than this threshold, will be rejected.")
            ("sift_filter_planar", po::value<bool>(&paramSIFT.filter_planar_)->default_value(false), "If true, filters keypoints which are on a planar surface..")
            ("sift_filter_points_above_plane", po::value<bool>(&paramSIFT.filter_points_above_plane_)->default_value(false), "If true, only recognizes points above dominant plane.")
            ("sift_min_plane_size", po::value<int>(&paramSIFT.min_plane_size_)->default_value(paramSIFT.min_plane_size_), "This is the minimum number of points required to estimate a plane for filtering.")
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
            ("shot_filter_points_above_plane", po::value<bool>(&paramSHOT.filter_points_above_plane_)->default_value(false), "If true, only recognizes points above dominant plane.")
            ("shot_plane_threshold", po::value<float>(&paramSHOT.threshold_planar_)->default_value(paramSHOT.threshold_planar_), "Threshold ratio for planarity definition")
            ("shot_plane_support_radius", po::value<float>(&paramSHOT.planar_support_radius_)->default_value(paramSHOT.planar_support_radius_), "Patch size used to check keypoints for planarity")
            ("shot_keypoint_type", po::value<int>(&keypoint_type)->default_value(keypoint_type), "Define keypoint extraction type (0... Uniform Sampling, 1... Intrinsic Shape Signature(ISS), 2... Narf, 3... Harris3D. To enable multiple keypoint extraction, just enter the sum of the desired combinations.")
            ("shot_filter_border_pts", po::value<bool>(&paramSHOT.filter_border_pts_)->default_value(paramSHOT.filter_border_pts_), "If true, filters keypoints at the boundary.")
            ("shot_border_width", po::value<int>(&paramSHOT.boundary_width_)->default_value(paramSHOT.boundary_width_), "Width in pixel of the depth discontinuity.")
            ("shot_visualize_keypoints", po::bool_switch(&paramSHOT.visualize_keypoints_), "If set, visualizes the extracted 3D keypoints")
            ("shot_z", po::value<double>(&paramSHOT.max_distance_)->default_value(1.5f), "points with z-component higher than chop_z_ will be ignored for SHOT (low chop_z reduces computation time and false positives (noise increase with z)")
            ("shot_knn", po::value<size_t>(&paramSHOT.knn_)->default_value(3), "sets the number k of matches for each extracted SHOT feature to its k nearest neighbors")
            ("shot_support_radius", po::value<float>(&shot_support_radius)->default_value(shot_support_radius), "Support radius for SHOT feature description")
            ("shot_codebook_size", po::value<size_t>(&paramSHOT.codebook_size_)->default_value(paramSHOT.codebook_size_), "number of clusters being computed for the codebook (K-Means)")
            ("shot_codebook_filter_ratio", po::value<float>(&paramSHOT.codebook_filter_ratio_)->default_value(paramSHOT.codebook_filter_ratio_), "signatures clustered into a cluster which occures more often (w.r.t the total number of signatures) than this threshold, will be rejected.")
            ("shot_kernel_sigma", po::value<float>(&paramSHOT.kernel_sigma_)->default_value(paramSHOT.kernel_sigma_), "signatures clustered into a cluster which occures more often (w.r.t the total number of signatures) than this threshold, will be rejected.")

            ("do_esf", po::value<bool>(&do_esf)->default_value(false), "if true, generates hypotheses using ESF (global geometrical properties, requires segmentation!)")
            ("esf_vis", po::bool_switch(&paramESF.visualize_clusters_), "If set, visualizes the cluster and displays recognition information for each.")
            ("esf_knn", po::value<size_t>(&paramESF.knn_)->default_value(paramESF.knn_), "sets the number k of matches for each extracted global feature to its k nearest neighbors")
            ("esf_check_elongation", po::value<bool>(&paramESF.check_elongations_)->default_value(paramESF.check_elongations_), "if true, checks each segment for its elongation along the two eigenvectors (with greatest eigenvalue) if they fit the matched model")
            ("esf_min_elongation_ratio", po::value<float>(&paramESF.min_elongation_ratio_)->default_value(paramESF.min_elongation_ratio_), "Minimum ratio of the elongation of the matched object to the extracted cluster to be accepted")
            ("esf_max_elongation_ratio", po::value<float>(&paramESF.max_elongation_ratio_)->default_value(paramESF.max_elongation_ratio_), "Maxium ratio of the elongation of the matched object to the extracted cluster to be accepted")
            ("esf_use_table_plane_for_alignment", po::value<bool>(&paramESF.use_table_plane_for_alignment_)->default_value(1), "if true, aligns the object model such that the centroid is equal to the centroid of the segmented cluster projected onto the found table plane. The z-axis is aligned with the normal vector of the plane and the other axis are on the table plane")
            ("esf_svm_model", po::value<std::string>(&esf_svm_model_fn)->default_value(""), "filename to read svm model for ESF. If set (file exists), training is skipped and this model loaded instead.")

            ("do_alexnet", po::value<bool>(&do_alexnet)->default_value(false), "if true, generates hypotheses using AlexNet features ")
            ("do_svm_cross_validation", po::value<bool>(&paramSVM.do_cross_validation_)->default_value(paramSVM.do_cross_validation_), "if true, does k cross validation on the svm training data")
            ("alexnet_svm_model", po::value<std::string>(&alexnet_svm_model_fn)->default_value(""), "filename to read svm model for AlexNet. If set (file exists), training is skipped and this model loaded instead.")

            ("max_corr_distance", po::value<double>(&param_.max_corr_distance_)->default_value(param_.max_corr_distance_,  boost::str(boost::format("%.2e") % param_.max_corr_distance_)), "defines the margin for the bounding box used when doing pose refinement with ICP of the cropped scene to the model")
            ("merge_close_hypotheses", po::value<bool>(&param_.merge_close_hypotheses_)->default_value(param_.merge_close_hypotheses_), "if true, close correspondence clusters (object hypotheses) of the same object model are merged together and this big cluster is refined")
            ("merge_close_hypotheses_dist", po::value<double>(&param_.merge_close_hypotheses_dist_)->default_value(param_.merge_close_hypotheses_dist_, boost::str(boost::format("%.2e") % param_.merge_close_hypotheses_dist_)), "defines the maximum distance of the centroids in meter for clusters to be merged together")
            ("merge_close_hypotheses_angle", po::value<double>(&param_.merge_close_hypotheses_angle_)->default_value(param_.merge_close_hypotheses_angle_, boost::str(boost::format("%.2e") % param_.merge_close_hypotheses_angle_) ), "defines the maximum angle in degrees for clusters to be merged together")

            ("normal_method,n", po::value<int>(&normal_computation_method)->default_value(normal_computation_method), "chosen normal computation method of the V4R library")
;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(arguments).options(desc).allow_unregistered().run();
    std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; to_pass_further.push_back("-h"); }
    try { po::notify(vm); }
    catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }

    typename HypothesisVerification<PointT, PointT>::Parameter paramGHV;
    to_pass_further = paramGHV.init(to_pass_further);
    hv_algorithm_.reset(new HypothesisVerification<PointT, PointT>(paramGHV) );

    typename GraphGeometricConsistencyGrouping<PointT, PointT>::Parameter paramGgcg;
    to_pass_further = paramGgcg.init(to_pass_further);
    cg_algorithm_.reset( new GraphGeometricConsistencyGrouping<PointT, PointT> (paramGgcg));


    if( !to_pass_further.empty() )  ///TODO: check this after adding the recognizers
    {
        std::cerr << "Unused command line arguments: ";
        for(size_t c=0; c<to_pass_further.size(); c++)
            std::cerr << to_pass_further[c] << " ";

        std::cerr << "!" << std::endl;
    }

    paramSIFT.normal_computation_method_ = paramSHOT.normal_computation_method_ = param_.normal_computation_method_ = normal_computation_method;

    typename RegisteredViewsSource<PointT>::Ptr src (new RegisteredViewsSource<PointT>(resolution_mm));
    src->setPath (models_dir);
    src->generate ();

    if (do_sift)
    {
        typename SIFTLocalEstimation<PointT>::Ptr estimator (new SIFTLocalEstimation<PointT>());
        estimator->setMaxDistance(sift_z);
        estimator->param_.dense_extraction_ = sift_make_dense;
        estimator->param_.stride_ = sift_kp_stride;

        typename LocalRecognitionPipeline<PointT>::Ptr sift_r (new LocalRecognitionPipeline<PointT> (paramSIFT));
        sift_r->setDataSource (src);
        sift_r->setModelsDir (models_dir);
        sift_r->setFeatureEstimator (estimator);

        addRecognizer (sift_r);
    }

    if (do_shot)
    {

        typename LocalRecognitionPipeline<PointT>::Ptr local (new LocalRecognitionPipeline<PointT> (paramSHOT));
        local->setDataSource (src);
        local->setModelsDir (models_dir);

        if (shot_use_color)
        {
            throw std::runtime_error("SHOT Color not implemented right now.");
//            typename SHOTColorLocalEstimation<PointT>::Ptr shot_estimator (new SHOTColorLocalEstimation<PointT>(shot_support_radius));
//              local->setFeatureEstimator (shot_estimator);
        }
        else
        {
            typename SHOTLocalEstimation<PointT>::Ptr shot_estimator (new SHOTLocalEstimation<PointT>(shot_support_radius));
            local->setFeatureEstimator (shot_estimator);
        }


        if ( keypoint_type == KeypointType::UniformSampling )
        {
            typename UniformSamplingExtractor<PointT>::Ptr extr ( new UniformSamplingExtractor<PointT>(0.01f));
            typename KeypointExtractor<PointT>::Ptr keypoint_extractor = boost::static_pointer_cast<KeypointExtractor<PointT> > (extr);
            local->addKeypointExtractor (keypoint_extractor);
        }

#if PCL_VERSION >= 100702
        if ( keypoint_type == KeypointType::ISS )
        {
            typename IssKeypointExtractor<PointT>::Ptr extr (new IssKeypointExtractor<PointT>);
            typename KeypointExtractor<PointT>::Ptr keypoint_extractor = boost::static_pointer_cast<KeypointExtractor<PointT> > (extr);
            local->addKeypointExtractor (keypoint_extractor);
        }
#endif

        if ( keypoint_type == KeypointType::NARF )
        {
            typename NarfKeypointExtractor<PointT>::Ptr extr (new NarfKeypointExtractor<PointT>);
            typename KeypointExtractor<PointT>::Ptr keypoint_extractor = boost::static_pointer_cast<KeypointExtractor<PointT> > (extr);
            local->addKeypointExtractor (keypoint_extractor);
        }

#if PCL_VERSION >= 100702
        if ( keypoint_type == KeypointType::HARRIS3D )
        {
            typename Harris3DKeypointExtractor<PointT>::Ptr extr (new Harris3DKeypointExtractor<PointT>);
            typename KeypointExtractor<PointT>::Ptr keypoint_extractor = boost::static_pointer_cast<KeypointExtractor<PointT> > (extr);
            local->addKeypointExtractor (keypoint_extractor);
        }
#endif
        addRecognizer(local);
    }

    if (do_esf)
    {
        // feature type
        typename ESFEstimation<PointT>::Ptr est (new ESFEstimation<PointT>);
//        typename OURCVFHEstimator<PointT>::Ptr est (new OURCVFHEstimator<PointT>);

        // segmentation type
        typename DominantPlaneSegmenter<PointT>::Parameter param;
        std::vector<std::string> not_used_params = param.init(to_pass_further);
        typename DominantPlaneSegmenter<PointT>::Ptr seg (new DominantPlaneSegmenter<PointT> (param));
//        typename EuclideanSegmenter<PointT>::Ptr seg (new EuclideanSegmenter<PointT> (argc, argv));
//        typename SmoothEuclideanSegmenter<PointT>::Ptr seg (new SmoothEuclideanSegmenter<PointT> (argc, argv));

        // classifier
//        NearestNeighborClassifier::Ptr classifier (new NearestNeighborClassifier);
        svmClassifier::Ptr classifier (new svmClassifier (paramSVM));
        classifier->setInFilename(esf_svm_model_fn);

        typename GlobalRecognizer<PointT>::Ptr global_r (new GlobalRecognizer<PointT>(paramESF));
        global_r->setSegmentationAlgorithm(seg);
        global_r->setFeatureEstimator(est);
        global_r->setDataSource(src);
        global_r->setModelsDir(models_dir);
        global_r->setClassifier(classifier);
        addRecognizer(global_r);
    }


#ifdef HAVE_CAFFE
    if (do_alexnet)
    {
        // feature type
        typename CNN_Feat_Extractor<PointT>::Parameter cnn_params;
        std::vector<std::string> not_used_params = cnn_params.init(to_pass_further);
        typename CNN_Feat_Extractor<PointT>::Ptr est (new CNN_Feat_Extractor<PointT>(cnn_params));

        // segmentation type
        typename DominantPlaneSegmenter<PointT>::Parameter param;
        not_used_params = param.init(not_used_params);
        typename DominantPlaneSegmenter<PointT>::Ptr seg (new DominantPlaneSegmenter<PointT> (param));

        // classifier
//        NearestNeighborClassifier::Ptr classifier (new NearestNeighborClassifier);
        svmClassifier::Ptr classifier (new svmClassifier (paramSVM));
        classifier->setInFilename(alexnet_svm_model_fn);

        typename GlobalRecognizer<PointT>::Ptr global_r (new GlobalRecognizer<PointT>(paramESF));
        global_r->setSegmentationAlgorithm(seg);
        global_r->setFeatureEstimator(est);
        global_r->setDataSource(src);
        global_r->setModelsDir(models_dir);
        global_r->setClassifier(classifier);
        addRecognizer(global_r);
    }
#endif

    this->setDataSource(src);
    initialize(false);
}

}
