/******************************************************************************
 * Copyright (c) 2013 Federico Tombari, Aitor Aldoma
 * Copyright (c) 2016 Thomas Faeulhammer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/

#ifndef V4R_HYPOTHESIS_VERIFICATION_H__
#define V4R_HYPOTHESIS_VERIFICATION_H__

#include <v4r/core/macros.h>
#include <v4r/common/color_transforms.h>
#include <v4r/common/plane_model.h>
#include <v4r/recognition/ghv_opt.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/recognition/object_hypothesis.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>
#include <metslib/mets.hh>
#include <glog/logging.h>

#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>
namespace po = boost::program_options;

namespace v4r
{

/**
   * \brief A hypothesis verification method for 3D Object Instance Recognition
   * \author Thomas Faeulhammer (based on the work of Federico Tombari and Aitor Aldoma)
   * \date April, 2016
   */
template<typename ModelT, typename SceneT>
class V4R_EXPORTS HypothesisVerification
{
    friend class GHVmove_manager<ModelT, SceneT>;
    friend class GHVSAModel<ModelT, SceneT>;

public:
    class V4R_EXPORTS Parameter
    {
    public:
        int resolution_mm_; /// @brief The resolution of models and scene used to verify hypotheses (in meters)
        double inliers_threshold_; /// @brief Represents the maximum distance between model and scene points in order to state that a scene point is explained by a model point. Valid model points that do not have any corresponding scene point within this threshold are considered model outliers
        double occlusion_thres_;    /// @brief Threshold for a point to be considered occluded when model points are back-projected to the scene ( depends e.g. on sensor noise)
        int zbuffer_self_occlusion_resolution_;
        double focal_length_; /// @brief defines the focal length used for back-projecting points to the image plane (used for occlusion / visibility reasoning)
        int img_width_; /// @brief image width of the camera in pixel (used for computing pairwise intersection)
        int img_height_;  /// @brief image height of the camera in pixel (used for computing pairwise intersection)
        int smoothing_radius_; /// @brief radius in pixel used for smoothing the visible image mask of an object hypotheses (used for computing pairwise intersection)
        bool do_smoothing_;   /// @brief if true, smoothes the silhouette of the reproject object hypotheses (used for computing pairwise intersection)
        bool do_erosion_; /// @brief if true, performs erosion on the silhouette of the reproject object hypotheses. This should avoid a pairwise cost for touching objects (used for computing pairwise intersection)
        int erosion_radius_;  /// @brief erosion radius in px (used for computing pairwise intersection)
        bool do_occlusion_reasoning_;
        int icp_iterations_;

        double color_sigma_l_; /// @brief allowed illumination (L channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)
        double color_sigma_ab_; /// @brief allowed chrominance (AB channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)
        double sigma_normals_deg_; /// @brief variance for normals between model and scene
        double regularizer_; /// @brief represents a penalty multiplier for model outliers. In particular, each model outlier associated with an active hypothesis increases the global cost function.
        double radius_neighborhood_clutter_; /// @brief defines the maximum distance between an <i>explained</i> scene point <b>p</b> and other unexplained scene points such that they influence the clutter term associated with <b>p</b>
        int normal_method_; /// @brief method used for computing the normals of the downsampled scene point cloud (defined by the V4R Library)
        bool ignore_color_even_if_exists_; /// @brief if true, only checks 3D Eucliden distance of neighboring points
        int max_iterations_; /// @brief max iterations the optimization strategy explores local neighborhoods before stopping because the cost does not decrease.
        double clutter_regularizer_; /// @brief The penalty multiplier used to penalize unexplained scene points within the clutter influence radius <i>radius_neighborhood_clutter_</i> of an explained scene point when they belong to the same smooth segment.
        bool use_replace_moves_; /// @brief parameter for optimization. If true, local search uses replace moves (deactivatives one hypotheses and activates another one). Otherwise, it only searches locally by enabling/disabling hypotheses one at a time.
        int opt_type_; /// @brief defines the optimization methdod<BR><BR> 0: Local search (converges quickly, but can easily get trapped in local minima),<BR> 1: Tabu Search,<BR> 2; Tabu Search + Local Search (Replace active hypotheses moves),<BR> 3: Simulated Annealing
        bool use_histogram_specification_; /// @brief if true, tries to globally match brightness (L channel of LAB color space) of visible hypothesis cloud to brightness of nearby scene points. It does so by computing the L channel histograms for both clouds and shifting it to maximize histogram intersection.
        bool initial_status_; /// @brief sets the initial activation status of each hypothesis to this value before starting optimization. E.g. If true, all hypotheses will be active and the cost will be optimized from that initial status.
        int color_space_; /// @brief specifies the color space being used for verification (0... LAB, 1... RGB, 2... Grayscale,  3,4,5,6... ???)
        bool use_noise_model_;  /// @brief if set, uses Nguyens noise model for setting threshold parameters
        bool visualize_go_cues_; /// @brief visualizes the cues during the computation and shows cost and number of evaluations. Useful for debugging
        bool visualize_model_cues_; /// @brief visualizes the model cues. Useful for debugging
        bool visualize_pairwise_cues_; /// @brief visualizes the pairwise cues. Useful for debugging
        int knn_inliers_; /// @brief number of nearby scene points to check for a query model point

        float min_visible_ratio_; /// @brief defines how much of the object has to be visible in order to be included in the verification stage
        float min_model_fitness_lower_bound_; /// @brief defines the lower bound (i.e. when model visibility is min_visible_ratio_) of the fitness threshold for a hypothesis to be kept for optimization (0... no threshold, 1... everything gets rejected)
        float min_model_fitness_upper_bound_; /// @brief defines the upper bound (i.e. when model visibility is 0.5) of the fitness threshold for a hypothesis to be kept for optimization (0... no threshold, 1... everything gets rejected)
        int knn_color_neighborhood_; /// @brief number of nearest neighbors used for describing the color around a point
        float color_std_dev_multiplier_threshold_; /// @brief standard deviation multiplier threshold for the local color description for each color channel

        bool check_plane_intersection_; /// @brief if true, extracts planes and checks if they intersect with hypotheses
        int plane_method_;  /// method used for plane extraction (0... RANSAC (only available for single-view), 1... region growing)

        //Euclidean smooth segmenation
        bool check_smooth_clusters_;
        float eps_angle_threshold_deg_;
        float curvature_threshold_;
        float cluster_tolerance_;
        int min_points_;
        float min_ratio_cluster_explained_; /// @brief defines the minimum ratio a smooth cluster has to be explained by the visible points (given there are at least 100 points)
        bool z_adaptive_;   /// @brief if true, scales the smooth segmentation parameters linear with distance (constant till 1m at the given parameters)

        bool vis_for_paper_; /// @brief optimizes visualization for paper (white background, no text labels...)

        Parameter (
                int resolution_mm = 5,
                double inliers_threshold = 0.01f, // 0.005f
                double occlusion_thres = 0.01f, // 0.005f
                int zbuffer_self_occlusion_resolution = 250,
                double focal_length = 525.f,
                int img_width = 640,
                int img_height = 480,
                int smoothing_radius = 2,
                bool do_smoothing = true,
                bool do_erosion = true,
                int erosion_radius = 4,
                bool do_occlusion_reasoning = true,
                int icp_iterations = 10,
                double color_sigma_l = 100.f,
                double color_sigma_ab = 20.f,
                double sigma_normals_deg = 30.f,
                double regularizer = 1.f,
                double radius_neighborhood_clutter = 0.02f,
                int normal_method = 2,
                bool ignore_color_even_if_exists = false,
                int max_iterations = 5000,
                double clutter_regularizer =  10000.f,
                bool use_replace_moves = true,
                int opt_type = OptimizationType::LocalSearch,
                bool use_histogram_specification = false,
                bool initial_status = false,
                int color_space = ColorTransformOMP::LAB,
                bool use_noise_model = true,
                bool visualize_go_cues = false,
                int knn_inliers = 3,
                float min_visible_ratio = 0.15f,
                float min_model_fitness_lower_bound = 0.20f,
                float min_model_fitness_upper_bound = 0.40f,
                int knn_color_neighborhood = 10,
                float color_std_dev_multiplier_threshold = 1.f,
                bool check_plane_intersection = true,
                int plane_method = 2,
                bool check_smooth_clusters = true,
                float eps_angle_threshold_deg = 5.f, //0.25rad
                float curvature_threshold = 0.04f,
                float cluster_tolerance = 0.01f, //0.015f;
                int min_points = 100, // 20
                float min_ratio_cluster_explained = 0.5,
                bool z_adaptive = true,
                bool vis_for_paper = false)
            : resolution_mm_ (resolution_mm),
              inliers_threshold_(inliers_threshold),
              occlusion_thres_ (occlusion_thres),
              zbuffer_self_occlusion_resolution_(zbuffer_self_occlusion_resolution),
              focal_length_ (focal_length),
              img_width_ (img_width),
              img_height_ (img_height),
              smoothing_radius_ (smoothing_radius),
              do_smoothing_ (do_smoothing),
              do_erosion_ (do_erosion),
              erosion_radius_ (erosion_radius),
              do_occlusion_reasoning_ (do_occlusion_reasoning),
              icp_iterations_ (icp_iterations),
              color_sigma_l_ (color_sigma_l),
              color_sigma_ab_ (color_sigma_ab),
              sigma_normals_deg_ (sigma_normals_deg),
              regularizer_ (regularizer),
              radius_neighborhood_clutter_ (radius_neighborhood_clutter),
              normal_method_ (normal_method),
              ignore_color_even_if_exists_ (ignore_color_even_if_exists),
              max_iterations_ (max_iterations),
              clutter_regularizer_ (clutter_regularizer),
              use_replace_moves_ (use_replace_moves),
              opt_type_ (opt_type),
              use_histogram_specification_ (use_histogram_specification),
              initial_status_ (initial_status),
              color_space_ (color_space),
              use_noise_model_ (use_noise_model),
              visualize_go_cues_ ( visualize_go_cues ),
              knn_inliers_ (knn_inliers),
              min_visible_ratio_ (min_visible_ratio),
              min_model_fitness_lower_bound_ (min_model_fitness_lower_bound),
              min_model_fitness_upper_bound_ (min_model_fitness_upper_bound),
              knn_color_neighborhood_ (knn_color_neighborhood),
              color_std_dev_multiplier_threshold_ (color_std_dev_multiplier_threshold),
              check_plane_intersection_ ( check_plane_intersection ),
              plane_method_ (plane_method),
              check_smooth_clusters_ ( check_smooth_clusters ),
              eps_angle_threshold_deg_ (eps_angle_threshold_deg),
              curvature_threshold_ (curvature_threshold),
              cluster_tolerance_ (cluster_tolerance),
              min_points_ (min_points),
              min_ratio_cluster_explained_ ( min_ratio_cluster_explained ),
              z_adaptive_ ( z_adaptive ),
              vis_for_paper_ ( vis_for_paper )
        {}


        /**
         * @brief init parameters
         * @param command_line_arguments (according to Boost program options library)
         * @return unused parameters (given parameters that were not used in this initialization call)
         */
        std::vector<std::string>
        init(int argc, char **argv)
        {
                std::vector<std::string> arguments(argv + 1, argv + argc);
                return init(arguments);
        }

        /**
         * @brief init parameters
         * @param command_line_arguments (according to Boost program options library)
         * @return unused parameters (given parameters that were not used in this initialization call)
         */
        std::vector<std::string>
        init(const std::vector<std::string> &command_line_arguments)
        {
            po::options_description desc("Hypothesis Verification Parameters\n=====================");
            desc.add_options()
                    ("help,h", "produce help message")
                    ("hv_icp_iterations", po::value<int>(&icp_iterations_)->default_value(icp_iterations_), "number of icp iterations. If 0, no pose refinement will be done")
                    ("hv_clutter_regularizer", po::value<double>(&clutter_regularizer_)->default_value(clutter_regularizer_, boost::str(boost::format("%.2e") % clutter_regularizer_) ), "The penalty multiplier used to penalize unexplained scene points within the clutter influence radius <i>radius_neighborhood_clutter_</i> of an explained scene point when they belong to the same smooth segment.")
                    ("hv_color_sigma_ab", po::value<double>(&color_sigma_ab_)->default_value(color_sigma_ab_, boost::str(boost::format("%.2e") % color_sigma_ab_) ), "allowed chrominance (AB channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)")
                    ("hv_color_sigma_l", po::value<double>(&color_sigma_l_)->default_value(color_sigma_l_, boost::str(boost::format("%.2e") % color_sigma_l_) ), "allowed illumination (L channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)")
                    ("hv_sigma_normals_deg", po::value<double>(&sigma_normals_deg_)->default_value(sigma_normals_deg_, boost::str(boost::format("%.2e") % sigma_normals_deg_) ), "variance for surface normals")
                    ("hv_histogram_specification", po::value<bool>(&use_histogram_specification_)->default_value(use_histogram_specification_), " ")
                    ("hv_ignore_color", po::value<bool>(&ignore_color_even_if_exists_)->default_value(ignore_color_even_if_exists_), " ")
                    ("hv_initial_status", po::value<bool>(&initial_status_)->default_value(initial_status_), "sets the initial activation status of each hypothesis to this value before starting optimization. E.g. If true, all hypotheses will be active and the cost will be optimized from that initial status.")
                    ("hv_color_space", po::value<int>(&color_space_)->default_value(color_space_), "specifies the color space being used for verification (0... LAB, 1... RGB, 2... Grayscale,  3,4,5,6... ?)")
                    ("hv_color_stddev_mul", po::value<float>(&color_std_dev_multiplier_threshold_)->default_value(color_std_dev_multiplier_threshold_), "standard deviation multiplier threshold for the local color description for each color channel")
                    ("hv_inlier_threshold", po::value<double>(&inliers_threshold_)->default_value(inliers_threshold_, boost::str(boost::format("%.2e") % inliers_threshold_) ), "Represents the maximum distance between model and scene points in order to state that a scene point is explained by a model point. Valid model points that do not have any corresponding scene point within this threshold are considered model outliers")
                    ("hv_occlusion_threshold", po::value<double>(&occlusion_thres_)->default_value(occlusion_thres_, boost::str(boost::format("%.2e") % occlusion_thres_) ), "Threshold for a point to be considered occluded when model points are back-projected to the scene ( depends e.g. on sensor noise)")
                    ("hv_optimizer_type", po::value<int>(&opt_type_)->default_value(opt_type_), "defines the optimization methdod. 0: Local search (converges quickly, but can easily get trapped in local minima), 1: Tabu Search, 4; Tabu Search + Local Search (Replace active hypotheses moves), else: Simulated Annealing")
                    ("hv_radius_clutter", po::value<double>(&radius_neighborhood_clutter_)->default_value(radius_neighborhood_clutter_, boost::str(boost::format("%.2e") % radius_neighborhood_clutter_) ), "defines the maximum distance between two points to be checked for label consistency")
                    ("hv_regularizer,r", po::value<double>(&regularizer_)->default_value(regularizer_, boost::str(boost::format("%.2e") % regularizer_) ), "represents a penalty multiplier for model outliers. In particular, each model outlier associated with an active hypothesis increases the global cost function.")
                    ("hv_min_model_fitness_lower_bound", po::value<float>(&min_model_fitness_lower_bound_)->default_value(min_model_fitness_lower_bound_, boost::str(boost::format("%.2e") % min_model_fitness_lower_bound_) ), "defines the fitness threshold for a hypothesis to be kept for optimization (0... no threshold, 1... everything gets rejected)")
                    ("hv_min_model_fitness_upper_bound", po::value<float>(&min_model_fitness_upper_bound_)->default_value(min_model_fitness_upper_bound_, boost::str(boost::format("%.2e") % min_model_fitness_upper_bound_) ), "defines the fitness threshold for a hypothesis to be kept for optimization (0... no threshold, 1... everything gets rejected)")
                    ("hv_min_visible_ratio", po::value<float>(&min_visible_ratio_)->default_value(min_visible_ratio_, boost::str(boost::format("%.2e") % min_visible_ratio_) ), "defines how much of the object has to be visible in order to be included in the verification stage")
                    ("hv_min_ratio_smooth_cluster_explained", po::value<float>(&min_ratio_cluster_explained_)->default_value(min_ratio_cluster_explained_, boost::str(boost::format("%.2e") % min_ratio_cluster_explained_) ), " defines the minimum ratio a smooth cluster has to be explained by the visible points (given there are at least 100 points)")
                    ("hv_eps_angle_threshold", po::value<float>(&eps_angle_threshold_deg_)->default_value(eps_angle_threshold_deg_), "smooth clustering parameter for the angle threshold")
                    ("hv_cluster_tolerance", po::value<float>(&cluster_tolerance_)->default_value(cluster_tolerance_), "smooth clustering parameter for cluster_tolerance")
                    ("hv_curvature_threshold", po::value<float>(&curvature_threshold_)->default_value(curvature_threshold_), "smooth clustering parameter for curvate")
                    ("hv_check_smooth_clusters", po::value<bool>(&check_smooth_clusters_)->default_value(check_smooth_clusters_), "if true, checks for each hypotheses how well it explains occupied smooth surface patches. Hypotheses are rejected if they only partially explain smooth clusters.")

                    ("hv_vis_cues", po::bool_switch(&visualize_go_cues_), "If set, visualizes cues computated at the hypothesis verification stage such as inlier, outlier points. Mainly used for debugging.")
                    ("hv_vis_model_cues", po::bool_switch(&visualize_model_cues_), "If set, visualizes the model cues. Useful for debugging")
                    ("hv_vis_pairwise_cues", po::bool_switch(&visualize_pairwise_cues_), "If set, visualizes the pairwise cues. Useful for debugging")
                    ;
            po::variables_map vm;
            po::parsed_options parsed = po::command_line_parser(command_line_arguments).options(desc).allow_unregistered().run();
            std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
            po::store(parsed, vm);
            if (vm.count("help")) { std::cout << desc << std::endl; to_pass_further.push_back("-h"); }
            try { po::notify(vm); }
            catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }
            return to_pass_further;
        }

    }param_;

private:
    mutable pcl::visualization::PCLVisualizer::Ptr vis_go_cues_, rm_vis_, vis_pairwise_;
    mutable std::vector<std::string> coordinate_axes_ids_;
    mutable int vp_active_hypotheses_, vp_scene_, vp_model_fitness_, vp_scene_fitness_;
    mutable int rm_v1, rm_v2, rm_v3, rm_v4, rm_v5, rm_v6, rm_v7, rm_v8, rm_v9, rm_v10, rm_v11, rm_v12, vp_pair_1_, vp_pair_2_;

    std::vector<bool> solution_; /// @brief Boolean vector indicating if a hypothesis is accepted (true) or rejected (false)

    typename pcl::PointCloud<SceneT>::ConstPtr scene_cloud_; /// @brief scene point clou
    typename pcl::PointCloud<pcl::Normal>::ConstPtr scene_normals_; /// @brief scene normals cloud
    typename pcl::PointCloud<SceneT>::Ptr scene_cloud_downsampled_; /// \brief Downsampled scene point cloud
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals_downsampled_; /// \brief Downsampled scene normals cloud
    std::vector<int> scene_sampled_indices_;    /// @brief downsampled indices of the scene

    std::vector<std::vector<typename HVRecognitionModel<ModelT>::Ptr > > obj_hypotheses_groups_;
    std::vector<typename HVRecognitionModel<ModelT>::Ptr > global_hypotheses_;  /// @brief all hypotheses not rejected by individual verification

    std::vector<int> recognition_models_map_;
    std::vector<boost::shared_ptr<HVRecognitionModel<ModelT> > > recognition_models_; /// @brief all models to be verified (including planar models if included)

    double model_fitness_, pairwise_cost_, scene_fitness_, cost_;
    Eigen::VectorXf model_fitness_v_;
    Eigen::VectorXf tmp_solution_;

    float Lmin_ = 0.f, Lmax_ = 100.f;
    int bins_ = 50;

    std::vector<float> sRGB_LUT;
    std::vector<float> sXYZ_LUT;

    Eigen::MatrixXf intersection_cost_; /// @brief represents the pairwise intersection cost
    Eigen::MatrixXf scene_explained_weight_; /// @brief for each point in the scene (row) store how good it is presented from each model (column)
    Eigen::MatrixXf scene_model_sqr_dist_; /// @brief for each point in the scene (row) store the distance to its model point
    Eigen::MatrixXf scene_explained_weight_compressed_; /// @brief for each point in the scene (row) store how good it is presented from each model (column)
    Eigen::VectorXf max_scene_explained_weight_; /// @brief for each point in the scene (row) store how good it is presented from each model (column)

    GHVSAModel<ModelT, SceneT> best_seen_;
    float initial_temp_;
    boost::shared_ptr<GHVCostFunctionLogger<ModelT,SceneT> > cost_logger_;
    Eigen::MatrixXf scene_color_channels_; /// @brief converted color values where each point corresponds to a row entry
    typename pcl::octree::OctreePointCloudSearch<SceneT>::Ptr octree_scene_downsampled_;
    boost::function<void (const std::vector<bool> &, float, int)> visualize_cues_during_logger_;

    std::vector<int> scene_smooth_labels_;  /// @brief stores a label for each point of the (downsampled) scene. Points belonging to the same smooth clusters, have the same label
    std::vector<size_t> smooth_label_count_;    /// @brief counts how many times a certain smooth label occurs in the scene

    std::vector<typename PlaneModel<SceneT>::Ptr > planes_; /// @brief all extracted planar surfaces (if check enabled)

    // ----- MULTI-VIEW VARIABLES------
    std::vector<typename pcl::PointCloud<SceneT>::ConstPtr> occlusion_clouds_; /// @brief scene clouds from multiple views
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > absolute_camera_poses_;
    std::vector<std::vector<bool> > model_is_present_in_view_; /// \brief for each model this variable stores information in which view it is present (used to check for visible model points - default all true = static scene)

    Eigen::Matrix4f poseRefinement(HVRecognitionModel<ModelT> &rm) const;

    void computeModelOcclusionByScene(HVRecognitionModel<ModelT> &rm, const std::vector<Eigen::MatrixXf> &depth_image_scene); /// @brief computes the visible points of the model in the given pose and the provided depth map(s) of the scene

    void computeVisibleModelsAndRefinePose();

    void downsampleSceneCloud ();    /// \brief downsamples the scene cloud

    void removeSceneNans ();    /// \brief removes nan values from the downsampled scene cloud

    bool removeNanNormals (HVRecognitionModel<ModelT> & recog_model); /// @brief remove all points from visible cloud and normals which are not-a-number

    template<typename PointT> void convertColor(const typename pcl::PointCloud<PointT> &cloud, Eigen::MatrixXf &color_mat); /// @brief converting points from RGB to desired color space

    void computeModel2SceneFitness (HVRecognitionModel<ModelT> &rm); /// @brief checks for each visible point in the model if there are nearby scene points and how good they match

    void removeModelsWithLowVisibility (); /// @brief remove recognition models where there are not enough visible points (>95% occluded)

    void computePairwiseIntersection (); /// @brief computes the overlap of two visible points when projected to camera view

    void computeModel2SceneDistances (HVRecognitionModel<ModelT> &rm);

    void computeLoffset (HVRecognitionModel<ModelT> &rm) const;

    void computePlanarSurfaces ( );

    void computePlaneIntersection ( );

    void initialize ();

    mets::gol_type evaluateSolution (const std::vector<bool> & active, int changed);

    std::vector<bool> optimize ();

    void visualizeGOCues (const std::vector<bool> & active_solution, float cost_, int times_eval) const;

    void visualizePairwiseIntersection () const;

    void visualizeGOCuesForModel (const HVRecognitionModel<ModelT> &rm) const;

    bool individualRejection(HVRecognitionModel<ModelT> &rm); /// @brief remove hypotheses badly explaining the scene, or only partially explaining smooth clusters (if check is enabled). Returns true if hypothesis is rejected.

    void cleanUp ()
    {
        octree_scene_downsampled_.reset();
        this->occlusion_clouds_.clear();
        this->absolute_camera_poses_.clear();
        scene_sampled_indices_.clear();
        this->model_is_present_in_view_.clear();
        scene_cloud_downsampled_.reset();
        scene_cloud_.reset();
        intersection_cost_.resize(0,0);
        scene_explained_weight_.resize(0,0);
        scene_model_sqr_dist_.resize(0,0);
        scene_explained_weight_compressed_.resize(0,0);
        max_scene_explained_weight_.resize(0);
        obj_hypotheses_groups_.clear();
    }

    void
    rgb2cielab (uint8_t R, uint8_t G, uint8_t B, float &L, float &A,float &B2)
    {
        CHECK(R < 256 && R >= 0);
        CHECK(G < 256 && G >= 0);
        CHECK(B < 256 && B >= 0);

        float fr = sRGB_LUT[R];
        float fg = sRGB_LUT[G];
        float fb = sRGB_LUT[B];

        // Use white = D65
        const float x = fr * 0.412453f + fg * 0.357580f + fb * 0.180423f;
        const float y = fr * 0.212671f + fg * 0.715160f + fb * 0.072169f;
        const float z = fr * 0.019334f + fg * 0.119193f + fb * 0.950227f;

        float vx = x / 0.95047f;
        float vy = y;
        float vz = z / 1.08883f;

        vx = sXYZ_LUT[std::min(int(vx*4000), 4000-1)];
        vy = sXYZ_LUT[std::min(int(vy*4000), 4000-1)];
        vz = sXYZ_LUT[std::min(int(vz*4000), 4000-1)];

        L = 116.0f * vy - 16.0f;
        if (L > 100)
            L = 100.0f;

        A = 500.0f * (vx - vy);
        if (A > 120)
            A = 120.0f;
        else if (A <- 120)
            A = -120.0f;

        B2 = 200.0f * (vy - vz);
        if (B2 > 120)
            B2 = 120.0f;
        else if (B2<- 120)
            B2 = -120.0f;
    }

    void
    convertToLABcolor(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, Eigen::MatrixXf &color_mat)
    {
        color_mat.resize( cloud.points.size(), 3);

        for(size_t j=0; j < cloud.points.size(); j++)
        {
            const pcl::PointXYZRGB &p = cloud.points[j];

            uint8_t rs, gs, bs;
            float LRef, aRef, bRef;
            rs = p.r;
            gs = p.g;
            bs = p.b;
            rgb2cielab(rs, gs, bs, LRef, aRef, bRef);
            color_mat(j, 0) = LRef;
            color_mat(j, 1) = aRef;
            color_mat(j, 2) = bRef;
        }
    }


    void extractEuclideanClustersSmooth ();

    void registerModelAndSceneColor(std::vector<size_t> &lookup, HVRecognitionModel<ModelT> & recog_model);

    /**
     * @brief creates look-up table to register colors from src to destination
     * @param[in] src color channels (row entries are the bins of the histogram, col entries the various dimensions)
     * @param[in] dst color channels (row entries are the bins of the histogram, col entries the various dimensions)
     * @param[out] lookup table
     */
    void specifyHistogram (const Eigen::MatrixXf & src, const Eigen::MatrixXf & dst, Eigen::MatrixXf & lookup) const;

    typedef pcl::PointCloud<ModelT> CloudM;
    typedef pcl::PointCloud<SceneT> CloudS;
    typedef typename pcl::traits::fieldList<typename CloudS::PointType>::type FieldListS;
    typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

public:
    HypothesisVerification (const Parameter &p = Parameter()) : param_(p)
    {
        initial_temp_ = 1000;
        sRGB_LUT.resize(256, -1);
        sXYZ_LUT.resize(4000, -1);

        // init color LUTs
        for (int i = 0; i < 256; i++)
        {
            float f = static_cast<float> (i) / 255.0f;
            if (f > 0.04045)
                sRGB_LUT[i] = powf ((f + 0.055f) / 1.055f, 2.4f);
            else
                sRGB_LUT[i] = f / 12.92f;
        }

        for (int i = 0; i < 4000; i++)
        {
            float f = static_cast<float> (i) / 4000.0f;
            if (f > 0.008856)
                sXYZ_LUT[i] = static_cast<float> (powf (f, 0.3333f));
            else
                sXYZ_LUT[i] = static_cast<float>((7.787 * f) + (16.0 / 116.0));
        }
    }

    /**
     *  \brief: Returns a vector of booleans representing which hypotheses have been accepted/rejected (true/false)
     *  mask vector of booleans
     */
    void
    getMask (std::vector<bool> & mask) const
    {
        mask = solution_;
    }

    /**
     * @brief returns the vector of verified object hypotheses
     * @return
     */
    std::vector<typename ObjectHypothesis<ModelT>::Ptr >
    getVerifiedHypotheses() const
    {
        std::vector<typename ObjectHypothesis<ModelT>::Ptr > verified_hypotheses  (global_hypotheses_.size());

        size_t kept=0;
        for(size_t i=0; i<global_hypotheses_.size(); i++)
        {
            if(solution_[i])
            {
                verified_hypotheses[kept] = global_hypotheses_[i];
                kept++;
            }
        }
        verified_hypotheses.resize(kept);
        return verified_hypotheses;
    }

    /**
     * @brief Sets the models (recognition hypotheses)
     * @param models vector of point clouds representing the models (in same coordinates as the scene_cloud_)
     * @param corresponding normal clouds
     */
    void
    addModels (const std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models,
               const std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr > &model_normals);


    /**
     *  \brief Sets the scene cloud
     *  \param scene_cloud Point cloud representing the scene
     */
    void
    setSceneCloud (const typename pcl::PointCloud<SceneT>::Ptr & scene_cloud)
    {
        scene_cloud_ = scene_cloud;
    }

    /**
     *  \brief Sets the scene normals cloud
     *  \param normals Point cloud representing the scene normals
     */
    void
    setNormals (const pcl::PointCloud<pcl::Normal>::Ptr & normals)
    {
        scene_normals_ = normals;
    }

    /**
     * @brief set Occlusion Clouds And Absolute Camera Poses (used for multi-view recognition)
     * @param occlusion clouds
     * @param absolute camera poses
     */
    void
    setOcclusionCloudsAndAbsoluteCameraPoses(const std::vector<typename pcl::PointCloud<SceneT>::ConstPtr > & occ_clouds,
                                             const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &absolute_camera_poses)
    {
        occlusion_clouds_ = occ_clouds;
        absolute_camera_poses_ = absolute_camera_poses;
    }


    /**
     * @brief for each model this variable stores information in which view it is present
     * @param presence in model and view
     */
    void
    setVisibleCloudsForModels(const std::vector<std::vector<bool> > &model_is_present_in_view)
    {
        model_is_present_in_view_ = model_is_present_in_view;
    }

//    /**
//     * @brief returns the refined transformation matrix aligning model with scene cloud (applies to object models only - not plane clouds) and is in order of the input of addmodels
//     * @return
//     */
//    void
//    getRefinedTransforms(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &tf) const
//    {
//        tf = refined_model_transforms_;
//    }


    enum OptimizationType
    {
        LocalSearch,
        TabuSearch,
        TabuSearchWithLSRM,
        SimulatedAnnealing
    };


    void
    setHypotheses(const std::vector<ObjectHypothesesGroup<ModelT> > &ohs)
    {
        obj_hypotheses_groups_.clear();
        obj_hypotheses_groups_.resize(ohs.size());
        for(size_t i=0; i<obj_hypotheses_groups_.size(); i++)
        {
            const ObjectHypothesesGroup<ModelT> &ohg = ohs[i];

            obj_hypotheses_groups_[i].resize(ohg.ohs_.size());
            for(size_t jj=0; jj<ohg.ohs_.size(); jj++)
            {
                const ObjectHypothesis<ModelT> &oh = *ohg.ohs_[jj];
                obj_hypotheses_groups_[i][jj].reset ( new HVRecognitionModel<ModelT>(oh) );
                HVRecognitionModel<ModelT> &hv_oh = *obj_hypotheses_groups_[i][jj];

                hv_oh.complete_cloud_.reset ( new pcl::PointCloud<ModelT>);
                hv_oh.complete_cloud_normals_.reset (new pcl::PointCloud<pcl::Normal>);

                typename pcl::PointCloud<ModelT>::ConstPtr model_cloud = oh.model_->getAssembled ( param_.resolution_mm_ );
                pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud_const = oh.model_->getNormalsAssembled ( param_.resolution_mm_ );
                pcl::transformPointCloud (*model_cloud, *hv_oh.complete_cloud_, oh.transform_);
                transformNormals(*normal_cloud_const, *hv_oh.complete_cloud_normals_, oh.transform_);
            }
        }
    }

    void
    writeToLog (std::ofstream & of, bool all_costs_ = false)
    {
        cost_logger_->writeToLog (of);
        if (all_costs_)
            cost_logger_->writeEachCostToLog (of);
    }


    /**
     *  \brief Function that performs the hypotheses verification
     *  This function modifies the values of mask_ and needs to be called after both scene and model have been added
     */
    void verify();

    typedef boost::shared_ptr< HypothesisVerification<ModelT, SceneT> > Ptr;
    typedef boost::shared_ptr< HypothesisVerification<ModelT, SceneT> const> ConstPtr;
};

}

#endif /* PCL_RECOGNITION_HYPOTHESIS_VERIFICATION_H_ */
