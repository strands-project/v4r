/******************************************************************************
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

#pragma once

#include <fstream>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/serialization.hpp>
#include <v4r/core/macros.h>
namespace po = boost::program_options;

namespace v4r
{

enum HV_OptimizationType
{
    LocalSearch,
    TabuSearch,
    TabuSearchWithLSRM,
    SimulatedAnnealing
};

class V4R_EXPORTS HV_Parameter
{
public:
    int resolution_mm_; ///< The resolution of models and scene used to verify hypotheses (in milli meters)
    float inliers_threshold_; ///< inlier distance in meters between model and scene point
    float inliers_surface_angle_thres_; ///< inlier distance in radians between model and scene surface normals
    float occlusion_thres_;    ///< Threshold for a point to be considered occluded when model points are back-projected to the scene ( depends e.g. on sensor noise)
    int smoothing_radius_; ///< radius in pixel used for smoothing the visible image mask of an object hypotheses (used for computing pairwise intersection)
    bool do_smoothing_;   ///< if true, smoothes the silhouette of the reproject object hypotheses (used for computing pairwise intersection)
    bool do_erosion_; ///< if true, performs erosion on the silhouette of the reproject object hypotheses. This should avoid a pairwise cost for touching objects (used for computing pairwise intersection)
    int erosion_radius_;  ///< erosion radius in px (used for computing pairwise intersection)
    int icp_iterations_;

    float w_normals_;   ///< weighting factor for normal fitness
    float w_color_;   ///< weighting factor for color fitness
    float w_3D_;   ///< weighting factor for 3D fitness

    float color_sigma_l_; ///< allowed illumination (L channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)
    float color_sigma_ab_; ///< allowed chrominance (AB channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)
    float sigma_normals_deg_; ///< variance for normals between model and scene
    float regularizer_; ///< represents a penalty multiplier for model outliers. In particular, each model outlier associated with an active hypothesis increases the global cost function.
    float radius_neighborhood_clutter_; ///< defines the maximum distance between an <i>explained</i> scene point <b>p</b> and other unexplained scene points such that they influence the clutter term associated with <b>p</b>
    int normal_method_; ///< method used for computing the normals of the downsampled scene point cloud (defined by the V4R Library)
    bool ignore_color_even_if_exists_; ///< if true, only checks 3D Eucliden distance of neighboring points
    int max_iterations_; ///< max iterations the optimization strategy explores local neighborhoods before stopping because the cost does not decrease.
    float clutter_regularizer_; ///< The penalty multiplier used to penalize unexplained scene points within the clutter influence radius <i>radius_neighborhood_clutter_</i> of an explained scene point when they belong to the same smooth segment.
    bool use_replace_moves_; ///< parameter for optimization. If true, local search uses replace moves (deactivatives one hypotheses and activates another one). Otherwise, it only searches locally by enabling/disabling hypotheses one at a time.
    int opt_type_; ///< defines the optimization methdod<BR><BR> 0: Local search (converges quickly, but can easily get trapped in local minima),<BR> 1: Tabu Search,<BR> 2; Tabu Search + Local Search (Replace active hypotheses moves),<BR> 3: Simulated Annealing
    bool use_histogram_specification_; ///< if true, tries to globally match brightness (L channel of LAB color space) of visible hypothesis cloud to brightness of nearby scene points. It does so by computing the L channel histograms for both clouds and shifting it to maximize histogram intersection.
    bool initial_status_; ///< sets the initial activation status of each hypothesis to this value before starting optimization. E.g. If true, all hypotheses will be active and the cost will be optimized from that initial status.
//        int color_space_; ///< specifies the color space being used for verification (0... LAB, 1... RGB, 2... Grayscale,  3,4,5,6... ???)
    int color_comparison_method_; ///< method used for color comparison (0... CIE76, 1... CIE94, 2... CIEDE2000)
    bool use_noise_model_;  ///< if set, uses Nguyens noise model for setting threshold parameters
    bool visualize_go_cues_; ///< visualizes the cues during the computation and shows cost and number of evaluations. Useful for debugging
    bool visualize_model_cues_; ///< visualizes the model cues. Useful for debugging
    bool visualize_pairwise_cues_; ///< visualizes the pairwise cues. Useful for debugging
    int knn_inliers_; ///< number of nearby scene points to check for a query model point

    float min_visible_ratio_; ///< defines how much of the object has to be visible in order to be included in the verification stage
    float min_model_fitness_lower_bound_; ///< defines the lower bound (i.e. when model visibility is min_visible_ratio_) of the fitness threshold for a hypothesis to be kept for optimization (0... no threshold, 1... everything gets rejected)
    float min_model_fitness_upper_bound_; ///< defines the upper bound (i.e. when model visibility is 0.5) of the fitness threshold for a hypothesis to be kept for optimization (0... no threshold, 1... everything gets rejected)
    int knn_color_neighborhood_; ///< number of nearest neighbors used for describing the color around a point
    float color_std_dev_multiplier_threshold_; ///< standard deviation multiplier threshold for the local color description for each color channel

    bool check_plane_intersection_; ///< if true, extracts planes and checks if they intersect with hypotheses
    int plane_method_;  ///< method used for plane extraction (0... RANSAC (only available for single-view), 1... region growing)

    //Euclidean smooth segmenation
    bool check_smooth_clusters_;
    float eps_angle_threshold_deg_;
    float curvature_threshold_;
    float cluster_tolerance_;
    int min_points_;
    float min_ratio_cluster_explained_; ///< defines the minimum ratio a smooth cluster has to be explained by the visible points (given there are at least 100 points)
    bool z_adaptive_;   ///< if true, scales the smooth segmentation parameters linear with distance (constant till 1m at the given parameters)

    HV_Parameter (
            int resolution_mm = 5,
            float inliers_threshold = 0.01f, // 0.005f
            float inliers_surface_angle_thres = M_PI / 16,
            float occlusion_thres = 0.01f, // 0.005f
            int smoothing_radius = 2,
            bool do_smoothing = true,
            bool do_erosion = true,
            int erosion_radius = 4,
            int icp_iterations = 10,
            float w_normals = .33f,
            float w_color = .33f,
            float w_3D = .33f,
            float color_sigma_l = 100.f,
            float color_sigma_ab = 20.f,
            float sigma_normals_deg = 30.f,
            float regularizer = 1.f,
            float radius_neighborhood_clutter = 0.02f,
            int normal_method = 2,
            bool ignore_color_even_if_exists = false,
            int max_iterations = 5000,
            float clutter_regularizer =  10000.f,
            bool use_replace_moves = true,
            int opt_type = HV_OptimizationType::LocalSearch,
            bool use_histogram_specification = false,
            bool initial_status = false,
//                int color_space = ColorTransformOMP::LAB,
            int color_comparison = ColorComparisonMethod::ciede2000,
            bool use_noise_model = true,
            bool visualize_go_cues = false,
            int knn_inliers = 5,
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
            bool z_adaptive = true)
        : resolution_mm_ (resolution_mm),
          inliers_threshold_(inliers_threshold),
          inliers_surface_angle_thres_ (inliers_surface_angle_thres),
          occlusion_thres_ (occlusion_thres),
          smoothing_radius_ (smoothing_radius),
          do_smoothing_ (do_smoothing),
          do_erosion_ (do_erosion),
          erosion_radius_ (erosion_radius),
          icp_iterations_ (icp_iterations),
          w_normals_ (w_normals),
          w_color_ (w_color),
          w_3D_  (w_3D),
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
//              color_space_ (color_space),
          color_comparison_method_ (color_comparison),
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
          z_adaptive_ ( z_adaptive )
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
                ("hv_clutter_regularizer", po::value<float>(&clutter_regularizer_)->default_value(clutter_regularizer_, boost::str(boost::format("%.2e") % clutter_regularizer_) ), "The penalty multiplier used to penalize unexplained scene points within the clutter influence radius <i>radius_neighborhood_clutter_</i> of an explained scene point when they belong to the same smooth segment.")
                ("hv_color_sigma_ab", po::value<float>(&color_sigma_ab_)->default_value(color_sigma_ab_, boost::str(boost::format("%.2e") % color_sigma_ab_) ), "allowed chrominance (AB channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)")
                ("hv_color_sigma_l", po::value<float>(&color_sigma_l_)->default_value(color_sigma_l_, boost::str(boost::format("%.2e") % color_sigma_l_) ), "allowed illumination (L channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)")
                ("hv_sigma_normals_deg", po::value<float>(&sigma_normals_deg_)->default_value(sigma_normals_deg_, boost::str(boost::format("%.2e") % sigma_normals_deg_) ), "variance for surface normals")
                ("hv_histogram_specification", po::value<bool>(&use_histogram_specification_)->default_value(use_histogram_specification_), " ")
                ("hv_ignore_color", po::value<bool>(&ignore_color_even_if_exists_)->default_value(ignore_color_even_if_exists_), " ")
                ("hv_initial_status", po::value<bool>(&initial_status_)->default_value(initial_status_), "sets the initial activation status of each hypothesis to this value before starting optimization. E.g. If true, all hypotheses will be active and the cost will be optimized from that initial status.")
//                    ("hv_color_space", po::value<int>(&color_space_)->default_value(color_space_), "specifies the color space being used for verification (0... LAB, 1... RGB, 2... Grayscale,  3,4,5,6... ?)")
                ("hv_color_comparison_method", po::value<int>(&color_comparison_method_)->default_value(color_comparison_method_), "method used for color comparison (0... CIE76, 1... CIE94, 2... CIEDE2000)")
                ("hv_color_stddev_mul", po::value<float>(&color_std_dev_multiplier_threshold_)->default_value(color_std_dev_multiplier_threshold_), "standard deviation multiplier threshold for the local color description for each color channel")
                ("hv_inlier_threshold", po::value<float>(&inliers_threshold_)->default_value(inliers_threshold_, boost::str(boost::format("%.2e") % inliers_threshold_) ), "Represents the maximum distance between model and scene points in order to state that a scene point is explained by a model point. Valid model points that do not have any corresponding scene point within this threshold are considered model outliers")
                ("hv_occlusion_threshold", po::value<float>(&occlusion_thres_)->default_value(occlusion_thres_, boost::str(boost::format("%.2e") % occlusion_thres_) ), "Threshold for a point to be considered occluded when model points are back-projected to the scene ( depends e.g. on sensor noise)")
                ("hv_optimizer_type", po::value<int>(&opt_type_)->default_value(opt_type_), "defines the optimization methdod. 0: Local search (converges quickly, but can easily get trapped in local minima), 1: Tabu Search, 4; Tabu Search + Local Search (Replace active hypotheses moves), else: Simulated Annealing")
                ("hv_radius_clutter", po::value<float>(&radius_neighborhood_clutter_)->default_value(radius_neighborhood_clutter_, boost::str(boost::format("%.2e") % radius_neighborhood_clutter_) ), "defines the maximum distance between two points to be checked for label consistency")
                ("hv_regularizer,r", po::value<float>(&regularizer_)->default_value(regularizer_, boost::str(boost::format("%.2e") % regularizer_) ), "represents a penalty multiplier for model outliers. In particular, each model outlier associated with an active hypothesis increases the global cost function.")
                ("hv_resolution_mm", po::value<int>(&resolution_mm_)->default_value(resolution_mm_), "The resolution of models and scene used to verify hypotheses (in milli meters)")
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

    friend class boost::serialization::access;

//    BOOST_CLASS_VERSION(HV_Parameter, 1)

    template<class Archive> V4R_EXPORTS void serialize(Archive & ar, const unsigned int version)
    {
        (void) version;
        ar & BOOST_SERIALIZATION_NVP(resolution_mm_)
                & BOOST_SERIALIZATION_NVP(inliers_threshold_)
                & BOOST_SERIALIZATION_NVP(inliers_surface_angle_thres_)
                & BOOST_SERIALIZATION_NVP(occlusion_thres_)
                & BOOST_SERIALIZATION_NVP(smoothing_radius_)
                & BOOST_SERIALIZATION_NVP(do_smoothing_)
                & BOOST_SERIALIZATION_NVP(do_erosion_)
                & BOOST_SERIALIZATION_NVP(erosion_radius_)
                & BOOST_SERIALIZATION_NVP(icp_iterations_)
                & BOOST_SERIALIZATION_NVP(w_normals_)
                & BOOST_SERIALIZATION_NVP(w_color_)
                & BOOST_SERIALIZATION_NVP(w_3D_)
                & BOOST_SERIALIZATION_NVP(color_sigma_l_)
                & BOOST_SERIALIZATION_NVP(color_sigma_ab_)
                & BOOST_SERIALIZATION_NVP(sigma_normals_deg_)
                & BOOST_SERIALIZATION_NVP(regularizer_)
                & BOOST_SERIALIZATION_NVP(radius_neighborhood_clutter_)
                & BOOST_SERIALIZATION_NVP(normal_method_)
                & BOOST_SERIALIZATION_NVP(ignore_color_even_if_exists_)
                & BOOST_SERIALIZATION_NVP(max_iterations_)
                & BOOST_SERIALIZATION_NVP(clutter_regularizer_)
                & BOOST_SERIALIZATION_NVP(use_replace_moves_)
                & BOOST_SERIALIZATION_NVP(opt_type_)
                & BOOST_SERIALIZATION_NVP(use_histogram_specification_)
                & BOOST_SERIALIZATION_NVP(initial_status_)
                //              color_space_
                & BOOST_SERIALIZATION_NVP(color_comparison_method_)
                & BOOST_SERIALIZATION_NVP(use_noise_model_)
                & BOOST_SERIALIZATION_NVP(knn_inliers_)
                & BOOST_SERIALIZATION_NVP( min_visible_ratio_)
                & BOOST_SERIALIZATION_NVP(min_model_fitness_lower_bound_)
                & BOOST_SERIALIZATION_NVP(min_model_fitness_upper_bound_)
                & BOOST_SERIALIZATION_NVP(knn_color_neighborhood_)
                & BOOST_SERIALIZATION_NVP(color_std_dev_multiplier_threshold_)
                & BOOST_SERIALIZATION_NVP(check_plane_intersection_)
                & BOOST_SERIALIZATION_NVP(plane_method_)
                & BOOST_SERIALIZATION_NVP(check_smooth_clusters_)
                & BOOST_SERIALIZATION_NVP(eps_angle_threshold_deg_)
                & BOOST_SERIALIZATION_NVP(curvature_threshold_)
                & BOOST_SERIALIZATION_NVP(cluster_tolerance_)
                & BOOST_SERIALIZATION_NVP(min_points_)
                & BOOST_SERIALIZATION_NVP(min_ratio_cluster_explained_)
                & BOOST_SERIALIZATION_NVP(z_adaptive_);
    }

    void
    save(const std::string &filename) const
    {
        std::ofstream ofs(filename);
        boost::archive::xml_oarchive oa(ofs);
        oa << BOOST_SERIALIZATION_NVP( *this );
        ofs.close();
    }

    HV_Parameter(const std::string &filename)
    {
        std::ifstream ifs(filename);
        boost::archive::xml_iarchive ia(ifs);
        ia >> BOOST_SERIALIZATION_NVP( *this );
        ifs.close();
    }
};

}
