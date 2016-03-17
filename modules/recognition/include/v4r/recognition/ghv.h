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

#ifndef V4R_GHV_H_
#define V4R_GHV_H_

#include <v4r/common/color_transforms.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/recognition/ghv_opt.h>
#include <pcl/common/common.h>
#include <pcl/octree/octree.h>
#include <metslib/mets.hh>
#include <pcl/visualization/cloud_viewer.h>

namespace v4r
{

/** \brief A hypothesis verification method for 3D Object Instance Recognition
   * \author Thomas Faeulhammer
   * \date April, 2016
   */
template<typename ModelT, typename SceneT>
class V4R_EXPORTS GHV : public HypothesisVerification<ModelT, SceneT>
{
    friend class GHVmove_manager<ModelT, SceneT>;
    friend class GHVSAModel<ModelT, SceneT>;

    //////////////////////////////////////////////////////////////////////////////////////////////
public:
    class V4R_EXPORTS Parameter : public HypothesisVerification<ModelT, SceneT>::Parameter
    {
    public:
        using HypothesisVerification<ModelT, SceneT>::Parameter::inliers_threshold_;
        using HypothesisVerification<ModelT, SceneT>::Parameter::resolution_;
        using HypothesisVerification<ModelT, SceneT>::Parameter::occlusion_thres_;
        using HypothesisVerification<ModelT, SceneT>::Parameter::zbuffer_self_occlusion_resolution_;
        using HypothesisVerification<ModelT, SceneT>::Parameter::focal_length_;
        using HypothesisVerification<ModelT, SceneT>::Parameter::do_occlusion_reasoning_;

        double color_sigma_l_; /// @brief allowed illumination (L channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)
        double color_sigma_ab_; /// @brief allowed chrominance (AB channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)
        double regularizer_; /// @brief represents a penalty multiplier for model outliers. In particular, each model outlier associated with an active hypothesis increases the global cost function.
        double radius_neighborhood_clutter_; /// @brief defines the maximum distance between an <i>explained</i> scene point <b>p</b> and other unexplained scene points such that they influence the clutter term associated with <b>p</b>
        int normal_method_; /// @brief method used for computing the normals of the downsampled scene point cloud (defined by the V4R Library)
        bool ignore_color_even_if_exists_;
        int max_iterations_; /// @brief max iterations without improvement
        double clutter_regularizer_; /// @brief The penalty multiplier used to penalize unexplained scene points within the clutter influence radius <i>radius_neighborhood_clutter_</i> of an explained scene point when they belong to the same smooth segment.
        bool use_replace_moves_;
        int opt_type_; /// @brief defines the optimization methdod<BR><BR> 0: Local search (converges quickly, but can easily get trapped in local minima),<BR> 1: Tabu Search,<BR> 2; Tabu Search + Local Search (Replace active hypotheses moves),<BR> 3: Simulated Annealing
        bool use_histogram_specification_;
        bool initial_status_; /// @brief sets the initial activation status of each hypothesis to this value before starting optimization. E.g. If true, all hypotheses will be active and the cost will be optimized from that initial status.
        int color_space_; /// @brief specifies the color space being used for verification (0... LAB, 1... RGB, 2... Grayscale,  3,4,5,6... ???)
        bool use_noise_model_;  /// @brief if set, uses Nguyens noise model for setting threshold parameters
        bool visualize_go_cues_; /// @brief visualizes the cues during the computation and shows cost and number of evaluations. Useful for debugging
        int knn_inliers_; /// @brief number of nearby scene points to check for a query model point

        double min_visible_ratio_; /// @brief defines how much of the object has to be visible in order to be included in the verification stage
        int knn_color_neighborhood_; /// @brief number of nearest neighbors used for describing the color around a point
        float color_std_dev_multiplier_threshold_; /// @brief standard deviation multiplier threshold for the local color description for each color channel

        Parameter (
                double color_sigma_l = 0.6f,
                double color_sigma_ab = 0.1f,
                double regularizer = 1.f, // 3
                double radius_neighborhood_clutter = 0.02f,
                int normal_method = 2,
                bool ignore_color_even_if_exists = false,
                int max_iterations = 5000,
                double clutter_regularizer =  1.f, //3.f,
                bool use_replace_moves = true, // true!!!,
                int opt_type = OptimizationType::LocalSearch,
                bool use_histogram_specification = false, // false
                bool initial_status = false,
                int color_space = ColorSpace::LAB,
                bool use_noise_model = true,
                bool visualize_go_cues = false,
                int knn_inliers = 3,
                double min_visible_ratio = 0.10f,
                int knn_color_neighborhood = 10,
                float color_std_dev_multiplier_threshold = 1.f
                )
            :
              HypothesisVerification<ModelT, SceneT>::Parameter(),
              color_sigma_l_ (color_sigma_l),
              color_sigma_ab_ (color_sigma_ab),
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
              knn_color_neighborhood_ (knn_color_neighborhood),
              color_std_dev_multiplier_threshold_ (color_std_dev_multiplier_threshold)
        {}
    }param_;

protected:
    using HypothesisVerification<ModelT, SceneT>::solution_;
    using HypothesisVerification<ModelT, SceneT>::recognition_models_;
    using HypothesisVerification<ModelT, SceneT>::recognition_models_map_;
    using HypothesisVerification<ModelT, SceneT>::scene_cloud_downsampled_;
    using HypothesisVerification<ModelT, SceneT>::normals_set_;
    using HypothesisVerification<ModelT, SceneT>::requires_normals_;
    using HypothesisVerification<ModelT, SceneT>::scene_cloud_;
    using HypothesisVerification<ModelT, SceneT>::scene_sampled_indices_;
    using HypothesisVerification<ModelT, SceneT>::cleanUp;
    using HypothesisVerification<ModelT, SceneT>::computeVisibleModelsAndRefinePose;

    mutable pcl::visualization::PCLVisualizer::Ptr vis_go_cues_;
    mutable boost::shared_ptr<pcl::visualization::PCLVisualizer> rm_vis_;
    mutable int vp_active_hypotheses_, vp_scene_, vp_model_fitness_, vp_scene_fitness_;
    mutable int rm_v1, rm_v2, rm_v3, rm_v4, rm_v5, rm_v6;

    double model_fitness_, pairwise_cost_, scene_fitness_, cost_;
    Eigen::VectorXf model_fitness_v_;
    Eigen::VectorXf tmp_solution_;

    pcl::PointCloud<pcl::Normal>::Ptr scene_normals_;
    bool scene_and_normals_set_from_outside_;
    Eigen::MatrixXf intersection_cost_; /// @brief represents the pairwise intersection cost
    Eigen::MatrixXf scene_explained_weight_; /// @brief for each point in the scene (row) store how good it is presented from each model (column)
    Eigen::MatrixXf scene_explained_weight_compressed_; /// @brief for each point in the scene (row) store how good it is presented from each model (column)
    Eigen::MatrixXf scene_explained_weight_duplicates_; //
    Eigen::VectorXf max_scene_explained_weight_; /// @brief for each point in the scene (row) store how good it is presented from each model (column)

    GHVSAModel<ModelT, SceneT> best_seen_;
    float initial_temp_;
    ColorTransformOMP color_transf_omp_;
    boost::shared_ptr<GHVCostFunctionLogger<ModelT,SceneT> > cost_logger_;
    Eigen::MatrixXf scene_color_channels_;
    typename boost::shared_ptr<pcl::octree::OctreePointCloudSearch<SceneT> > octree_scene_downsampled_;
    boost::function<void (const std::vector<bool> &, float, int)> visualize_cues_during_logger_;

    bool removeNanNormals (HVRecognitionModel<ModelT> & recog_model); /// @brief remove all points from visible cloud and normals which are not-a-number

    void convertSceneColor(); /// @brief converting scene points from RGB to desired color space

    void convertModelColor (HVRecognitionModel<ModelT> &rm); /// @brief converting visible points from the model from RGB to desired color space

    void computeModel2SceneFitness(HVRecognitionModel<ModelT> &rm, size_t model_idx); /// @brief checks for each visible point in the model if there are nearby scene points and how good they match

    void removeModelsWithLowVisibility(); /// @brief remove recognition models where there are not enough visible points (>95% occluded)

    void computePairwiseIntersection(); /// @brief computes the overlap of two visible points when projected to camera view

    void initialize();

    mets::gol_type evaluateSolution (const std::vector<bool> & active, int changed);

    std::vector<bool> optimize();

    void visualizeGOCues(const std::vector<bool> & active_solution, float cost_, int times_eval);

    void visualizeGOCuesForModel(const HVRecognitionModel<ModelT> &rm) const;

    void registerModelAndSceneColor(std::vector<size_t> &lookup, HVRecognitionModel<ModelT> & recog_model);

    typedef pcl::PointCloud<ModelT> CloudM;
    typedef pcl::PointCloud<SceneT> CloudS;
    typedef typename pcl::traits::fieldList<typename CloudS::PointType>::type FieldListS;
    typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

public:
    GHV (const Parameter &p=Parameter()) : HypothesisVerification<ModelT, SceneT> (p) , param_(p)
    {
        initial_temp_ = 1000;
        requires_normals_ = false;
        scene_and_normals_set_from_outside_ = false;
    }

    enum ColorSpace
    {
       LAB,
       RGB,
       GRAYSCALE
    };

    enum OptimizationType
    {
        LocalSearch,
        TabuSearch,
        TabuSearchWithLSRM,
        SimulatedAnnealing
    };

    void
    setSceneAndNormals(typename pcl::PointCloud<SceneT>::Ptr & scene,
                            typename pcl::PointCloud<pcl::Normal>::Ptr & scene_normals)
    {
        scene_cloud_downsampled_ = scene;
        scene_normals_ = scene_normals;
        scene_and_normals_set_from_outside_ = true;
    }

    void
    writeToLog (std::ofstream & of, bool all_costs_ = false)
    {
        cost_logger_->writeToLog (of);
        if (all_costs_)
            cost_logger_->writeEachCostToLog (of);
    }

    void
    setRequiresNormals (bool b)
    {
        requires_normals_ = b;
    }

    void
    verify();
};
}

#endif
