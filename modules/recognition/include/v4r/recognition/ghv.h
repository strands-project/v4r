/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma
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
#include <pcl/common/common.h>
#include <pcl/pcl_macros.h>
#include "hypotheses_verification.h"
//#include <pcl/recognition/3rdparty/metslib/mets.hh>
#include <metslib/mets.hh>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <map>
#include <stack>
#include <iostream>
#include <fstream>
#include "ghv_opt.h"
#include <v4r/common/common_data_structures.h>

namespace v4r
{

/** \brief A hypothesis verification method proposed in
   * "A Global Hypotheses Verification Method for 3D Object Recognition", A. Aldoma and F. Tombari and L. Di Stefano and Markus Vincze, ECCV 2012,
   * Extended with physical constraints and color information (see ICRA paper)
   * \author Aitor Aldoma
   * \date Feb, 2013
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
        using HypothesisVerification<ModelT, SceneT>::Parameter::zbuffer_scene_resolution_;
        using HypothesisVerification<ModelT, SceneT>::Parameter::zbuffer_self_occlusion_resolution_;
        using HypothesisVerification<ModelT, SceneT>::Parameter::self_occlusions_reasoning_;
        using HypothesisVerification<ModelT, SceneT>::Parameter::focal_length_;
        using HypothesisVerification<ModelT, SceneT>::Parameter::do_occlusion_reasoning_;

        double color_sigma_l_; /// @brief allowed illumination (L channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)
        double color_sigma_ab_; /// @brief allowed chrominance (AB channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)
        double regularizer_; /// @brief represents a penalty multiplier for model outliers. In particular, each model outlier associated with an active hypothesis increases the global cost function.
        double radius_neighborhood_clutter_; /// @brief defines the maximum distance between an <i>explained</i> scene point <b>p</b> and other unexplained scene points such that they influence the clutter term associated with <b>p</b>
        int normal_method_; /// @brief method used for computing the normals of the downsampled scene point cloud (defined by the V4R Library)
        double duplicy_weight_test_;
        double duplicity_curvature_max_;
        bool ignore_color_even_if_exists_;
        int max_iterations_; /// @brief max iterations without improvement
        double clutter_regularizer_; /// @brief The penalty multiplier used to penalize unexplained scene points within the clutter influence radius <i>radius_neighborhood_clutter_</i> of an explained scene point when they belong to the same smooth segment.
        bool detect_clutter_;
        double res_occupancy_grid_;
        double w_occupied_multiple_cm_;
        bool use_super_voxels_;
        bool use_replace_moves_;
        int opt_type_; /// @brief defines the optimization methdod<BR><BR> 0: Local search (converges quickly, but can easily get trapped in local minima),<BR> 1: Tabu Search,<BR> 2; Tabu Search + Local Search (Replace active hypotheses moves),<BR> 3: Simulated Annealing
        double active_hyp_penalty_;
        int multiple_assignment_penalize_by_one_;
        double d_weight_for_bad_normals_; /// @brief specifies the weight an outlier is multiplied with in case the corresponding scene point's orientation is facing away from the camera more than a certain threshold and potentially has inherent noise
        bool use_clutter_exp_;
        bool use_histogram_specification_;
        bool use_points_on_plane_side_;  /// @brief //compute for each planar model how many points for the other hypotheses (if in conflict) are on each side of the plane
        double best_color_weight_;
        bool initial_status_; /// @brief sets the initial activation status of each hypothesis to this value before starting optimization. E.g. If true, all hypotheses will be active and the cost will be optimized from that initial status.
        int color_space_; /// @brief specifies the color space being used for verification (0... LAB, 1... RGB, 2... Grayscale,  3,4,5,6... ???)
        int outliers_weight_computation_method_; /// @brief defines the method used for computing the overall outlier weight. 0... mean, 1... median

        //smooth segmentation parameters
        double eps_angle_threshold_;
        int min_points_;
        double curvature_threshold_;
        double cluster_tolerance_;

        bool use_normals_from_visible_;

        bool add_planes_;  /// @brief if true, adds planes as possible hypotheses (slower but decreases false positives especially for planes detected as flat objects like books)
        int plane_method_; /// @brief defines which method to use for plane extraction (if add_planes_ is true). 0... Multiplane Segmentation, 1... ClusterNormalsForPlane segmentation
        size_t min_plane_inliers_; /// @brief a planar cluster is only added as plane if it has at least min_plane_inliers_ points
        double plane_inlier_distance_; /// @brief Maximum inlier distance for plane clustering
        double plane_thrAngle_;  /// @brief Threshold of normal angle in degree for plane clustering
        int knn_plane_clustering_search_;  /// @brief sets the number of points used for searching nearest neighbors in unorganized point clouds (used in plane segmentation)
        bool visualize_go_cues_; /// @brief visualizes the cues during the computation and shows cost and number of evaluations. Useful for debugging

        Parameter (
                double color_sigma_l = 0.6f,
                double color_sigma_ab = 0.6f,
                double regularizer = 1.f, // 3
                double radius_neighborhood_clutter = 0.02f,
                int normal_method = 2,
                double duplicy_weight_test = 1.f,
                double duplicity_curvature_max = 0.03f,
                bool ignore_color_even_if_exists = false,
                int max_iterations = 5000,
                double clutter_regularizer =  1.f, //3.f,
                bool detect_clutter = true,
                double res_occupancy_grid = 0.005f,
                double w_occupied_multiple_cm = 2.f, //0.f
                bool use_super_voxels = false,
                bool use_replace_moves = true,
                int opt_type = OptimizationType::LocalSearch,
                double active_hyp_penalty = 0.f, // 0.05f
                int multiple_assignment_penalize_by_one = 2,
                double d_weight_for_bad_normals = 0.1f,
                bool use_clutter_exp = false,
                bool use_histogram_specification = false, // false
                bool use_points_on_plane_side = true,
                double best_color_weight = 0.8f,
                bool initial_status = false,
                int color_space = ColorSpace::LAB,
                int outliers_weight_computation_method = 0,
                double eps_angle_threshold = 0.25, //0.1f
                int min_points = 100, // 20
                double curvature_threshold = 0.04f,
                double cluster_tolerance = 0.01f, //0.015f;
                bool use_normals_from_visible = false,
                bool add_planes = true,
                int plane_method = 1,
                size_t min_plane_inliers = 5000,
                double plane_inlier_distance = 0.02f,
                double plane_thrAngle = 30,
                int knn_plane_clustering_search = 10,
                bool visualize_go_cues = false
                )
            :
              HypothesisVerification<ModelT, SceneT>::Parameter(),
              color_sigma_l_ (color_sigma_l),
              color_sigma_ab_ (color_sigma_ab),
              regularizer_ (regularizer),
              radius_neighborhood_clutter_ (radius_neighborhood_clutter),
              normal_method_ (normal_method),
              duplicy_weight_test_ (duplicy_weight_test),
              duplicity_curvature_max_ (duplicity_curvature_max),
              ignore_color_even_if_exists_ (ignore_color_even_if_exists),
              max_iterations_ (max_iterations),
              clutter_regularizer_ (clutter_regularizer),
              detect_clutter_ (detect_clutter),
              res_occupancy_grid_ (res_occupancy_grid),
              w_occupied_multiple_cm_ (w_occupied_multiple_cm),
              use_super_voxels_ (use_super_voxels),
              use_replace_moves_ (use_replace_moves),
              opt_type_ (opt_type),
              active_hyp_penalty_ (active_hyp_penalty),
              multiple_assignment_penalize_by_one_ (multiple_assignment_penalize_by_one),
              d_weight_for_bad_normals_ (d_weight_for_bad_normals),
              use_clutter_exp_ (use_clutter_exp),
              use_histogram_specification_ (use_histogram_specification),
              use_points_on_plane_side_ (use_points_on_plane_side),
              best_color_weight_ (best_color_weight),
              initial_status_ (initial_status),
              color_space_ (color_space),
              outliers_weight_computation_method_ (outliers_weight_computation_method),
              eps_angle_threshold_ (eps_angle_threshold),
              min_points_ (min_points),
              curvature_threshold_ (curvature_threshold),
              cluster_tolerance_ (cluster_tolerance),
              use_normals_from_visible_ (use_normals_from_visible),
              add_planes_ (add_planes),
              plane_method_ (plane_method),
              min_plane_inliers_ ( min_plane_inliers ),
              plane_inlier_distance_ ( plane_inlier_distance ),
              plane_thrAngle_ ( plane_thrAngle ),
              knn_plane_clustering_search_ ( knn_plane_clustering_search ),
              visualize_go_cues_ ( visualize_go_cues )
        {}
    }param_;

private:
    mutable int viewport_scene_and_hypotheses_, viewport_model_cues_, viewport_smooth_seg_, viewport_scene_cues_;


protected:
    using HypothesisVerification<ModelT, SceneT>::mask_;
    using HypothesisVerification<ModelT, SceneT>::recognition_models_;
    using HypothesisVerification<ModelT, SceneT>::scene_cloud_downsampled_;
    using HypothesisVerification<ModelT, SceneT>::scene_downsampled_tree_;
    using HypothesisVerification<ModelT, SceneT>::model_point_is_visible_;
    using HypothesisVerification<ModelT, SceneT>::normals_set_;
    using HypothesisVerification<ModelT, SceneT>::requires_normals_;
    using HypothesisVerification<ModelT, SceneT>::occlusion_cloud_;
    using HypothesisVerification<ModelT, SceneT>::scene_cloud_;
    using HypothesisVerification<ModelT, SceneT>::scene_sampled_indices_;
    using HypothesisVerification<ModelT, SceneT>::recognition_models_map_;

    template<typename PointT, typename NormalT>
    inline void
    extractEuclideanClustersSmooth (const typename pcl::PointCloud<PointT> &cloud, const typename pcl::PointCloud<NormalT> &normals, float tolerance,
                                    const typename pcl::search::Search<PointT>::Ptr &tree, std::vector<pcl::PointIndices> &clusters, double eps_angle,
                                    float curvature_threshold, size_t min_pts_per_cluster,
                                    unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ())
    {

        if (tree->getInputCloud ()->points.size () != cloud.points.size ())
        {
            PCL_ERROR("[pcl::extractEuclideanClusters] Tree built for a different point cloud dataset\n");
            return;
        }
        if (cloud.points.size () != normals.points.size ())
        {
            PCL_ERROR("[pcl::extractEuclideanClusters] Number of points in the input point cloud different than normals!\n");
            return;
        }


        std::vector<bool> processed (cloud.points.size (), false); // Create a boolean vector of processed point indices, and initialize it to false
        std::vector<int> nn_indices;
        std::vector<float> nn_distances;

        for (size_t i = 0; i < cloud.points.size(); i++) // Process all points in the indices vector
        {
            if (processed[i])
                continue;

            std::vector<size_t> seed_queue;
            size_t sq_idx = 0;
            seed_queue.push_back (i);

            processed[i] = true;

            while (sq_idx < seed_queue.size ())
            {

                if (normals.points[seed_queue[sq_idx]].curvature > curvature_threshold)
                {
                    sq_idx++;
                    continue;
                }

                // Search for sq_idx
                if (!tree->radiusSearch (seed_queue[sq_idx], tolerance, nn_indices, nn_distances))
                {
                    sq_idx++;
                    continue;
                }

                for (size_t j = 1; j < nn_indices.size (); ++j) // nn_indices[0] should be sq_idx
                {
                    if ( processed[nn_indices[j]] || normals.points[nn_indices[j]].curvature > curvature_threshold) // Has this point been processed before?
                        continue;


                    double dot_p = normals.points[seed_queue[sq_idx]].normal[0] * normals.points[nn_indices[j]].normal[0]
                            + normals.points[seed_queue[sq_idx]].normal[1] * normals.points[nn_indices[j]].normal[1] + normals.points[seed_queue[sq_idx]].normal[2]
                            * normals.points[nn_indices[j]].normal[2];

                    if (fabs (acos (dot_p)) < eps_angle)
                    {
                        processed[nn_indices[j]] = true;
                        seed_queue.push_back ( static_cast<size_t>(nn_indices[j]) );
                    }
                }

                sq_idx++;
            }

            // If this queue is satisfactory, add to the clusters
            if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)
            {
                pcl::PointIndices r;
                r.indices.resize (seed_queue.size ());
                for (size_t j = 0; j < seed_queue.size (); ++j)
                    r.indices[j] = seed_queue[j];

                std::sort (r.indices.begin (), r.indices.end ());
                r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());
                clusters.push_back (r); // We could avoid a copy by working directly in the vector
            }
        }
    }

    void
    computeClutterCueAtOnce ();

    bool
    removeNanNormals (HVRecognitionModel<ModelT> & recog_model);

    bool
    addModel (HVRecognitionModel<ModelT> &rm);

    void
    computeSceneOccupancyGridByRM();

    //Performs smooth segmentation of the scene cloud and compute the model cues
    bool
    initialize ();

    pcl::PointCloud<pcl::Normal>::Ptr scene_normals_;
    bool scene_and_normals_set_from_outside_;

    //class attributes
    pcl::PointCloud<pcl::PointXYZL>::Ptr clusters_cloud_;
    int max_label_clusters_cloud_;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr clusters_cloud_rgb_;
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals_for_clutter_term_;

    std::vector<int> complete_cloud_occupancy_by_RM_;

    std::vector<double> duplicates_by_RM_weighted_;
    std::vector<double> duplicates_by_RM_weighted_not_capped_;
    std::vector<int> explained_by_RM_; //represents the points of scene_cloud_ that are explained by the recognition models
    std::vector<double> explained_by_RM_distance_weighted_; //represents the points of scene_cloud_ that are explained by the recognition models
    std::vector<int> explained_by_RM_model_; //id of the model explaining the point
    std::vector< std::stack<std::pair<int, float>, std::vector<std::pair<int, float> > > > previous_explained_by_RM_distance_weighted_; //represents the points of scene_cloud_ that are explained by the recognition models
    std::vector<double> unexplained_by_RM_neighboorhods_; //represents the points of scene_cloud_ that are not explained by the active hypotheses in the neighboorhod of the recognition models
    //std::vector<size_t> indices_;

    double previous_explained_value_;
    double previous_duplicity_;
    int previous_duplicity_complete_models_;
    double previous_bad_info_;
    double previous_unexplained_;

    GHVSAModel<ModelT, SceneT> best_seen_;
    float initial_temp_;

    int min_contribution_;

    std::vector<std::vector<size_t> > rm_ids_explaining_scene_pt_; //if inner size > 1, conflict

    //mahalanobis stuff
    Eigen::MatrixXf inv_covariance_;
    Eigen::VectorXf mean_;

    ColorTransformOMP color_transf_omp_;


    double
    getExplainedByIndices (const std::vector<int> & indices,
                           const std::vector<float> & explained_values,
                           const std::vector<double> & explained_by_RM,
                           std::vector<int> & indices_to_update_in_RM_local) const;


    void
    updateUnexplainedVector (const std::vector<int> & unexplained, const std::vector<float> & unexplained_distances,
                             const std::vector<int> & explained, int sign)
    {
        double add_to_unexplained = 0.f;

        for (size_t i = 0; i < unexplained.size (); i++)
        {
            bool prev_unexplained = (unexplained_by_RM_neighboorhods_[unexplained[i]] > 0) && (explained_by_RM_[unexplained[i]] == 0);
            unexplained_by_RM_neighboorhods_[unexplained[i]] += (double) (sign * unexplained_distances[i]);

            if (sign < 0) //the hypothesis is being removed
            {
                if (prev_unexplained)
                {
                    //decrease by 1
                    add_to_unexplained -= unexplained_distances[i];
                }
            }
            else //the hypothesis is being added and unexplains unexplained_[i], so increase by 1 unless its explained by another hypothesis
            {
                if (explained_by_RM_[unexplained[i]] == 0)
                    add_to_unexplained += unexplained_distances[i];
            }
        }

        for (size_t i = 0; i < explained.size (); i++)
        {
            if (sign < 0) //the hypothesis is being removed, check that there are no points that become unexplained and have clutter unexplained hypotheses
            {
                if ((explained_by_RM_[explained[i]] == 0) && (unexplained_by_RM_neighboorhods_[explained[i]] > 0))
                    add_to_unexplained += unexplained_by_RM_neighboorhods_[explained[i]]; //the points become unexplained
            }
            else
            {
                if ((explained_by_RM_[explained[i]] == 1) && (unexplained_by_RM_neighboorhods_[explained[i]] > 0))
                { //the only hypothesis explaining that point
                    add_to_unexplained -= unexplained_by_RM_neighboorhods_[explained[i]]; //the points are not unexplained any longer because this hypothesis explains them
                }
            }
        }
        previous_unexplained_ += add_to_unexplained;
    }

    void
    updateExplainedVector (const std::vector<int> & vec, const std::vector<float> & vec_float, int sign, int model_id);

    void
    updateCMDuplicity (const std::vector<int> & vec, int sign);

    double
    getTotalExplainedInformation (const std::vector<int> & explained_, const std::vector<double> & explained_by_RM_distance_weighted_, double &duplicity_);

    double
    getTotalBadInformation (const std::vector<boost::shared_ptr<HVRecognitionModel<ModelT> > > & recog_models)
    {
        double bad_info = 0;
        for (size_t i = 0; i < recog_models.size(); i++)
            bad_info += recog_models[i]->outliers_total_weight_ * static_cast<double> (recog_models[i]->bad_information_);

        return bad_info;
    }

    double
    getUnexplainedInformationInNeighborhood (std::vector<double> & unexplained, std::vector<int> & explained)
    {
        double unexplained_sum = 0.f;
        for (size_t i = 0; i < unexplained.size (); i++)
        {
            if (unexplained[i] > 0 && explained[i] == 0)
                unexplained_sum += unexplained[i];
        }

        return unexplained_sum;
    }

    mets::gol_type
    evaluateSolution (const std::vector<bool> & active, int changed);

    std::vector<bool> optimize();

    void
    fill_structures (const std::vector<bool> &sub_solution, GHVSAModel<ModelT, SceneT> & model);

    void
    clear_structures ();

    double
    countActiveHypotheses (const std::vector<bool> & sol);

    double
    countPointsOnDifferentPlaneSides (const std::vector<bool> & sol);

    boost::shared_ptr<GHVCostFunctionLogger<ModelT,SceneT> > cost_logger_;

    void
    specifyHistograms(const std::vector<size_t> &src_hist, const std::vector<size_t> &dst_hist, std::vector<size_t> &lut);

    typename boost::shared_ptr<pcl::octree::OctreePointCloudSearch<SceneT> > octree_scene_downsampled_;

    Eigen::MatrixXf points_on_plane_sides_;

    boost::function<void (const std::vector<bool> &, float, int)> visualize_cues_during_logger_;

    void visualizeGOCues(const std::vector<bool> & active_solution, float cost, int times_eval);

    void visualizeGOCuesForModel(const HVRecognitionModel<ModelT> &rm) const;

    mutable pcl::visualization::PCLVisualizer::Ptr vis_go_cues_;

    mutable boost::shared_ptr<pcl::visualization::PCLVisualizer> rm_vis_;
    mutable int rm_v1, rm_v2, rm_v3, rm_v4, rm_v5, rm_v6;

    std::vector<pcl::PointCloud<pcl::PointXYZL>::Ptr> models_smooth_faces_;

    void
    registerModelAndSceneColor(std::vector<size_t> &lookup, HVRecognitionModel<ModelT> & recog_model);

    Eigen::MatrixXf scene_color_channels_;
    bool visualize_accepted_;
    typedef pcl::PointCloud<ModelT> CloudM;
    typedef pcl::PointCloud<SceneT> CloudS;
    typedef typename pcl::traits::fieldList<typename CloudS::PointType>::type FieldListS;
    typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

    double getCurvWeight(double p_curvature) const;

    std::vector<std::string> ply_paths_;
    std::vector<vtkSmartPointer <vtkTransform> > poses_ply_;
    size_t number_of_visible_points_;


    //compute mahalanobis distance
    static float
    mahalanobis(const Eigen::VectorXf & mu, const Eigen::VectorXf & x, const Eigen::MatrixXf & inv_cov)
    {
        float product = (x - mu).transpose() * inv_cov * (x - mu);
        return sqrt(product);
    }

    void segmentScene();
    void convertColor();

public:

    GHV (const Parameter &p=Parameter()) : HypothesisVerification<ModelT, SceneT> (p) , param_(p)
    {
        initial_temp_ = 1000;
        requires_normals_ = false;
        visualize_accepted_ = false;
        scene_and_normals_set_from_outside_ = false;
        min_contribution_ = 0;
    }

    enum ColorSpace
    {
       LAB,
       RGB,
       GRAYSCALE
    };

    enum OutlierType
    {
       DIST,
       COLOR
    };

    enum OptimizationType
    {
        LocalSearch,
        TabuSearch,
        TabuSearchWithLSRM,
        SimulatedAnnealing
    };

    void setMeanAndCovariance(const Eigen::VectorXf & mean, const Eigen::MatrixXf & cov)
    {
        mean_ = mean;
        inv_covariance_ = cov;
    }

    void setSceneAndNormals(typename pcl::PointCloud<SceneT>::Ptr & scene,
                            typename pcl::PointCloud<pcl::Normal>::Ptr & scene_normals)
    {
        scene_cloud_downsampled_ = scene;
        scene_normals_ = scene_normals;
        scene_and_normals_set_from_outside_ = true;
    }

    void setPlyPathsAndPoses(std::vector<std::string> & ply_paths_for_go, std::vector<vtkSmartPointer <vtkTransform> > & poses_ply)
    {
        ply_paths_ = ply_paths_for_go;
        poses_ply_ = poses_ply;
    }

    void setVisualizeAccepted(bool b)
    {
        visualize_accepted_ = b;
    }

    void setSmoothFaces(std::vector<pcl::PointCloud<pcl::PointXYZL>::Ptr> & aligned_smooth_faces)
    {
        models_smooth_faces_ = aligned_smooth_faces;
    }

    void setNormalsForClutterTerm(pcl::PointCloud<pcl::Normal>::Ptr & normals)
    {
        scene_normals_for_clutter_term_ = normals;
    }

    void
    addPlanarModels(const std::vector<PlaneModel<ModelT> > &planar_models);

    void
    writeToLog (std::ofstream & of, bool all_costs_ = false)
    {
        cost_logger_->writeToLog (of);
        if (all_costs_)
            cost_logger_->writeEachCostToLog (of);
    }

    /*void logCosts() {
       cost_logger_.reset(new CostFunctionLogger());
       }*/


    void
    setRequiresNormals (bool b)
    {
        requires_normals_ = b;
    }


    void
    verify();

    void
    setInitialTemp (float t)
    {
        initial_temp_ = t;
    }

    //Same length as the recognition models
    void
    setExtraWeightVectorForInliers (std::vector<float> & weights)
    {
        if(recognition_models_.size() != weights.size())
            throw std::runtime_error ("Size of extra weights is not equal the number of recognition models!");

        for (size_t i=0; i<weights.size(); i++)
            recognition_models_[i]->extra_weight_ = weights[i];
    }

    void
    getOutliersForAcceptedModels(std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud) const;

    void
    getOutliersForAcceptedModels(std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud_color,
                                 std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud_3d) const;

};
}

#endif
