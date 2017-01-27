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

#pragma once

#include <v4r/core/macros.h>
#include <v4r/common/camera.h>
#include <v4r/common/color_comparison.h>
#include <v4r/common/rgb2cielab.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/common/plane_model.h>
#include <v4r/recognition/ghv_opt.h>
#include <v4r/recognition/hypotheses_verification_param.h>
#include <v4r/recognition/object_hypothesis.h>

#include <glog/logging.h>
#include <metslib/mets.hh>
#include <opencv/cv.h>
#include <pcl/common/angles.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/octree/impl/octree_iterator.hpp>

#include <boost/bind.hpp>
#include <boost/format.hpp>
#include <boost/function.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

namespace v4r
{

template<typename ModelT, typename SceneT> class GHVmove_manager;   // forward declaration
template<typename ModelT, typename SceneT> class GHVSAModel;   // forward declaration
template<typename ModelT, typename SceneT> class GHVCostFunctionLogger;   // forward declaration


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
    typedef boost::shared_ptr< HypothesisVerification<ModelT, SceneT> > Ptr;
    typedef boost::shared_ptr< HypothesisVerification<ModelT, SceneT> const> ConstPtr;

public:
    HV_Parameter param_;

private:

    typedef boost::mpl::map
    <
    boost::mpl::pair<pcl::PointXYZ,          pcl::PointNormal>,
    boost::mpl::pair<pcl::PointNormal,       pcl::PointNormal>,
    boost::mpl::pair<pcl::PointXYZRGB,       pcl::PointXYZRGBNormal>,
    boost::mpl::pair<pcl::PointXYZRGBA,      pcl::PointXYZRGBNormal>,
    boost::mpl::pair<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>,
    boost::mpl::pair<pcl::PointXYZI,         pcl::PointXYZINormal>,
    boost::mpl::pair<pcl::PointXYZINormal,   pcl::PointXYZINormal>
    > PointTypeAssociations;
    BOOST_MPL_ASSERT ((boost::mpl::has_key<PointTypeAssociations, SceneT>));

    typedef typename boost::mpl::at<PointTypeAssociations, SceneT>::type SceneTWithNormal;

    mutable pcl::visualization::PCLVisualizer::Ptr vis_go_cues_, rm_vis_, vis_pairwise_;
    mutable std::vector<std::string> coordinate_axes_ids_;
    mutable int vp_active_hypotheses_, vp_scene_, vp_model_scene_3D_dist_, vp_model_scene_color_dist_, vp_scene_fitness_;

    mutable int rm_vp_scene_, rm_vp_model_, rm_vp_visible_model_,
    rm_vp_model_scene_3d_dist_, rm_vp_model_scene_color_dist_, rm_vp_model_scene_normals_dist_, rm_vp_model_scene_model_fit_,
    rm_vp_smooth_labels_, rm_vp_scene_fitness_, rm_vp_scene_and_model_, rm_v7, rm_v8, rm_v9, rm_v10, rm_v11, rm_v12, vp_pair_1_, vp_pair_2_, vp_pair_3_;

    Camera::ConstPtr cam_;
    PCLVisualizationParams::ConstPtr vis_param_;
    ColorTransform::Ptr colorTransf_;

    typename Source<ModelT>::ConstPtr m_db_;  ///< model data base

    boost::dynamic_bitset<> solution_; ///< Boolean vector indicating if a hypothesis is accepted (true) or rejected (false)
    boost::dynamic_bitset<> solution_tmp_;

    typename pcl::PointCloud<SceneT>::ConstPtr scene_cloud_; ///< scene point clou
    typename pcl::PointCloud<pcl::Normal>::ConstPtr scene_normals_; ///< scene normals cloud
    typename pcl::PointCloud<SceneT>::Ptr scene_cloud_downsampled_; ///< Downsampled scene point cloud
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals_downsampled_; ///< Downsampled scene normals cloud
    std::vector<int> scene_sampled_indices_;    ///< downsampled indices of the scene

    std::vector<std::vector<typename HVRecognitionModel<ModelT>::Ptr > > obj_hypotheses_groups_;
    std::vector<typename HVRecognitionModel<ModelT>::Ptr > global_hypotheses_;  ///< all hypotheses not rejected by individual verification

    std::vector<int> recognition_models_map_;
    std::map<std::string, boost::shared_ptr< pcl::octree::OctreePointCloudPointVector<ModelT> > > octree_model_representation_; ///< for each model we create an octree representation (used for computing visible points)
    typename pcl::search::KdTree<SceneT>::Ptr kdtree_scene_;
    std::vector<typename HVRecognitionModel<ModelT>::Ptr > recognition_models_; ///< all models to be verified (including planar models if included)

    double model_fitness_, pairwise_cost_, scene_fitness_;
    double model_fitness_tmp_, pairwise_cost_tmp_, scene_fitness_tmp_;

    Eigen::VectorXf model_fitness_v_;

    float Lmin_ = 0.f, Lmax_ = 100.f;
    int bins_ = 50;

    class PtFitness
    {
    public:
        float fit_;
        size_t rm_id_;

        bool operator<(const PtFitness &other) const { return this->fit_ < other.fit_; }

        PtFitness(float fit, size_t rm_id)
            : fit_(fit), rm_id_ (rm_id) {}
    };

    void applySolution( const boost::dynamic_bitset<> &new_solution )
    {
        updateTerms ( new_solution );
        solution_ = new_solution;

        model_fitness_ = model_fitness_tmp_;
        pairwise_cost_ = pairwise_cost_tmp_;
        scene_fitness_ = scene_fitness_tmp_;
    }

    Eigen::MatrixXf intersection_cost_; ///< represents the pairwise intersection cost
    std::vector<std::vector<PtFitness> > scene_pts_explained_vec_;

    GHVSAModel<ModelT, SceneT> best_seen_;
    float initial_temp_;
    boost::shared_ptr<GHVCostFunctionLogger<ModelT,SceneT> > cost_logger_;
    Eigen::MatrixXf scene_color_channels_; ///< converted color values where each point corresponds to a row entry
    typename pcl::octree::OctreePointCloudSearch<SceneT>::Ptr octree_scene_downsampled_;
    boost::function<void (const boost::dynamic_bitset<> &, float, int)> visualize_cues_during_logger_;

    std::vector<int> scene_smooth_labels_;  ///< stores a label for each point of the (downsampled) scene. Points belonging to the same smooth clusters, have the same label
    std::vector<size_t> smooth_label_count_;    ///< counts how many times a certain smooth label occurs in the scene

    // ----- MULTI-VIEW VARIABLES------
    std::vector<typename pcl::PointCloud<SceneT>::ConstPtr> occlusion_clouds_; ///< scene clouds from multiple views (stored as organized point clouds)
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > absolute_camera_poses_;
    std::vector<boost::dynamic_bitset<> > model_is_present_in_view_; ///< for each model this variable stores information in which view it is present (used to check for visible model points - default all true = static scene)

    boost::function<float (const Eigen::VectorXf&, const Eigen::VectorXf&)> color_dist_f_;

    Eigen::Matrix4f refinePose(HVRecognitionModel<ModelT> &rm) const;

    /**
     * @brief computeVisiblePoints first renders the model cloud in the given pose onto the image plane and checks via z-buffering
     * for each model point if it is visible or self-occluded. The visible model cloud is then compared to the scene cloud for occlusion
     * caused by the input scene
     * @param rm recongition model
     */
    void computeModelOcclusionByScene(HVRecognitionModel<ModelT> &rm) const; ///< computes the visible points of the model in the given pose and the provided depth map(s) of the scene


    /**
     * @brief computeVisibleOctreeNodes the visible point computation so far misses out points that are not visible just because of the discretization from the image back-projection
     * for each pixel only one point falling on that pixel can be said to be visible with the approach so far.
     * Particularly in high-resolution models it is however possible that multiple points project on the same pixel and they should be also visible.
     * if there is at least one visible point computed from the approach above. If so, the leaf node is counted visible and all its containing points
     * are also set as visible.
     * @param rm recognition model
     */
    void computeVisibleOctreeNodes(HVRecognitionModel<ModelT> &rm);

    void downsampleSceneCloud ();    ///< downsamples the scene cloud

    void removeSceneNans ();    ///< removes nan values from the downsampled scene cloud

    bool removeNanNormals (HVRecognitionModel<ModelT> & recog_model) const; ///< remove all points from visible cloud and normals which are not-a-number

    void removeModelsWithLowVisibility (); ///< remove recognition models where there are not enough visible points (>95% occluded)

    void computePairwiseIntersection (); ///< computes the overlap of two visible points when projected to camera view

    void computeModelFitness (HVRecognitionModel<ModelT> &rm) const;

    void initialize ();

    mets::gol_type evaluateSolution (const boost::dynamic_bitset<> &solution);

    void updateTerms(const boost::dynamic_bitset<> &solution);

    void optimize();

    void visualizeGOCues (const boost::dynamic_bitset<> &active_solution, float cost, int times_eval) const;

    void visualizePairwiseIntersection () const;

    void visualizeGOCuesForModel (const HVRecognitionModel<ModelT> &rm) const;

    bool individualRejection(HVRecognitionModel<ModelT> &rm) const;///< remove hypotheses badly explaining the scene, or only partially explaining smooth clusters (if check is enabled). Returns true if hypothesis is rejected.

    void cleanUp ()
    {
        octree_scene_downsampled_.reset();
        occlusion_clouds_.clear();
        absolute_camera_poses_.clear();
        scene_sampled_indices_.clear();
        model_is_present_in_view_.clear();
        scene_cloud_downsampled_.reset();
        scene_cloud_.reset();
        intersection_cost_.resize(0,0);
        scene_pts_explained_vec_.clear();
        obj_hypotheses_groups_.clear();
    }

    /**
     * @brief extractEuclideanClustersSmooth
     */
    void
    extractEuclideanClustersSmooth ();

    /**
     * @brief getFitness
     * @param c
     * @return
     */
    float
    getFitness( const ModelSceneCorrespondence& c ) const
    {
        return (param_.w_3D_ * modelScene3DDistCostTerm(c) + param_.w_color_ * modelSceneColorCostTerm(c) + param_.w_normals_ * modelSceneNormalsCostTerm(c)) / (param_.w_3D_ + param_.w_color_ + param_.w_normals_);
    }

    /**
     * @brief modelSceneColorCostTerm
     * @param model scene correspondence
     * @return
     */
    float
    modelSceneColorCostTerm( const ModelSceneCorrespondence& c ) const
    {
        //        return exp( - c.color_distance_ / param_.color_sigma_ab_ );
        //        return std::max(0.f, std::min(1.f, 1.f-c.color_distance_ / 50.f));
        return exp (- c.color_distance_/param_.color_sigma_ab_ );
    }

    /**
     * @brief modelScene3DDistCostTerm
     * @param model scene correspondence
     * @return distance in centimeter
     */
    float
    modelScene3DDistCostTerm( const ModelSceneCorrespondence& c ) const
    {
        if ( c.dist_3D_ < param_.inliers_threshold_ )
            return 1.f;
        else
            return exp( -(c.dist_3D_ - param_.inliers_threshold_) * (c.dist_3D_ - param_.inliers_threshold_) / (param_.inliers_threshold_ * param_.inliers_threshold_) );
    }

    /**
     * @brief modelSceneNormalsCostTerm
     * @param model scene correspondence
     * @return angle between corresponding surface normals (fliped such that they are pointing into the same direction)
     */
    float modelSceneNormalsCostTerm( const ModelSceneCorrespondence& c ) const
    {
        if ( c.angle_surface_normals_rad_ < param_.inliers_surface_angle_thres_)
            return 1.f;

        if ( c.angle_surface_normals_rad_ > M_PI/2)
            return 0.f;

        return 1 - (c.angle_surface_normals_rad_ - param_.inliers_surface_angle_thres_) / (M_PI/2 - param_.inliers_surface_angle_thres_);
    }

    void
    computeLOffset( HVRecognitionModel<ModelT> &rm ) const;

    float
    customColorDistance(const Eigen::VectorXf &color_a, const Eigen::VectorXf &color_b)
    {
        float L_dist  = ( color_a(0) - color_b(0) )*( color_a(0) - color_b(0) );
        CHECK(L_dist >= 0.f && L_dist <= 1.f);
        L_dist /= param_.color_sigma_l_ ;
        float AB_dist = ( color_a.tail(2) - color_b.tail(2) ).squaredNorm(); // ( param_.color_sigma_ab_ * param_.color_sigma_ab_ );
        CHECK(AB_dist >= 0.f && AB_dist <= 1.f);
        return L_dist + AB_dist ;
    }

    /**
     * @brief customRegionGrowing constraint function which decides if two points are to be merged as one "smooth" cluster
     * @param seed_pt
     * @param candidate_pt
     * @param squared_distance
     * @return
     */
    bool
    customRegionGrowing (const SceneTWithNormal& seed_pt, const SceneTWithNormal& candidate_pt, float squared_distance) const
    {
        float curvature_threshold = param_.curvature_threshold_ ;
        float radius = param_.cluster_tolerance_;
        float eps_angle_threshold_rad = pcl::deg2rad(param_.eps_angle_threshold_deg_);

        if ( param_.z_adaptive_ )
        {
            float mult = std::max(seed_pt.z, 1.f);
//            mult *= mult;
            radius = param_.cluster_tolerance_ * mult;
            curvature_threshold = param_.curvature_threshold_ * mult;
            eps_angle_threshold_rad = eps_angle_threshold_rad * mult;
        }

        if( seed_pt.curvature > param_.curvature_threshold_)
            return false;

        if(candidate_pt.curvature > param_.curvature_threshold_)
            return false;

        if (squared_distance > radius * radius)
            return false;

        float dotp = seed_pt.getNormalVector3fMap().dot (candidate_pt.getNormalVector3fMap() );
        if (fabs (dotp) < cos(eps_angle_threshold_rad))
            return false;

        float intensity_a = .2126 * seed_pt.r + .7152 * seed_pt.g + .0722 * seed_pt.b;
        float intensity_b = .2126 * candidate_pt.r + .7152 * candidate_pt.g + .0722 * candidate_pt.b;

        if( fabs(intensity_a - intensity_b) > 5.)
            return false;

        return true;
    }

public:

    HypothesisVerification (const Camera::ConstPtr &cam,
                            const HV_Parameter &p = HV_Parameter(),
                            const PCLVisualizationParams::ConstPtr &vis_params = boost::make_shared<PCLVisualizationParams>())
        : param_(p), cam_(cam), vis_param_(vis_params), initial_temp_(1000)
    {
        colorTransf_.reset(new RGB2CIELAB);

        switch (param_.color_comparison_method_)
        {
        case ColorComparisonMethod::cie76 :
            color_dist_f_ = CIE76; break;

        case ColorComparisonMethod::cie94 :
            color_dist_f_ = CIE76; break;

        case ColorComparisonMethod::ciede2000 :
            color_dist_f_ = CIEDE2000; break;

        case 3:
            color_dist_f_ = boost::bind( &HypothesisVerification::customColorDistance, this, _1, _2 ); break;

        default:
            throw std::runtime_error("Color comparison method not defined!");
        }
    }

    /**
     *  \brief: Returns a vector of booleans representing which hypotheses have been accepted/rejected (true/false)
     *  mask vector of booleans
     */
    boost::dynamic_bitset<>
    getSolution () const
    {
        return solution_;
    }

    /**
     * @brief setSolution
     * @param solution
     */
    void
    setSolution (const boost::dynamic_bitset<> &solution)
    {
        CHECK(solution.size() == global_hypotheses_.size());
        solution_ = solution;
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
    setSceneCloud (const typename pcl::PointCloud<SceneT>::ConstPtr & scene_cloud)
    {
        scene_cloud_ = scene_cloud;
    }

    /**
     *  \brief Sets the scene normals cloud
     *  \param normals Point cloud representing the scene normals
     */
    void
    setNormals (const pcl::PointCloud<pcl::Normal>::ConstPtr & normals)
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
        for (size_t i=0; i<occ_clouds.size(); i++)
            CHECK( occ_clouds[i]->isOrganized() ) << "Occlusion clouds need to be organized!";

        occlusion_clouds_ = occ_clouds;
        absolute_camera_poses_ = absolute_camera_poses;
    }


    /**
     * @brief for each model this variable stores information in which view it is present
     * @param presence in model and view
     */
    void
    setVisibleCloudsForModels(const std::vector<boost::dynamic_bitset<> > &model_is_present_in_view)
    {
        model_is_present_in_view_ = model_is_present_in_view;
    }

    /**
     * @brief setHypotheses
     * @param ohs
     */
    void
    setHypotheses(const std::vector<ObjectHypothesesGroup<ModelT> > &ohs);

    /**
     * @brief writeToLog
     * @param of
     * @param all_costs_
     */
    void
    writeToLog (std::ofstream & of, bool all_costs_ = false)
    {
        cost_logger_->writeToLog (of);
        if (all_costs_)
            cost_logger_->writeEachCostToLog (of);
    }

    /**
     * @brief setModelDatabase
     * @param m_db model database
     */
    void
    setModelDatabase(const typename Source<ModelT>::ConstPtr &m_db)
    {
        m_db_ = m_db;
    }



    /**
     *  \brief Function that performs the hypotheses verification
     *  This function modifies the values of mask_ and needs to be called after both scene and model have been added
     */
    void
    verify();
};

}
