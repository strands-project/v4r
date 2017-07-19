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
#include <v4r/recognition/ghv_opt.h>
#include <v4r/recognition/hypotheses_verification_param.h>
#include <v4r/recognition/hypotheses_verification_visualization.h>
#include <v4r/recognition/object_hypothesis.h>

#include <glog/logging.h>
#include <metslib/mets.hh>
#include <pcl/common/angles.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree.h>
#include <pcl/search/kdtree.h>

#include <pcl/common/time.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/octree/impl/octree_iterator.hpp>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>

namespace v4r
{

// forward declarations
template<typename ModelT, typename SceneT> class GHVmove_manager;
template<typename ModelT, typename SceneT> class GHVSAModel;
template<typename ModelT, typename SceneT> class GHVCostFunctionLogger;
template<typename ModelT, typename SceneT> class HV_ModelVisualizer;
template<typename ModelT, typename SceneT> class HV_CuesVisualizer;
template<typename ModelT, typename SceneT> class HV_PairwiseVisualizer;

class PtFitness
{
public:
    float fit_;
    size_t rm_id_;

    bool operator<(const PtFitness &other) const { return this->fit_ < other.fit_; }

    PtFitness(float fit, size_t rm_id)
        : fit_(fit), rm_id_ (rm_id) {}
};


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
    friend class GHVCostFunctionLogger<ModelT, SceneT>;
    friend class HV_ModelVisualizer<ModelT, SceneT>;
    friend class HV_CuesVisualizer<ModelT, SceneT>;
    friend class HV_PairwiseVisualizer<ModelT, SceneT>;

public:
    typedef boost::shared_ptr< HypothesisVerification<ModelT, SceneT> > Ptr;
    typedef boost::shared_ptr< HypothesisVerification<ModelT, SceneT> const> ConstPtr;

    HV_Parameter param_;

protected:

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

    mutable typename HV_ModelVisualizer<ModelT, SceneT>::Ptr vis_model_;
    mutable typename HV_CuesVisualizer<ModelT, SceneT>::Ptr vis_cues_;
    mutable typename HV_PairwiseVisualizer<ModelT, SceneT>::Ptr vis_pairwise_;

    Camera::ConstPtr cam_;
    ColorTransform::Ptr colorTransf_;

    bool visualize_pairwise_cues_; ///< visualizes the pairwise cues. Useful for debugging

    typename Source<ModelT>::ConstPtr m_db_;  ///< model data base

    boost::dynamic_bitset<> solution_; ///< Boolean vector indicating if a hypothesis is accepted (true) or rejected (false)

    typename pcl::PointCloud<SceneT>::ConstPtr scene_cloud_; ///< scene point clou
    typename pcl::PointCloud<pcl::Normal>::ConstPtr scene_normals_; ///< scene normals cloud
    typename pcl::PointCloud<SceneT>::Ptr scene_cloud_downsampled_; ///< Downsampled scene point cloud
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals_downsampled_; ///< Downsampled scene normals cloud
    std::vector<int> scene_sampled_indices_;    ///< downsampled indices of the scene

    std::vector<std::vector<typename HVRecognitionModel<ModelT>::Ptr > > obj_hypotheses_groups_;
    std::vector<typename HVRecognitionModel<ModelT>::Ptr > global_hypotheses_;  ///< all hypotheses not rejected by individual verification

    std::map<std::string, boost::shared_ptr< pcl::octree::OctreePointCloudPointVector<ModelT> > > octree_model_representation_; ///< for each model we create an octree representation (used for computing visible points)
    typename pcl::search::KdTree<SceneT>::Ptr kdtree_scene_;
    std::vector<typename HVRecognitionModel<ModelT>::Ptr > recognition_models_; ///< all models to be verified

    double model_fitness_, pairwise_cost_, scene_fitness_;

    float Lmin_ = 0.f, Lmax_ = 100.f;
    int bins_ = 50;

    Eigen::MatrixXf intersection_cost_; ///< represents the pairwise intersection cost

    std::vector<std::vector<PtFitness> > scene_pts_explained_solution_;

    float initial_temp_;
    boost::shared_ptr<GHVCostFunctionLogger<ModelT,SceneT> > cost_logger_;
    Eigen::MatrixXf scene_color_channels_; ///< converted color values where each point corresponds to a row entry
    typename pcl::octree::OctreePointCloudSearch<SceneT>::Ptr octree_scene_downsampled_;
    boost::function<void (const boost::dynamic_bitset<> &, float, int)> visualize_cues_during_logger_;

    Eigen::VectorXi scene_pt_smooth_label_id_;  ///< stores a label for each point of the (downsampled) scene. Points belonging to the same smooth clusters, have the same label

    // ----- MULTI-VIEW VARIABLES------
    std::vector<typename pcl::PointCloud<SceneT>::ConstPtr> occlusion_clouds_; ///< scene clouds from multiple views (stored as organized point clouds)
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > absolute_camera_poses_;
    std::vector<boost::dynamic_bitset<> > model_is_present_in_view_; ///< for each model this variable stores information in which view it is present (used to check for visible model points - default all true = static scene)

    boost::function<float (const Eigen::VectorXf&, const Eigen::VectorXf&)> color_dist_f_;

    Eigen::Matrix4f refinePose(HVRecognitionModel<ModelT> &rm) const;

    cv::Mat img_boundary_distance_; ///< saves for each pixel how far it is away from the boundary (taking into account extrinsics of the camera)

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

    void visualizeGOcues(const boost::dynamic_bitset<> & active_solution, float cost, int times_evaluated) const
    {
        vis_cues_->visualize( this, active_solution, cost, times_evaluated );
    }

    void applySolution(const boost::dynamic_bitset<> &sol)
    {
        solution_ = sol;    // nothing needs to be updated as we always compute from scratch
    }

    void initialize ();

    mets::gol_type evaluateSolution (const boost::dynamic_bitset<> &solution);

    void optimize();

    /**
     * @brief isOutlier remove hypotheses with a lot of outliers. Returns true if hypothesis is rejected.
     * @param rm
     * @return
     */
    bool isOutlier(HVRecognitionModel<ModelT> &rm) const
    {
        return ( rm.confidence_ < param_.min_fitness_ );
    }


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
        obj_hypotheses_groups_.clear();
        scene_pt_smooth_label_id_.resize(0);
        scene_color_channels_.resize(0,0);
        scene_pts_explained_solution_.clear();
        kdtree_scene_.reset();
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
        float fit_3d = modelScene3DDistCostTerm (c);
        float fit_color = modelSceneColorCostTerm (c);
        float fit_normal = modelSceneNormalsCostTerm(c);

        if ( fit_3d < std::numeric_limits<float>::epsilon() ||
             fit_color < std::numeric_limits<float>::epsilon() ||
             fit_normal < std::numeric_limits<float>::epsilon()  )
            return 0.f;

        float sum_weights = param_.w_3D_ + param_.w_color_ + param_.w_normals_;
        float weighted_geometric_mean = exp( ( param_.w_3D_      * log(fit_3d) +
                                               param_.w_color_   * log(fit_color ) +
                                               param_.w_normals_ * log(fit_normal)  )
                                             / sum_weights );
        return weighted_geometric_mean;
    }

    /**
     * @brief modelSceneColorCostTerm
     * @param model scene correspondence
     * @return
     */
    float
    modelSceneColorCostTerm( const ModelSceneCorrespondence& c ) const
    {
//        return exp (- c.color_distance_/param_.color_sigma_ab_ );
//        std::cout << c.color_distance_ << std::endl;
//        return std::min(1.f, std::max(0.f, 1.f - c.color_distance_/param_.color_sigma_ab_));
        return 0.5f * (1.f-tanh( (c.color_distance_ - param_.color_sigma_ab_) / param_.color_sigma_l_ ) );
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
        if ( c.angle_surface_normals_rad_ < 0.f)
            return 0.f;

        return std::max(0.f, std::min<float>(1.f, 0.5f + 0.5f*tanh( (c.angle_surface_normals_rad_ - param_.inliers_surface_angle_thres_dotp_) / 0.1f ) ) );   ///TODO: Speed up with LUT

//        return 1 - (c.angle_surface_normals_rad_ - param_.inliers_surface_angle_thres_) / (M_PI/2 - param_.inliers_surface_angle_thres_);
    }

    void
    computeLOffset( HVRecognitionModel<ModelT> &rm ) const;

    float
    customColorDistance(const Eigen::VectorXf &color_a, const Eigen::VectorXf &color_b)
    {
        float L_dist  = ( color_a(0) - color_b(0) )*( color_a(0) - color_b(0) );
        CHECK(L_dist >= 0.f && L_dist <= 1.f);
        L_dist /= param_.color_sigma_l_ ;
        float AB_dist = ( color_a.tail(2) - color_b.tail(2) ).norm(); // ( param_.color_sigma_ab_ * param_.color_sigma_ab_ );
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
    customRegionGrowing (const SceneTWithNormal& seed_pt, const SceneTWithNormal& candidate_pt, float squared_distance) const;

    /**
     *  \brief: Returns a vector of booleans representing which hypotheses have been accepted/rejected (true/false)
     *  mask vector of booleans
     */
    boost::dynamic_bitset<>
    getSolution () const
    {
        return solution_;
    }

public:

    HypothesisVerification (const Camera::ConstPtr &cam,
                            const HV_Parameter &p = HV_Parameter())
        : param_(p), cam_(cam), initial_temp_(1000)
    {
        colorTransf_.reset(new RGB2CIELAB);

        switch (param_.color_comparison_method_)
        {
        case ColorComparisonMethod::cie76 :
            color_dist_f_ = CIE76; break;

        case ColorComparisonMethod::cie94 :
            color_dist_f_ = CIE94_DEFAULT; break;

        case ColorComparisonMethod::ciede2000 :
            color_dist_f_ = CIEDE2000; break;

        case 3:
            color_dist_f_ = boost::bind( &HypothesisVerification::customColorDistance, this, _1, _2 ); break;

        default:
            throw std::runtime_error("Color comparison method not defined!");
        }
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
     * @brief visualizeModelCues visualizes the model cues during computation. Useful for debugging
     * @param vis_params visualization parameters
     */
    void
    visualizeModelCues(const PCLVisualizationParams::ConstPtr &vis_params = boost::make_shared<PCLVisualizationParams>())
    {
        vis_model_.reset( new HV_ModelVisualizer<ModelT, SceneT>(vis_params) );
    }

    /**
     * @brief visualizeCues visualizes the cues during the computation and shows cost and number of evaluations. Useful for debugging
     * @param vis_params visualization parameters
     */
    void
    visualizeCues(const PCLVisualizationParams::ConstPtr &vis_params = boost::make_shared<PCLVisualizationParams>())
    {
        vis_cues_.reset( new HV_CuesVisualizer<ModelT, SceneT>(vis_params) );
    }

    /**
     * @brief visualizePairwiseCues visualizes the pairwise intersection of two hypotheses during computation. Useful for debugging
     * @param vis_params visualization parameters
     */
    void
    visualizePairwiseCues(const PCLVisualizationParams::ConstPtr &vis_params = boost::make_shared<PCLVisualizationParams>())
    {
        vis_pairwise_.reset ( new HV_PairwiseVisualizer<ModelT, SceneT>(vis_params) );
    }


    /**
     *  \brief Function that performs the hypotheses verification
     *  This function modifies the values of mask_ and needs to be called after both scene and model have been added
     */
    void
    verify();
};

}
