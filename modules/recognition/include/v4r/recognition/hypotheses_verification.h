/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef V4R_HYPOTHESIS_VERIFICATION_H__
#define V4R_HYPOTHESIS_VERIFICATION_H__

#include <v4r/core/macros.h>
#include <v4r/recognition/recognition_model_hv.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>

namespace v4r
{

  /**
   * \brief Abstract class for hypotheses verification methods
   * \author Aitor Aldoma, Federico Tombari, Thomas Faeulhammer
   */
  template<typename ModelT, typename SceneT>
  class V4R_EXPORTS HypothesisVerification
  {
  public:
      class V4R_EXPORTS Parameter
      {
      public:
          double resolution_; /// @brief The resolution of models and scene used to verify hypotheses (in meters)
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

          Parameter (
                  double resolution = 0.005f,
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
                  int icp_iterations = 10)
              : resolution_ (resolution),
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
                icp_iterations_ (icp_iterations)
          {}
      }param_;

  protected:
    std::vector<bool> solution_; /// @brief Boolean vector indicating if a hypothesis is accepted (true) or rejected (false)

    typename pcl::PointCloud<SceneT>::ConstPtr scene_cloud_; /// @brief scene point cloud

    std::vector<int> recognition_models_map_;

    typename pcl::PointCloud<SceneT>::Ptr scene_cloud_downsampled_; /// \brief Downsampled scene point cloud

    std::vector<boost::shared_ptr<HVRecognitionModel<ModelT> > > recognition_models_; /// @brief all models to be verified (including planar models if included)

    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > refined_model_transforms_; /// @brief fine registration of model clouds to scene clouds after ICP (this applies to object model only - not to planes)

    std::vector<int> scene_sampled_indices_;

    // ----- MULTI-VIEW VARIABLES------
    std::vector<typename pcl::PointCloud<SceneT>::ConstPtr> occlusion_clouds_; /// @brief scene clouds from multiple views
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > absolute_camera_poses_;
    std::vector<std::vector<bool> > model_is_present_in_view_; /// \brief for each model this variable stores information in which view it is present (used to check for visible model points - default all true = static scene)

    Eigen::Matrix4f poseRefinement(const HVRecognitionModel<ModelT> &rm) const;

    void computeVisibleModelsAndRefinePose();

    void cleanUp()
    {
        recognition_models_.clear();
        recognition_models_map_.clear();
        occlusion_clouds_.clear();
        absolute_camera_poses_.clear();
        scene_sampled_indices_.clear();
        model_is_present_in_view_.clear();
        scene_cloud_downsampled_.reset();
        scene_cloud_.reset();
    }

  public:
    HypothesisVerification (const Parameter &p = Parameter()) : param_(p)
    { }

    float getResolution() const
    {
        return param_.resolution_;
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
     * @brief Sets the models (recognition hypotheses)
     * @param models vector of point clouds representing the models (in same coordinates as the scene_cloud_)
     * @param corresponding normal clouds
     */
    void
    addModels (std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models,
               std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr > &model_normals);


    /**
     *  \brief Sets the scene cloud
     *  \param scene_cloud Point cloud representing the scene
     */
    void
    setSceneCloud (const typename pcl::PointCloud<SceneT>::Ptr & scene_cloud);

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

    /**
     * @brief returns the refined transformation matrix aligning model with scene cloud (applies to object models only - not plane clouds) and is in order of the input of addmodels
     * @return
     */
    void
    getRefinedTransforms(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &tf) const
    {
        tf = refined_model_transforms_;
    }

    /**
     *  \brief Function that performs the hypotheses verification, needs to be implemented in the subclasses
     *  This function modifies the values of mask_ and needs to be called after both scene and model have been added
     */
    virtual void verify() = 0;
  };

}

#endif /* PCL_RECOGNITION_HYPOTHESIS_VERIFICATION_H_ */
