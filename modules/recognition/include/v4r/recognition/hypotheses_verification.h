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

#ifndef FAATPCL_RECOGNITION_HYPOTHESIS_VERIFICATION_H_
#define FAATPCL_RECOGNITION_HYPOTHESIS_VERIFICATION_H_

#include <pcl/pcl_macros.h>
#include <v4r/core/macros.h>
#include <v4r/common/common_data_structures.h>
#include <v4r/common/zbuffering.h>
#include <v4r/recognition/recognition_model_hv.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>

namespace v4r
{

  /**
   * \brief Abstract class for hypotheses verification methods
   * \author Aitor Aldoma, Federico Tombari
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
          int zbuffer_scene_resolution_; /// @brief Resolutions in pixel for the depth scene buffer
          int zbuffer_self_occlusion_resolution_;
          bool self_occlusions_reasoning_;
          double focal_length_; /// @brief defines the focal length used for back-projecting points to the image plane (used for occlusion / visibility reasoning)
          bool do_occlusion_reasoning_;

          Parameter (
                  double resolution = 0.005f,
                  double inliers_threshold = 0.015f, // 0.005f
                  double occlusion_thres = 0.02f, // 0.005f
                  int zbuffer_scene_resolution = 100,
                  int zbuffer_self_occlusion_resolution = 250,
                  bool self_occlusions_reasoning = true,
                  double focal_length = 525.f,
                  bool do_occlusion_reasoning = true)
              : resolution_ (resolution),
                inliers_threshold_(inliers_threshold),
                occlusion_thres_ (occlusion_thres),
                zbuffer_scene_resolution_(zbuffer_scene_resolution),
                zbuffer_self_occlusion_resolution_(zbuffer_self_occlusion_resolution),
                self_occlusions_reasoning_(self_occlusions_reasoning),
                focal_length_ (focal_length),
                do_occlusion_reasoning_ (do_occlusion_reasoning)
          {}
      }param_;

  protected:
    /**
     * @brief Boolean vector indicating if a hypothesis is accepted/rejected (output of HV stage)
     */
    std::vector<bool> mask_;

    /**
     * @brief Scene point cloud
     */
    typename pcl::PointCloud<SceneT>::ConstPtr scene_cloud_;

    /**
     * \brief Scene point cloud
     */
    typename pcl::PointCloud<SceneT>::ConstPtr occlusion_cloud_;

    bool occlusion_cloud_set_;

    /**
     * \brief Downsampled scene point cloud
     */
     typename pcl::PointCloud<SceneT>::Ptr scene_cloud_downsampled_;

    /**
     * \brief Scene tree of the downsampled cloud
     */
    typename pcl::search::KdTree<SceneT>::Ptr scene_downsampled_tree_;

    /**
     * \brief Vector of point clouds representing the 3D models after occlusion reasoning
	 * the 3D models are pruned of occluded points, and only visible points are left. 
	 * the coordinate system is that of the scene cloud
     */
    std::vector< std::vector<bool> > model_point_is_visible_;

    std::vector<boost::shared_ptr<HVRecognitionModel<ModelT> > > recognition_models_; /// @brief all models to be verified (including planar models if included)

    bool requires_normals_; /// \brief Whether the HV method requires normals or not, by default = false
    bool normals_set_; /// \brief Whether the normals have been set

    std::vector<int> scene_sampled_indices_;

  public:
    HypothesisVerification (const Parameter &p = Parameter()) : param_(p)
    {
      occlusion_cloud_set_ = false;
      normals_set_ = false;
      requires_normals_ = false;
    }

    bool getRequiresNormals()
    {
      return requires_normals_;
    }

    float getResolution() const
    {
        return param_.resolution_;
    }

    /**
     *  \brief: Returns a vector of booleans representing which hypotheses have been accepted/rejected (true/false)
     *  mask vector of booleans
     */
    void
    getMask (std::vector<bool> & mask)
    {
      mask = mask_;
    }

    /**
     *  \brief Sets the normals of the 3D complete models and sets normals_set_ to true.
     *  Normals need to be added before calling the addModels method.
     *  complete_models The normals of the models.
     */
    void
    addNormalsClouds (std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> & complete_models)
    {
//      complete_normal_models_ = complete_models;
        (void)complete_models;
        throw std::runtime_error("This function is not properly implemented right now!");
      normals_set_ = true;
    }

    /**
     *  \brief Sets the models (recognition hypotheses) - requires the scene_cloud_ to be set first if reasoning about occlusions
     *  mask models Vector of point clouds representing the models (in same coordinates as the scene_cloud_)
     */
    virtual
    void
    addModels (std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models,
               std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr > &model_normals = std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr >());


    /**
     *  \brief Sets the scene cloud
     *  \param scene_cloud Point cloud representing the scene
     */
    void
    setSceneCloud (const typename pcl::PointCloud<SceneT>::Ptr & scene_cloud);

    void setOcclusionCloud (const typename pcl::PointCloud<SceneT>::Ptr & occ_cloud)
    {
      occlusion_cloud_ = occ_cloud;
      occlusion_cloud_set_ = true;
    }

    /**
     *  \brief Function that performs the hypotheses verification, needs to be implemented in the subclasses
     *  This function modifies the values of mask_ and needs to be called after both scene and model have been added
     */
    virtual void verify() = 0;
  };

}

#endif /* PCL_RECOGNITION_HYPOTHESIS_VERIFICATION_H_ */
