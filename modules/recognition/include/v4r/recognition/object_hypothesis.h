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


#ifndef V4R_OBJECT_HYPOTHESIS_H__
#define V4R_OBJECT_HYPOTHESIS_H__

#include <v4r/core/macros.h>
#include <pcl/common/common.h>
#include <v4r/recognition/model.h>
#include <Eigen/StdVector>


namespace v4r
{
    template<typename PointT>
    class V4R_EXPORTS ObjectHypothesis
    {
    public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typename Model<PointT>::Ptr model_; /// @brief object instance model.
        Eigen::Matrix4f transform_; /// @brief 4x4 homogenous transformation to project model into camera coordinate system.
        float confidence_; /// @brief confidence score (coming from feature matching stage)

        typedef boost::shared_ptr< ObjectHypothesis<PointT> > Ptr;
        typedef boost::shared_ptr< ObjectHypothesis<PointT> const> ConstPtr;
    };



    template<typename PointT>
    class V4R_EXPORTS HVRecognitionModel : public ObjectHypothesis<PointT>
    {
      public:
        typename pcl::PointCloud<PointT>::Ptr complete_cloud_;
        typename pcl::PointCloud<PointT>::Ptr visible_cloud_;
        std::vector<std::vector<bool> > image_mask_; /// @brief image mask per view (in single-view case, there will be only one element in outer vector). Used to compute pairwise intersection
        pcl::PointCloud<pcl::Normal>::Ptr visible_cloud_normals_;
        pcl::PointCloud<pcl::Normal>::Ptr complete_cloud_normals_;
        std::vector<int> visible_indices_;
        pcl::Correspondences model_scene_c_; /// @brief correspondences between visible model points and scene
        double model_fit_; /// @brief the fitness score of the visible cloud to the model scene (sum of model_scene_c correspondenes weight divided by the number of visible points)

        Eigen::MatrixXf pt_color_;  /// @brief color values for each point of the (complete) model (row_id). Width is equal to the number of color channels
        float mean_brigthness_;   /// @brief average value of the L channel for all visible model points
        float mean_brigthness_scene_;   /// @brief average value of the L channel for all scene points close to the visible model points
        std::vector<int> scene_indices_in_crop_box_; /// @brief indices of the scene that are occupied from the bounding box of the (complete) hypothesis
        float L_value_offset_; /// @brief the offset being added to the computed L color values to compensate for different lighting conditions

        std::vector<size_t> explained_pts_per_smooth_cluster_ ; /// @brief counts how many points in each smooth cluster the recognition model explains

        Eigen::Matrix4f refined_pose_;
        Eigen::VectorXf scene_model_sqr_dist_;   /// @brief stores for each scene point the distance to its closest visible model point (negative if distance too large)
        Eigen::VectorXf scene_explained_weight_;   /// @brief stores for each scene point how well it is explained by the visible model points

        bool rejected_due_to_low_visibility_;   ///@brief true if the object model rendered in the view is not visible enough
        bool rejected_due_to_low_model_fitness_;    ///@brief true if the object model is not able to explain the scene well enough
        bool rejected_due_to_smooth_cluster_check_; /// @brief true if the object model does not well explain all points in the smooth clusters it occupies
        bool rejected_globally_;

        HVRecognitionModel()
        {
            L_value_offset_ = 0.f;
            refined_pose_ = Eigen::Matrix4f::Identity();
            rejected_due_to_low_visibility_ = rejected_due_to_low_model_fitness_ =
                    rejected_due_to_smooth_cluster_check_ = rejected_globally_ = false;
        }

        HVRecognitionModel(const ObjectHypothesis<PointT> &oh) : ObjectHypothesis<PointT>(oh)
        {
            L_value_offset_ = 0.f;
            refined_pose_ = Eigen::Matrix4f::Identity();
            rejected_due_to_low_visibility_ = rejected_due_to_low_model_fitness_ =
                    rejected_due_to_smooth_cluster_check_ = rejected_globally_ = false;
        }

        void
        freeSpace()
        {
            complete_cloud_.reset();
            visible_cloud_.reset();
            visible_cloud_normals_.reset();
            complete_cloud_normals_.reset();
            visible_indices_.clear();
            image_mask_.clear();
            model_scene_c_.clear();
            pt_color_.resize(0,0);
            scene_indices_in_crop_box_.clear();
            explained_pts_per_smooth_cluster_.clear();
            scene_model_sqr_dist_.resize(0);
            scene_explained_weight_.resize(0);
        }

        bool
        isRejected() const
        {
            return rejected_due_to_low_model_fitness_ || rejected_due_to_low_visibility_ || rejected_due_to_smooth_cluster_check_ || rejected_globally_;
        }

        /**
         * @brief does dilation and erosion on the occupancy image of the rendered point cloud
         * @param do_smoothing
         * @param smoothing_radius
         * @param do_erosion
         * @param erosion_radius
         * @param img_width
         */
        void
        processSilhouette(bool do_smoothing=true, size_t smoothing_radius=2, bool do_erosion=true, size_t erosion_radius=4, size_t img_width=640);

        typedef boost::shared_ptr< HVRecognitionModel> Ptr;
        typedef boost::shared_ptr< HVRecognitionModel const> ConstPtr;
    };



    template<typename PointT>
    class V4R_EXPORTS ObjectHypothesesGroup
    {
    public:
        std::vector<typename ObjectHypothesis<PointT>::Ptr > ohs_; /// @brief Each hypothesis can have several object model (e.g. global recognizer tries to macht several object instances for a clustered point cloud segment).
        bool global_hypotheses_; /// @brief if true, hypothesis was generated by global recognition pipeline. Otherwise, from local feature matches-

        typedef boost::shared_ptr< ObjectHypothesesGroup > Ptr;
        typedef boost::shared_ptr< ObjectHypothesesGroup const> ConstPtr;
    };
}

#endif
