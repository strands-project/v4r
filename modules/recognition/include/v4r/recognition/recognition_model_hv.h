#ifndef V4R_RECOGNITION_MODEL_HV_H__
#define V4R_RECOGNITION_MODEL_HV_H__

#include <v4r/core/macros.h>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/correspondence.h>

namespace v4r
{
  template<typename ModelT>
  class V4R_EXPORTS HVRecognitionModel
  {
    public:
      typename pcl::PointCloud<ModelT>::Ptr complete_cloud_;
      typename pcl::PointCloud<ModelT>::Ptr visible_cloud_;
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

      HVRecognitionModel()
      {
          L_value_offset_ = 0.f;
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
}

#endif
