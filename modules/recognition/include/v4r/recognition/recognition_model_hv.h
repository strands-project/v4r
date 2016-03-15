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

      Eigen::MatrixXf pt_color_;  /// @brief color values for each visible point of the model (row_id). Width is equal to the number of color channels

      HVRecognitionModel()
      { }

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
