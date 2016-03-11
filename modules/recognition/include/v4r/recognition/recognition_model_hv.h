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
      std::vector<bool> image_mask_;
      pcl::PointCloud<pcl::Normal>::Ptr visible_cloud_normals_;
      pcl::PointCloud<pcl::Normal>::Ptr complete_cloud_normals_;
      std::vector<int> visible_indices_;
      pcl::Correspondences model_scene_c_; /// @brief correspondences between visible model points and scene
      double model_fit_; /// @brief the fitness score of the visible cloud to the model scene (sum of model_scene_c correspondenes weight divided by the number of visible points)

      Eigen::MatrixXf pt_color_;  /// @brief color values for each visible point of the model (row_id). Width is equal to the number of color channels

      HVRecognitionModel()
      { }

      typedef boost::shared_ptr< HVRecognitionModel> Ptr;
      typedef boost::shared_ptr< HVRecognitionModel const> ConstPtr;
  };
}

#endif
