#ifndef V4R_RECOGNITION_MODEL_HV_H__
#define V4R_RECOGNITION_MODEL_HV_H__

#include <v4r/core/macros.h>
#include <pcl/common/common.h>
#include <v4r/common/common_data_structures.h>

namespace v4r
{
  template<typename ModelT>
  class V4R_EXPORTS HVRecognitionModel
  {
    public:
      class Parameter
      {
      public:
          int outliers_weight_computation_method_;
          Parameter( int outliers_weight_computation_method = OutliersWeightType::MEAN ) :
              outliers_weight_computation_method_ (outliers_weight_computation_method)
          {}
      }param_;

      double outliers_total_weight_;
      std::vector<int> outlier_indices_; /// @brief outlier indices of this model (coming from all types)
      std::vector<bool> scene_pt_is_explained_; /// @brief boolean vector indicating if a scene point is explained by this model or not

      typename pcl::PointCloud<ModelT>::Ptr visible_cloud_;
      typename pcl::PointCloud<ModelT>::Ptr complete_cloud_;
      pcl::PointCloud<pcl::Normal>::Ptr visible_cloud_normals_;
      pcl::PointCloud<pcl::Normal>::Ptr complete_cloud_normals_;
      std::vector<int> visible_indices_;
      std::vector< std::vector<float> > noise_term_visible_pt_; /// @brief expected (axial and lateral) noise level at visible point
      pcl::Correspondences model_scene_c_; /// @brief correspondences between visible model points and scene
      double model_fit_; /// @brief the fitness score of the visible cloud to the model scene (sum of model_scene_c correspondenes weight divided by the number of visible points)

      float extra_weight_; /// @brief descriptor distance weight for instance
      Eigen::MatrixXf pt_color_;  /// @brief color values for each visible point of the model (row_id). Width is equal to the number of color channels
      Eigen::MatrixXf local_pt_color_;  /// @brief color description for each visible point of the model (row_id) as a function of its neighboring pixels. The columns represent average and standard deviation of each color channel (column[0-1]: avg+std color channel[0], column[2-3]: avg+std color channel[1], ...)
      std::vector<int> cloud_indices_specified_;
      typename PlaneModel<ModelT>::Ptr plane_model_;

      //inlier indices and distances for cloud_ (this avoids recomputing radius searches twice (one for specification and one for inlier/outlier detection)
      std::vector<std::vector<int> > scene_inlier_indices_for_visible_pt_;
      std::vector<std::vector<float> > scene_inlier_distances_for_visible_pt_;

      HVRecognitionModel() : extra_weight_(1.f)
      { }

      enum OutliersWeightType{
          MEAN,
          MEDIAN
      };

      typedef boost::shared_ptr< HVRecognitionModel> Ptr;
      typedef boost::shared_ptr< HVRecognitionModel const> ConstPtr;
  };
}

#endif
