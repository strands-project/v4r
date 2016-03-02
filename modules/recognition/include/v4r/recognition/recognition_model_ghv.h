#ifndef V4R_RECOGNITION_MODEL_GHV_H__
#define V4R_RECOGNITION_MODEL_GHV_H__

#include <v4r/core/macros.h>
#include <v4r/recognition/recognition_model_hv.h>
#include <pcl/common/common.h>

namespace v4r
{
  template<typename ModelT>
  class V4R_EXPORTS GHVRecognitionModel : public HVRecognitionModel<ModelT>
  {
    public:
      using HVRecognitionModel<ModelT>::explained_;
      using HVRecognitionModel<ModelT>::explained_distances_;
      using HVRecognitionModel<ModelT>::unexplained_in_neighborhood;
      using HVRecognitionModel<ModelT>::unexplained_in_neighborhood_weights;
      using HVRecognitionModel<ModelT>::outlier_indices_;
      using HVRecognitionModel<ModelT>::color_outliers_indices_;
      using HVRecognitionModel<ModelT>::outliers_3d_indices_;
      using HVRecognitionModel<ModelT>::complete_cloud_occupancy_indices_;
      using HVRecognitionModel<ModelT>::scene_point_explained_by_hypothesis_;
      using HVRecognitionModel<ModelT>::visible_cloud_;
      using HVRecognitionModel<ModelT>::complete_cloud_;
      using HVRecognitionModel<ModelT>::visible_cloud_normals_;
      using HVRecognitionModel<ModelT>::complete_cloud_normals_;
      using HVRecognitionModel<ModelT>::visible_indices_;
      using HVRecognitionModel<ModelT>::bad_information_;
      using HVRecognitionModel<ModelT>::outliers_weight_;
      using HVRecognitionModel<ModelT>::id_;
      using HVRecognitionModel<ModelT>::extra_weight_;
      using HVRecognitionModel<ModelT>::color_similarity_;
      using HVRecognitionModel<ModelT>::median_;
      using HVRecognitionModel<ModelT>::mean_;
      using HVRecognitionModel<ModelT>::color_mapping_;
      using HVRecognitionModel<ModelT>::hyp_penalty_;
      using HVRecognitionModel<ModelT>::id_s_;
      using HVRecognitionModel<ModelT>::cloud_color_channels_;
      using HVRecognitionModel<ModelT>::cloud_GS_;
      using HVRecognitionModel<ModelT>::min_contribution_;
      using HVRecognitionModel<ModelT>::normal_angle_histogram_;
      using HVRecognitionModel<ModelT>::color_diff_histogram_;
      using HVRecognitionModel<ModelT>::normal_entropy_;
      using HVRecognitionModel<ModelT>::color_entropy_;
      using HVRecognitionModel<ModelT>::cloud_indices_specified_;
      using HVRecognitionModel<ModelT>::color_diff_trhough_specification_;
      using HVRecognitionModel<ModelT>::visible_labels_;
      using HVRecognitionModel<ModelT>::is_planar_;
      using HVRecognitionModel<ModelT>::inlier_indices_;
      using HVRecognitionModel<ModelT>::inlier_distances_;
      using HVRecognitionModel<ModelT>::plane_model_;
      using HVRecognitionModel<ModelT>::smooth_faces_;


      typedef boost::shared_ptr< GHVRecognitionModel> Ptr;
      typedef boost::shared_ptr< GHVRecognitionModel const> ConstPtr;
  };
}

#endif
