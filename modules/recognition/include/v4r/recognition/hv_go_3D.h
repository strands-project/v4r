/*
 * hv_go_3D.h
 *
 *  Created on: Sep 9, 2013
 *      Author: aitor
 */

#ifndef HV_GO_3D_H_
#define HV_GO_3D_H_

#include <pcl/common/common.h>
#include <pcl/pcl_macros.h>
#include "hypotheses_verification.h"
#include "ghv.h"
#include <metslib/mets.hh>
#include <pcl/features/normal_3d.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <map>
#include <iostream>
#include <fstream>
#include <pcl/common/time.h>
#include <pcl/segmentation/supervoxel_clustering.h>

namespace v4r
{
  template<typename ModelT, typename SceneT>
  //class FAAT_REC_API GO3D : public v4r::GlobalHypothesesVerification_1<ModelT, SceneT>
  class V4R_EXPORTS GO3D : public v4r::GHV<ModelT, SceneT>
  {

  public:
    class V4R_EXPORTS Parameter : public v4r::GHV<ModelT, SceneT>::Parameter
    {
    public:
         using v4r::GHV<ModelT, SceneT>::Parameter::color_sigma_ab_;
         using v4r::GHV<ModelT, SceneT>::Parameter::color_sigma_l_;
         using v4r::GHV<ModelT, SceneT>::Parameter::regularizer_;
         using v4r::GHV<ModelT, SceneT>::Parameter::radius_neighborhood_clutter_;
         using v4r::GHV<ModelT, SceneT>::Parameter::radius_normals_;
         using v4r::GHV<ModelT, SceneT>::Parameter::duplicy_weight_test_;
         using v4r::GHV<ModelT, SceneT>::Parameter::duplicity_curvature_max_;
         using v4r::GHV<ModelT, SceneT>::Parameter::ignore_color_even_if_exists_;
         using v4r::GHV<ModelT, SceneT>::Parameter::max_iterations_;
         using v4r::GHV<ModelT, SceneT>::Parameter::clutter_regularizer_;
         using v4r::GHV<ModelT, SceneT>::Parameter::detect_clutter_;
         using v4r::GHV<ModelT, SceneT>::Parameter::res_occupancy_grid_;
         using v4r::GHV<ModelT, SceneT>::Parameter::w_occupied_multiple_cm_;
         using v4r::GHV<ModelT, SceneT>::Parameter::use_super_voxels_;
         using v4r::GHV<ModelT, SceneT>::Parameter::use_replace_moves_;
         using v4r::GHV<ModelT, SceneT>::Parameter::opt_type_;
         using v4r::GHV<ModelT, SceneT>::Parameter::active_hyp_penalty_;
         using v4r::GHV<ModelT, SceneT>::Parameter::multiple_assignment_penalize_by_one_;
         using v4r::GHV<ModelT, SceneT>::Parameter::d_weight_for_bad_normals_;
         using v4r::GHV<ModelT, SceneT>::Parameter::use_clutter_exp_;
         using v4r::GHV<ModelT, SceneT>::Parameter::use_histogram_specification_;
         using v4r::GHV<ModelT, SceneT>::Parameter::use_points_on_plane_side_;
         using v4r::GHV<ModelT, SceneT>::Parameter::best_color_weight_;
         using v4r::GHV<ModelT, SceneT>::Parameter::eps_angle_threshold_;
         using v4r::GHV<ModelT, SceneT>::Parameter::min_points_;
         using v4r::GHV<ModelT, SceneT>::Parameter::curvature_threshold_;
         using v4r::GHV<ModelT, SceneT>::Parameter::cluster_tolerance_;
         using v4r::GHV<ModelT, SceneT>::Parameter::use_normals_from_visible_;

        Parameter()
        {}

    };

  private:
    using v4r::GHV<ModelT, SceneT>::mask_;
    using v4r::GHV<ModelT, SceneT>::scene_cloud_downsampled_;
    using v4r::GHV<ModelT, SceneT>::scene_downsampled_tree_;
    using v4r::GHV<ModelT, SceneT>::visible_models_;
    using v4r::GHV<ModelT, SceneT>::visible_normal_models_;
    using v4r::GHV<ModelT, SceneT>::visible_indices_;
    using v4r::GHV<ModelT, SceneT>::complete_models_;
    using v4r::GHV<ModelT, SceneT>::normals_set_;
    using v4r::GHV<ModelT, SceneT>::requires_normals_;
    using v4r::GHV<ModelT, SceneT>::object_ids_;
    using v4r::GHV<ModelT, SceneT>::extra_weights_;
    using v4r::GHV<ModelT, SceneT>::scene_normals_;
    using v4r::GHV<ModelT, SceneT>::recognition_models_;
    using v4r::GHV<ModelT, SceneT>::computeRGBHistograms;
    using v4r::GHV<ModelT, SceneT>::specifyRGBHistograms;
    using v4r::GHV<ModelT, SceneT>::unexplained_by_RM_neighboorhods;
    using v4r::GHV<ModelT, SceneT>::explained_by_RM_distance_weighted;
    using v4r::GHV<ModelT, SceneT>::explained_by_RM_;
    using v4r::GHV<ModelT, SceneT>::complete_cloud_occupancy_by_RM_;
    using v4r::GHV<ModelT, SceneT>::octree_scene_downsampled_;
    using v4r::GHV<ModelT, SceneT>::cc_;
    using v4r::GHV<ModelT, SceneT>::n_cc_;
    using v4r::GHV<ModelT, SceneT>::valid_model_;
    using v4r::GHV<ModelT, SceneT>::clusters_cloud_rgb_;
    using v4r::GHV<ModelT, SceneT>::clusters_cloud_;
    using v4r::GHV<ModelT, SceneT>::points_explained_by_rm_;
    using v4r::GHV<ModelT, SceneT>::extractEuclideanClustersSmooth;
    using v4r::GHV<ModelT, SceneT>::complete_normal_models_;
    using v4r::GHV<ModelT, SceneT>::scene_LAB_values_;
    using v4r::GHV<ModelT, SceneT>::scene_RGB_values_;
    using v4r::GHV<ModelT, SceneT>::scene_GS_values_;
    using v4r::GHV<ModelT, SceneT>::computeClutterCueAtOnce;

    //typename pcl::PointCloud<SceneT>::Ptr scene_cloud_downsampled_GO3D_;
    //typename pcl::PointCloud<pcl::Normal>::Ptr scene_normals_go3D_;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > absolute_poses_camera_to_global_;
    std::vector<typename pcl::PointCloud<SceneT>::ConstPtr > occ_clouds_;

    static float sRGB_LUT[256];
    static float sXYZ_LUT[4000];

    //////////////////////////////////////////////////////////////////////////////////////////////
    void
    RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2)
    {
      if (sRGB_LUT[0] < 0)
      {
        for (int i = 0; i < 256; i++)
        {
          float f = static_cast<float> (i) / 255.0f;
          if (f > 0.04045)
            sRGB_LUT[i] = powf ((f + 0.055f) / 1.055f, 2.4f);
          else
            sRGB_LUT[i] = f / 12.92f;
        }

        for (int i = 0; i < 4000; i++)
        {
          float f = static_cast<float> (i) / 4000.0f;
          if (f > 0.008856)
            sXYZ_LUT[i] = static_cast<float> (powf (f, 0.3333f));
          else
            sXYZ_LUT[i] = static_cast<float>((7.787 * f) + (16.0 / 116.0));
        }
      }

      float fr = sRGB_LUT[R];
      float fg = sRGB_LUT[G];
      float fb = sRGB_LUT[B];

      // Use white = D65
      const float x = fr * 0.412453f + fg * 0.357580f + fb * 0.180423f;
      const float y = fr * 0.212671f + fg * 0.715160f + fb * 0.072169f;
      const float z = fr * 0.019334f + fg * 0.119193f + fb * 0.950227f;

      float vx = x / 0.95047f;
      float vy = y;
      float vz = z / 1.08883f;

      vx = sXYZ_LUT[int(vx*4000)];
      vy = sXYZ_LUT[int(vy*4000)];
      vz = sXYZ_LUT[int(vz*4000)];

      L = 116.0f * vy - 16.0f;
      if (L > 100)
        L = 100.0f;

      A = 500.0f * (vx - vy);
      if (A > 120)
        A = 120.0f;
      else if (A <- 120)
        A = -120.0f;

      B2 = 200.0f * (vy - vz);
      if (B2 > 120)
        B2 = 120.0f;
      else if (B2<- 120)
        B2 = -120.0f;
    }

    typedef pcl::PointCloud<ModelT> CloudM;
    typedef pcl::PointCloud<SceneT> CloudS;
    typedef typename pcl::traits::fieldList<typename CloudS::PointType>::type FieldListS;
    typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;
    typedef typename pcl::NormalEstimation<SceneT, pcl::Normal> NormalEstimator_;

    /*bool
    addModel (int i, boost::shared_ptr<RecognitionModel<ModelT> > & recog_model);*/

      //void initialize ();

      /*bool
      handlingNormals (boost::shared_ptr<GHVRecognitionModel<ModelT> > & recog_model, int i, bool is_planar_model, int object_models_size);*/

    public:

      Parameter param_;

      GO3D(const Parameter &p=Parameter())
      {
         param_ = p;
      }

      /*void setSceneAndNormals(typename pcl::PointCloud<SceneT>::Ptr & scene_cloud_downsampled_GO3D,
                              typename pcl::PointCloud<pcl::Normal>::Ptr & scene_normals_go3D)
      {
          scene_cloud_downsampled_GO3D_ = scene_cloud_downsampled_GO3D;
          scene_normals_go3D_ = scene_normals_go3D;
      }

      typename pcl::PointCloud<SceneT>::Ptr getSceneCloud()
      {
        return scene_cloud_downsampled_GO3D_;
      }*/

      bool getInlierOutliersCloud(int hyp_idx, typename pcl::PointCloud<ModelT>::Ptr & cloud);

      //for each cloud, we will need a pose
      //then the models will be checked against all of them
      void setAbsolutePoses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & absolute_poses_camera_to_global)
      {
        absolute_poses_camera_to_global_ = absolute_poses_camera_to_global;
      }

      void
      setOcclusionsClouds(std::vector<typename pcl::PointCloud<SceneT>::ConstPtr > & occ_clouds)
      {
        occ_clouds_ = occ_clouds;
      }

      void
      addModels (std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models, bool occlusion_reasoning = false);

      std::vector<typename pcl::PointCloud<ModelT>::ConstPtr>
      getVisibleModels()
      {
        return visible_models_;
      }
  };
}

#endif /* HV_GO_3D_H_ */
