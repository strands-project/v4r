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
  class V4R_EXPORTS GO3D : public GHV<ModelT, SceneT>
  {

  public:
    class V4R_EXPORTS Parameter : public GHV<ModelT, SceneT>::Parameter
    {
    public:
         using GHV<ModelT, SceneT>::Parameter::color_sigma_ab_;
         using GHV<ModelT, SceneT>::Parameter::color_sigma_l_;
         using GHV<ModelT, SceneT>::Parameter::regularizer_;
         using GHV<ModelT, SceneT>::Parameter::radius_neighborhood_clutter_;
         using GHV<ModelT, SceneT>::Parameter::radius_normals_;
         using GHV<ModelT, SceneT>::Parameter::duplicy_weight_test_;
         using GHV<ModelT, SceneT>::Parameter::duplicity_curvature_max_;
         using GHV<ModelT, SceneT>::Parameter::ignore_color_even_if_exists_;
         using GHV<ModelT, SceneT>::Parameter::max_iterations_;
         using GHV<ModelT, SceneT>::Parameter::clutter_regularizer_;
         using GHV<ModelT, SceneT>::Parameter::detect_clutter_;
         using GHV<ModelT, SceneT>::Parameter::res_occupancy_grid_;
         using GHV<ModelT, SceneT>::Parameter::w_occupied_multiple_cm_;
         using GHV<ModelT, SceneT>::Parameter::use_super_voxels_;
         using GHV<ModelT, SceneT>::Parameter::use_replace_moves_;
         using GHV<ModelT, SceneT>::Parameter::opt_type_;
         using GHV<ModelT, SceneT>::Parameter::active_hyp_penalty_;
         using GHV<ModelT, SceneT>::Parameter::multiple_assignment_penalize_by_one_;
         using GHV<ModelT, SceneT>::Parameter::d_weight_for_bad_normals_;
         using GHV<ModelT, SceneT>::Parameter::use_clutter_exp_;
         using GHV<ModelT, SceneT>::Parameter::use_histogram_specification_;
         using GHV<ModelT, SceneT>::Parameter::use_points_on_plane_side_;
         using GHV<ModelT, SceneT>::Parameter::best_color_weight_;
         using GHV<ModelT, SceneT>::Parameter::eps_angle_threshold_;
         using GHV<ModelT, SceneT>::Parameter::min_points_;
         using GHV<ModelT, SceneT>::Parameter::curvature_threshold_;
         using GHV<ModelT, SceneT>::Parameter::cluster_tolerance_;
         using GHV<ModelT, SceneT>::Parameter::use_normals_from_visible_;

        Parameter(const typename GHV<ModelT, SceneT>::Parameter &p = typename GHV<ModelT, SceneT>::Parameter())
        {
            color_sigma_ab_ = p.color_sigma_ab_;
            color_sigma_l_ = p.color_sigma_l_;
            regularizer_ = p.regularizer_;
            radius_neighborhood_clutter_ = p.radius_neighborhood_clutter_;
            radius_normals_ = p.radius_normals_;
            duplicy_weight_test_ = p.duplicy_weight_test_;
            duplicity_curvature_max_ = p.duplicity_curvature_max_;
            ignore_color_even_if_exists_ = p.ignore_color_even_if_exists_;
            max_iterations_ = p.max_iterations_;
            clutter_regularizer_ = p.clutter_regularizer_;
            detect_clutter_ = p.detect_clutter_;
            res_occupancy_grid_ = p.res_occupancy_grid_;
            w_occupied_multiple_cm_ = p.w_occupied_multiple_cm_;
            use_super_voxels_ = p.use_super_voxels_;
            use_replace_moves_ = p.use_replace_moves_;
            opt_type_ = p.opt_type_;
            active_hyp_penalty_ = p.active_hyp_penalty_;
            multiple_assignment_penalize_by_one_ = p.multiple_assignment_penalize_by_one_;
            d_weight_for_bad_normals_ = p.d_weight_for_bad_normals_;
            use_clutter_exp_ = p.use_clutter_exp_;
            use_histogram_specification_ = p.use_histogram_specification_;
            use_points_on_plane_side_ = p.use_points_on_plane_side_;
            best_color_weight_ = p.best_color_weight_;
//            initial_status_ = p.initial_status_;
            eps_angle_threshold_ = p.eps_angle_threshold_;
            min_points_ = p.min_points_;
            curvature_threshold_ = p.curvature_threshold_;
            cluster_tolerance_ = p.cluster_tolerance_;
            use_normals_from_visible_ = p.use_normals_from_visible_;
        }
    }param_;

  private:
    using GHV<ModelT, SceneT>::mask_;
    using GHV<ModelT, SceneT>::scene_cloud_downsampled_;
    using GHV<ModelT, SceneT>::scene_downsampled_tree_;
    using GHV<ModelT, SceneT>::visible_models_;
    using GHV<ModelT, SceneT>::visible_normal_models_;
    using GHV<ModelT, SceneT>::visible_indices_;
    using GHV<ModelT, SceneT>::complete_models_;
    using GHV<ModelT, SceneT>::normals_set_;
    using GHV<ModelT, SceneT>::requires_normals_;
    using GHV<ModelT, SceneT>::object_ids_;
    using GHV<ModelT, SceneT>::extra_weights_;
    using GHV<ModelT, SceneT>::scene_normals_;
    using GHV<ModelT, SceneT>::recognition_models_;
    using GHV<ModelT, SceneT>::computeRGBHistograms;
    using GHV<ModelT, SceneT>::specifyRGBHistograms;
    using GHV<ModelT, SceneT>::unexplained_by_RM_neighboorhods;
    using GHV<ModelT, SceneT>::explained_by_RM_distance_weighted;
    using GHV<ModelT, SceneT>::explained_by_RM_;
    using GHV<ModelT, SceneT>::complete_cloud_occupancy_by_RM_;
    using GHV<ModelT, SceneT>::octree_scene_downsampled_;
    using GHV<ModelT, SceneT>::cc_;
    using GHV<ModelT, SceneT>::n_cc_;
    using GHV<ModelT, SceneT>::valid_model_;
    using GHV<ModelT, SceneT>::clusters_cloud_rgb_;
    using GHV<ModelT, SceneT>::clusters_cloud_;
    using GHV<ModelT, SceneT>::points_explained_by_rm_;
    using GHV<ModelT, SceneT>::extractEuclideanClustersSmooth;
    using GHV<ModelT, SceneT>::complete_normal_models_;
    using GHV<ModelT, SceneT>::scene_LAB_values_;
    using GHV<ModelT, SceneT>::scene_RGB_values_;
    using GHV<ModelT, SceneT>::scene_GS_values_;
    using GHV<ModelT, SceneT>::computeClutterCueAtOnce;

    //typename pcl::PointCloud<SceneT>::Ptr scene_cloud_downsampled_GO3D_;
    //typename pcl::PointCloud<pcl::Normal>::Ptr scene_normals_go3D_;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > absolute_poses_camera_to_global_;
    std::vector<typename pcl::PointCloud<SceneT>::ConstPtr > occ_clouds_;

    typedef pcl::PointCloud<ModelT> CloudM;
    typedef pcl::PointCloud<SceneT> CloudS;
    typedef typename pcl::traits::fieldList<typename CloudS::PointType>::type FieldListS;
    typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;
    typedef typename pcl::NormalEstimation<SceneT, pcl::Normal> NormalEstimator_;

    public:

      GO3D(const Parameter &p=Parameter()) : GHV<ModelT, SceneT>(p)
      {
         param_ = p;
      }

      bool getInlierOutliersCloud(int hyp_idx, typename pcl::PointCloud<ModelT>::Ptr & cloud);

      //for each cloud, we will need a pose
      //then the models will be checked against all of them
      void setAbsolutePoses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & absolute_poses_camera_to_global)
      {
        absolute_poses_camera_to_global_ = absolute_poses_camera_to_global;
      }

      void
      setOcclusionClouds(std::vector<typename pcl::PointCloud<SceneT>::ConstPtr > & occ_clouds)
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

      virtual
      bool uses_3D() const
      {
          return true;
      }
  };
}

#endif /* HV_GO_3D_H_ */
