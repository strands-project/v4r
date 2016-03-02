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
#include <pcl/visualization/pcl_visualizer.h>

namespace v4r
{
  template<typename ModelT, typename SceneT>
  class V4R_EXPORTS GO3D : public GHV<ModelT, SceneT>
  {

  public:
    class V4R_EXPORTS Parameter : public GHV<ModelT, SceneT>::Parameter
    {
    public:
         using GHV<ModelT, SceneT>::Parameter::radius_neighborhood_clutter_;
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
         using GHV<ModelT, SceneT>::Parameter::add_planes_;
         using GHV<ModelT, SceneT>::Parameter::plane_method_;
         using GHV<ModelT, SceneT>::Parameter::focal_length_;
         using GHV<ModelT, SceneT>::Parameter::visualize_go_cues_;
         using GHV<ModelT, SceneT>::Parameter::color_space_;
         using GHV<ModelT, SceneT>::Parameter::do_occlusion_reasoning_;

         bool visualize_cues_;

        Parameter(
                    bool visualize_cues = false
                 )
            : visualize_cues_( visualize_cues )
        {}
    }param_;

  private:
    using GHV<ModelT, SceneT>::mask_;
    using GHV<ModelT, SceneT>::scene_cloud_;
    using GHV<ModelT, SceneT>::scene_cloud_downsampled_;
    using GHV<ModelT, SceneT>::scene_downsampled_tree_;
    using GHV<ModelT, SceneT>::model_point_is_visible_;
    using GHV<ModelT, SceneT>::normals_set_;
    using GHV<ModelT, SceneT>::requires_normals_;
    using GHV<ModelT, SceneT>::scene_normals_;
    using GHV<ModelT, SceneT>::recognition_models_;
    using GHV<ModelT, SceneT>::unexplained_by_RM_neighboorhods;
    using GHV<ModelT, SceneT>::explained_by_RM_distance_weighted;
    using GHV<ModelT, SceneT>::explained_by_RM_;
    using GHV<ModelT, SceneT>::complete_cloud_occupancy_by_RM_;
    using GHV<ModelT, SceneT>::octree_scene_downsampled_;
    using GHV<ModelT, SceneT>::cc_;
    using GHV<ModelT, SceneT>::n_cc_;
    using GHV<ModelT, SceneT>::clusters_cloud_rgb_;
    using GHV<ModelT, SceneT>::clusters_cloud_;
    using GHV<ModelT, SceneT>::points_explained_by_rm_;

    //typename pcl::PointCloud<SceneT>::Ptr scene_cloud_downsampled_GO3D_;
    //typename pcl::PointCloud<pcl::Normal>::Ptr scene_normals_go3D_;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > absolute_camera_poses_;
    std::vector<typename pcl::PointCloud<SceneT>::ConstPtr > occ_clouds_;
    std::vector<std::vector<bool> > model_is_present_in_view_; /// \brief for each model this variable stores information in which view it is present (used to check for visible model points - default all true = static scene)

    mutable pcl::visualization::PCLVisualizer::Ptr vis_;
    mutable int vp1_, vp2_;

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

      /**
       * @brief getInlierOutliersCloud
       * @param hyp_idx
       * @return colored point cloud with green points representing inliers and red points outliers
       */
      pcl::PointCloud<pcl::PointXYZRGB> getInlierOutliersCloud(int hyp_idx) const;


      /**
       * @brief for each cloud, we will need a pose. Then the models will be checked against all of them
       */
      void
      setAbsolutePoses(const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & absolute_camera_poses)
      {
        absolute_camera_poses_ = absolute_camera_poses;
      }


      void
      setOcclusionClouds(const std::vector<typename pcl::PointCloud<SceneT>::ConstPtr > & occ_clouds)
      {
        occ_clouds_ = occ_clouds;
      }


      /**
       * @brief addModels Adds object hypotheses
       * @param models set of object hypotheses (setAbsolutePoses must be called in advances!)
       * @param normals of the models [optional]
       */
      void
      addModels (std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models,
                 std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr > &model_normals = std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr >());
      /**
       * @brief for each model this variable stores information in which view it is present
       * @param presence in model and view
       */
      void
      setVisibleCloudsForModels(const std::vector<std::vector<bool> > &model_is_present_in_view)
      {
          model_is_present_in_view_ = model_is_present_in_view;
      }

      virtual
      bool uses_3D() const
      {
          return true;
      }

      void visualize() const;
  };
}

#endif /* HV_GO_3D_H_ */
