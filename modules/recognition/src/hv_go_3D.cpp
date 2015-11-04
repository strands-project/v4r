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

#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/impl/instantiate.hpp>
#include <v4r/common/impl/occlusion_reasoning.hpp>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/binary_algorithms.h>
#include <v4r/recognition/hv_go_3D.h>
#include <pcl/features/normal_3d_omp.h>
#include <functional>
#include <numeric>

namespace v4r {

template<typename ModelT, typename SceneT>
bool
GO3D<ModelT, SceneT>::getInlierOutliersCloud(int hyp_idx, typename pcl::PointCloud<ModelT>::Ptr & cloud)
{
  if(hyp_idx < 0 || hyp_idx > (recognition_models_.size() - 1))
    return false;

  boost::shared_ptr<GHVRecognitionModel<ModelT> > recog_model = recognition_models_[hyp_idx];
  cloud.reset(new pcl::PointCloud<ModelT>(*recog_model->cloud_));


  for(size_t i=0; i < cloud->points.size(); i++)
  {
    cloud->points[i].r = 0;
    cloud->points[i].g = 255;
    cloud->points[i].b = 0;
  }

  for(size_t i=0; i < recog_model->outlier_indices_.size(); i++)
  {
    cloud->points[recog_model->outlier_indices_[i]].r = 255;
    cloud->points[recog_model->outlier_indices_[i]].g = 0;
    cloud->points[recog_model->outlier_indices_[i]].b = 0;
  }

  return true;
}

template<typename ModelT, typename SceneT>
void
GO3D<ModelT, SceneT>::addModels (std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models, bool occlusion_reasoning)
{
  mask_.clear();

  if (!occlusion_reasoning)
    visible_models_ = models;
  else
  {
    visible_indices_.resize(models.size());
    model_point_is_visible_.clear();
    model_point_is_visible_.resize(models.size());

    for (size_t i = 0; i < models.size (); i++)
        model_point_is_visible_[i].resize(models[i]->points.size(), false);

      //a point is occluded if it is occluded in all views

      //scene-occlusions
      for(size_t k=0; k < occ_clouds_.size(); k++)
      {
          const Eigen::Matrix4f trans =  absolute_poses_camera_to_global_[k].inverse();

          for(size_t m=0; m<models.size(); m++)
          {
            //transform model to camera coordinate
            typename pcl::PointCloud<ModelT> model_in_view_coordinates;
            pcl::transformPointCloud(*models[m], model_in_view_coordinates, trans);

            std::vector<bool> pt_is_occluded = occlusion_reasoning::computeOccludedPoints(*occ_clouds_[k], model_in_view_coordinates, param_.focal_length_, param_.occlusion_thres_, true);
            std::vector<bool> model_point_is_visible_in_occ_k(models[m]->points.size(), false);

            for(size_t idx=0; idx<model_point_is_visible_[m].size(); idx++) {
                if ( !pt_is_occluded[idx] ) {
                    model_point_is_visible_[m][idx] = true;
                    model_point_is_visible_in_occ_k[idx] = true;
                }
            }
          }
      }

    for (size_t i = 0; i < models.size (); i++) {

      visible_indices_[i] = createIndicesFromMask<int>( model_point_is_visible_[i] );

      typename pcl::PointCloud<ModelT>::Ptr filtered (new pcl::PointCloud<ModelT> ());
      pcl::copyPointCloud(*models[i], model_point_is_visible_[i], *filtered);

      if(normals_set_ && requires_normals_) {
        pcl::PointCloud<pcl::Normal>::Ptr filtered_normals (new pcl::PointCloud<pcl::Normal> ());
        pcl::copyPointCloud(*complete_normal_models_[i], model_point_is_visible_[i], *filtered_normals);
        visible_normal_models_.push_back(filtered_normals);
      }

//      pcl::visualization::PointCloudColorHandlerRGBField<ModelT> handler (filtered);
//      vis.addPointCloud(filtered, handler, "model");
//      vis.removeAllPointClouds();

      visible_models_.push_back (filtered);
    }

    complete_models_ = models;
  }

  normals_set_ = false;
}

template class V4R_EXPORTS GO3D<pcl::PointXYZRGB,pcl::PointXYZRGB>;

}
