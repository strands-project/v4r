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
#include <v4r/common/impl/zbuffering.hpp>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/binary_algorithms.h>
#include <v4r/recognition/hv_go_3D.h>
#include <pcl/features/normal_3d_omp.h>
#include <functional>
#include <numeric>
#include <glog/logging.h>

namespace v4r {

template<typename ModelT, typename SceneT>
pcl::PointCloud<pcl::PointXYZRGB>
GO3D<ModelT, SceneT>::getInlierOutliersCloud(int hyp_idx) const
{
    CHECK(hyp_idx >= 0 && hyp_idx < recognition_models_.size() ) << "Hypothesis ID " << hyp_idx << " does not exist. Please choose a value between 0 and " << recognition_models_.size() - 1 << ".";

    const boost::shared_ptr<GHVRecognitionModel<ModelT> > &recog_model = recognition_models_[hyp_idx];
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.points.resize( recog_model->visible_cloud_->points.size() );

    for(size_t i=0; i < cloud.points.size(); i++)
    {
        pcl::PointXYZRGB &pt = cloud.points[i];
        pt.g = 255;
        pt.r = pt.b = 0;

        const ModelT &m_pt = recog_model->visible_cloud_->points[i];
        pt.x = m_pt.x;
        pt.y = m_pt.y;
        pt.z = m_pt.z;
    }

    for(size_t i=0; i < recog_model->outlier_indices_.size(); i++)
    {
        pcl::PointXYZRGB &pt = cloud.points[ recog_model->outlier_indices_[i] ];
        pt.r = 255;
        pt.g = pt.b = 0;
    }

    return cloud;
}

template<typename ModelT, typename SceneT>
void
GO3D<ModelT, SceneT>::addModels (std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models, bool occlusion_reasoning)
{
    mask_.clear();
    complete_models_ = models;

    if (!occlusion_reasoning) {   // just copy complete models
        visible_models_.resize(models.size());
        for(size_t i=0; i<models.size(); i++) {
            if(!visible_models_[i])
                visible_models_[i].reset(new pcl::PointCloud<ModelT>);
            pcl::copyPointCloud(*models[i], *visible_models_[i]);
        }
    }
    else
    {
        visible_indices_.clear();
        model_point_is_visible_.resize(models.size());
        visible_indices_.resize(models.size());
        visible_models_.resize(models.size());

        if(normals_set_ && requires_normals_)
            visible_normal_models_.resize(models.size());

        //compute visible model points based on scene - a point is occluded if it is occluded in all views
        #pragma omp parallel for schedule(dynamic)
        for(size_t m=0; m<models.size(); m++)
        {
            model_point_is_visible_[m].clear(); // DO NOT DELETE THIS! Only doing resize will only re-initialize indices greater than the previous size of this vector!!
            model_point_is_visible_[m].resize(models[m]->points.size(), false);
            for(size_t k=0; k < occ_clouds_.size(); k++)
            {
                const Eigen::Matrix4f &trans = absolute_camera_poses_[k].inverse();

                if (model_is_present_in_view_.size() == models.size() &&
                        model_is_present_in_view_[m].size() == occ_clouds_.size() &&
                        !model_is_present_in_view_[m][k])   // if model is not present in view k (scene not static), it does not make sense to check for occlusion
                    continue;

                //transform model to camera coordinate
                typename pcl::PointCloud<ModelT> model_in_view_coordinates;
                pcl::transformPointCloud(*models[m], model_in_view_coordinates, trans);

                std::vector<bool> pt_is_occluded = computeOccludedPoints(*occ_clouds_[k], model_in_view_coordinates, param_.focal_length_, param_.occlusion_thres_, true);
                std::vector<bool> model_point_is_visible_in_occ_k(models[m]->points.size(), false);

                for(size_t idx=0; idx<model_point_is_visible_[m].size(); idx++) {
                    if ( !pt_is_occluded[idx] ) {
                        model_point_is_visible_[m][idx] = true;
                        model_point_is_visible_in_occ_k[idx] = true;
                    }
                }
            }

            visible_indices_[m] = createIndicesFromMask<int>( model_point_is_visible_[m] );

            if(!visible_models_[m])
                visible_models_[m].reset(new pcl::PointCloud<ModelT> ());
            pcl::copyPointCloud(*models[m], visible_indices_[m], *visible_models_[m]);

            if(normals_set_ && requires_normals_) {
                if(!visible_normal_models_[m])
                    visible_normal_models_[m].reset(new pcl::PointCloud<pcl::Normal> ());
                pcl::copyPointCloud(*complete_normal_models_[m], visible_indices_[m], *visible_normal_models_[m]);
            }
        }

//        pcl::visualization::PCLVisualizer vis ("all visible model points");
//        for(size_t m=0; m<models.size(); m++)
//        {
//            std::stringstream cloud_id; cloud_id << "cloud_" << m;
//            vis.addPointCloud(visible_models_[m], cloud_id.str());
//        }
//        vis.spin();
    }

    normals_set_ = false;
}


template<typename ModelT, typename SceneT>
void
GO3D<ModelT, SceneT>::visualize () const
{
    if(!vis_) {
        vis_.reset(new pcl::visualization::PCLVisualizer(" GO 3D visualization "));
        vis_->createViewPort(0, 0, 0.5, 1, vp1_);
        vis_->createViewPort(0.5, 0, 1, 1, vp2_);
    }

    vis_->removeAllPointClouds(vp1_);
    vis_->addPointCloud(scene_cloud_, "scene", vp1_);

    for(size_t i=0; i<recognition_models_.size(); i++) {
        pcl::PointCloud<pcl::PointXYZRGB> inl_outl_cloud = getInlierOutliersCloud(i);
        vis_->removeAllPointClouds(vp2_);
        vis_->removePointCloud("inl_outl_cloud_vp1", vp1_);
        vis_->addPointCloud(inl_outl_cloud.makeShared(), "inl_outl_cloud_vp1", vp1_);
        vis_->addPointCloud(inl_outl_cloud.makeShared(), "inl_outl_cloud_vp2", vp2_);
        vis_->spin();
    }
}


template class V4R_EXPORTS GO3D<pcl::PointXYZRGB,pcl::PointXYZRGB>;
}
