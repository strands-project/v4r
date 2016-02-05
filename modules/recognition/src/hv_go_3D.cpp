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

    const HVRecognitionModel<ModelT> &rm = *recognition_models_[hyp_idx];

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.points.resize( rm.visible_cloud_->points.size() );

    for(size_t i=0; i < cloud.points.size(); i++)
    {
        pcl::PointXYZRGB &pt = cloud.points[i];
        pt.g = 255;
        pt.r = pt.b = 0;

        const ModelT &m_pt = rm.visible_cloud_->points[i];
        pt.x = m_pt.x;
        pt.y = m_pt.y;
        pt.z = m_pt.z;
    }

    for(size_t i=0; i < rm.outlier_indices_.size(); i++)
    {
        pcl::PointXYZRGB &pt = cloud.points[ rm.outlier_indices_[i] ];
        pt.r = 255;
        pt.g = pt.b = 0;
    }

    return cloud;
}

template<typename ModelT, typename SceneT>
void
GO3D<ModelT, SceneT>::addModels  (std::vector<typename pcl::PointCloud<ModelT>::ConstPtr> & models, std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> &model_normals)
{
    size_t existing_models = recognition_models_.size();
    recognition_models_.resize( existing_models + models.size() );

    if(param_.icp_iterations_) {
        refined_model_transforms_.clear();
        refined_model_transforms_.resize( models.size() );
    }

    //pcl::visualization::PCLVisualizer vis ("all visible model points");

    #pragma omp parallel for schedule (dynamic)
    for(size_t i=0; i<models.size(); i++)
    {
        recognition_models_[existing_models + i].reset(new GHVRecognitionModel<ModelT>);
        HVRecognitionModel<ModelT> &rm = *recognition_models_[existing_models + i];

        rm.is_planar_ = false;
        rm.complete_cloud_.reset(new pcl::PointCloud<ModelT>(*models[i]));
        rm.visible_cloud_.reset( new pcl::PointCloud<ModelT> );
        bool redo;

        do
        {
            redo = false;
            if (!param_.do_occlusion_reasoning_)   // just copy complete models
                *rm.visible_cloud_ = *models[i];
            else
            {
                rm.visible_indices_.clear();
                std::vector<bool> model_point_is_visible (models[i]->points.size(), false);

                for(size_t k=0; k < occ_clouds_.size(); k++)
                {
                    if (model_is_present_in_view_.size() == models.size() &&
                            model_is_present_in_view_[i].size() == occ_clouds_.size() &&
                            !model_is_present_in_view_[i][k])   // if model is not present in view k (scene not static), it does not make sense to check for occlusion
                        continue;

                    //transform model to camera coordinate
                    const Eigen::Matrix4f &trans = absolute_camera_poses_[k].inverse();
                    typename pcl::PointCloud<ModelT> model_in_view_coordinates;
                    pcl::transformPointCloud(*models[i], model_in_view_coordinates, trans);

                    std::vector<int> self_occlusion_indices;
                    typename ZBuffering<ModelT>::Parameter zbuffParam;
                    zbuffParam.inlier_threshold_ = param_.zbuffer_self_occlusion_resolution_;
                    zbuffParam.f_ = param_.focal_length_;
                    zbuffParam.width_ = 640;
                    zbuffParam.height_ = 480;
                    zbuffParam.u_margin_ = 5;
                    zbuffParam.v_margin_ = 5;
                    zbuffParam.compute_focal_length_ = false;
                    zbuffParam.do_smoothing_ = true;
                    zbuffParam.smoothing_radius_ = 2;
                    ZBuffering<ModelT> zbuffer_self_occlusion (zbuffParam);
                    zbuffer_self_occlusion.computeDepthMap (model_in_view_coordinates);
                    zbuffer_self_occlusion.getKeptIndices(self_occlusion_indices);

                    std::vector<bool> pt_is_occluded = computeOccludedPoints(*occ_clouds_[k], model_in_view_coordinates, param_.focal_length_, param_.occlusion_thres_, true);
                    std::vector<bool> model_point_is_visible_in_occ_k(models[i]->points.size(), false);


                    for(size_t idx=0; idx<self_occlusion_indices.size(); idx++) {
                        size_t midx = self_occlusion_indices[idx];
                        if ( !pt_is_occluded[midx]) {
                            model_point_is_visible[midx] = true;
                            model_point_is_visible_in_occ_k[midx] = true;
                        }
                    }


//                    pcl::visualization::PCLVisualizer vis;
//                    int v1, v2, v3;
//                    vis.createViewPort(0,0,0.5,0.5,v1);
//                    vis.createViewPort(0.5,0,1,0.5,v2);
//                    vis.createViewPort(0.5,0.5,1,1,v3);
//                    vis.addPointCloud(occ_clouds_[k],"occ_cloud",v1);
//                    vis.addPointCloud(model_in_view_coordinates.makeShared(),"model",v1);
//                    pcl::PointCloud<ModelT> visible_cloud, visible_cloud_so;
//                    pcl::copyPointCloud(model_in_view_coordinates, self_occlusion_indices, visible_cloud_so);
//                    vis.addPointCloud(visible_cloud_so.makeShared(),"visible_model",v2);
//                    pcl::copyPointCloud(model_in_view_coordinates, model_point_is_visible_in_occ_k, visible_cloud);
//                    vis.addPointCloud(visible_cloud.makeShared(),"visible_model_occ",v3);
//                    vis.spin();
                }
                rm.visible_indices_ = createIndicesFromMask<int>( model_point_is_visible );
                pcl::copyPointCloud(*rm.complete_cloud_, rm.visible_indices_, *rm.visible_cloud_);
            }

            if(param_.icp_iterations_ && !refined_model_transforms_[i])
            {
                refined_model_transforms_[i].reset(new Eigen::Matrix4f (poseRefinement(rm)));

                pcl::PointCloud<ModelT> aligned_cloud;
                pcl::transformPointCloud(*rm.visible_cloud_, aligned_cloud, *refined_model_transforms_[i]);
                *rm.visible_cloud_  = aligned_cloud;
                pcl::transformPointCloud(*rm.complete_cloud_, aligned_cloud, *refined_model_transforms_[i]);
                *rm.complete_cloud_ = aligned_cloud;
                redo = true;
            }
        }
        while(redo);

        // copy normals if provided
        if ( i<model_normals.size() )
        {
            rm.complete_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal> (*model_normals[i]) );
            rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal> ());
            pcl::copyPointCloud(*rm.complete_cloud_normals_, rm.visible_indices_, *rm.visible_cloud_normals_);

            if (refined_model_transforms_[i])
                v4r::transformNormals(*model_normals[i], *rm.complete_cloud_normals_, *refined_model_transforms_[i]);

            if (!param_.do_occlusion_reasoning_)   // just copy complete models
                rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal> (*rm.complete_cloud_normals_) );
            else
            {
                rm.visible_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal>);
                pcl::copyPointCloud(*rm.complete_cloud_normals_, rm.visible_indices_, *rm.visible_cloud_normals_);
            }
        }
    }
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
