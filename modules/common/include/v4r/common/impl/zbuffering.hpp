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

#include <v4r/common/zbuffering.h>
#include <v4r/common/miscellaneous.h>
#include <omp.h>

namespace v4r
{

///////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT>
void
ZBuffering<PointT>::filter (const typename pcl::PointCloud<PointT> & model, typename pcl::PointCloud<PointT> & filtered)
{
    std::vector<int> indices_to_keep;
    filter(model, indices_to_keep);
    pcl::copyPointCloud (model, indices_to_keep, filtered);
}

///////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT>
void
ZBuffering<PointT>::filter (const typename pcl::PointCloud<PointT> & model, std::vector<int> & indices_to_keep)
{
    float cx, cy;
    cx = static_cast<float> (param_.width_) / 2.f - 0.5f;
    cy = static_cast<float> (param_.height_) / 2.f - 0.5f;

    indices_to_keep.resize (model.points.size ());
    size_t kept = 0;
    for (size_t i = 0; i < model.points.size (); i++)
    {
        float x = model.points[i].x;
        float y = model.points[i].y;
        float z = model.points[i].z;
        int u = static_cast<int> (param_.f_ * x / z + cx);
        int v = static_cast<int> (param_.f_ * y / z + cy);

        if (u >= (param_.width_ - param_.u_margin_) || v >= (param_.height_ - param_.v_margin_) || u < param_.u_margin_ || v < param_.v_margin_)
            continue;

        //Check if point depth (distance to camera) is greater than the (u,v) meaning that the point is not visible
        if ((z - param_.inlier_threshold_) > depth_[u * param_.width_ + v] || !pcl_isfinite(depth_[u * param_.height_ + v]))
            continue;

        indices_to_keep[kept] = static_cast<int> (i);
        kept++;
    }

    indices_to_keep.resize (kept);
}

///////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
ZBuffering<PointT>::computeDepthMap (const typename pcl::PointCloud<PointT> & scene)
{
    float cx = static_cast<float> (param_.width_) / 2.f - 0.5f;
    float cy = static_cast<float> (param_.height_) / 2.f - 0.5f;

    //compute the focal length
    if (param_.compute_focal_length_)
    {
        float max_u, max_v, min_u, min_v;
        max_u = max_v = std::numeric_limits<float>::max () * -1;
        min_u = min_v = std::numeric_limits<float>::max ();

        for (size_t i = 0; i < scene.points.size (); i++)
        {
            float b_x = scene.points[i].x / scene.points[i].z;
            if (b_x > max_u)
                max_u = b_x;
            if (b_x < min_u)
                min_u = b_x;

            float b_y = scene.points[i].y / scene.points[i].z;
            if (b_y > max_v)
                max_v = b_y;
            if (b_y < min_v)
                min_v = b_y;
        }

        float maxC = std::max (std::max (std::abs (max_u), std::abs (max_v)), std::max (std::abs (min_u), std::abs (min_v)));
        param_.f_ = (cx) / maxC;
    }

    depth_.resize(param_.width_ * param_.height_, std::numeric_limits<float>::quiet_NaN());
    std::vector<omp_lock_t> depth_locks (param_.width_ * param_.height_);
    for(size_t i=0; i<depth_locks.size(); i++)
        omp_init_lock(&depth_locks[i]);

    std::vector<int> indices2input (param_.width_ * param_.height_, -1);

    #pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<scene.points.size(); i++)
    {
        const PointT &pt = scene.points[i];
        int u = static_cast<int> (param_.f_ * pt.x / pt.z + cx);
        int v = static_cast<int> (param_.f_ * pt.y / pt.z + cy);

        if (u >= param_.width_ - param_.u_margin_ || v >= param_.height_ - param_.v_margin_ || u < param_.u_margin_ || v < param_.v_margin_)
            continue;

        int idx = v * param_.width_ + u;

        omp_set_lock(&depth_locks[idx]);

        if ( (pt.z < depth_[idx]) || !pcl_isfinite(depth_[idx]) ) {
            depth_[idx] = pt.z;
            indices2input [idx] = i;
        }

        omp_unset_lock(&depth_locks[idx]);
    }

    for(size_t i=0; i<depth_locks.size(); i++)
        omp_destroy_lock(&depth_locks[i]);

    if (param_.do_smoothing_)
    {
        //Dilate and smooth the depth map
        std::vector<float> depth_smooth (param_.width_ * param_.height_, std::numeric_limits<float>::quiet_NaN());
        std::vector<int> indices2input_smooth = indices2input;

        for (int u = param_.smoothing_radius_; u < (param_.width_ - param_.smoothing_radius_); u++)
        {
            for (int v = param_.smoothing_radius_; v < (param_.height_ - param_.smoothing_radius_); v++)
            {
                float min = std::numeric_limits<float>::max();
                int min_idx = v * param_.width_ + u;
                for (int j = (u - param_.smoothing_radius_); j <= (u + param_.smoothing_radius_); j++)
                {
                    for (int i = (v - param_.smoothing_radius_); i <= (v + param_.smoothing_radius_); i++)
                    {
                        if( j<0 || i<0 || j>=param_.height_ || i>=param_.width_)    // this shouldn't happen anyway
                            continue;

                        int idx = i * param_.width_ + j;
                        if (pcl_isfinite(depth_[idx]) && (depth_[idx] < min)) {
                            min = depth_[idx];
                            min_idx = idx;
                        }
                    }
                }

                if ( min < std::numeric_limits<float>::max() - 0.001 ) {
                    depth_smooth[v * param_.width_ + u] = min;
                    indices2input_smooth[v * param_.width_ + u] = indices2input[min_idx];
                }
            }
        }
        depth_ = depth_smooth;
        indices2input = indices2input_smooth;
    }


    std::vector<bool> pt_is_visible(scene.points.size(), false);
    for(size_t i=0; i<indices2input.size(); i++)
    {
        int input_idx = indices2input[i];
        if(input_idx>=0)
            pt_is_visible[input_idx] = true;
    }
    kept_indices_ = createIndicesFromMask<int>(pt_is_visible);
}

}
