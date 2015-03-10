/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012-, Thomas Mörwald
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
 *   * Neither the name of the copyright holder(s) nor the names of its
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

#ifndef NURBS_FITTING_SURFACE_DEPTH_PCA_H
#define NURBS_FITTING_SURFACE_DEPTH_PCA_H

#include "fitting_surface_depth.h"
#include <pcl/pcl_base.h>
#include <pcl/common/pca.h>

namespace pcl{
namespace on_nurbs{


template <class PointT>
class FittingSurfaceDepthPCA : public FittingSurfaceDepth, public PCA<PointT>
{
public:
  typedef PCLBase <PointT> Base;
  typedef typename Base::PointCloud PointCloud;
  typedef typename Base::PointCloudPtr PointCloudPtr;
  typedef typename Base::PointCloudConstPtr PointCloudConstPtr;
  typedef typename Base::PointIndicesPtr PointIndicesPtr;
  typedef typename Base::PointIndicesConstPtr PointIndicesConstPtr;

  using PCA<PointT>::input_;
  using PCA<PointT>::indices_;
  using Base::initCompute;
  using FittingSurfaceDepth::operator delete;
  using Base::operator delete;
  using FittingSurfaceDepth::getSurface;

  FittingSurfaceDepthPCA() : order_(3), cps_x_(3), cps_y_(3), nurbs_done_(false){}

  inline void
  setInputCloud (const PointCloudConstPtr &cloud)
  {
    PCA<PointT>::setInputCloud (cloud);
    nurbs_done_ = false;
  }

  inline void
  setIndices (const IndicesPtr &indices)
  {
    PCA<PointT>::setInputCloud(input_); // hack to set PCA<PointT>::compute_done_ = false
    PCA<PointT>::setIndices(indices);
    nurbs_done_ = false;
  }

  void
  setIndices (const PointIndicesConstPtr &indices)
  {
    PCA<PointT>::setInputCloud(input_); // hack to set PCA<PointT>::compute_done_ = false
    PCA<PointT>::setIndices(indices);
    nurbs_done_ = false;
  }

  inline void
  setParameter(int order, int cps_x, int cps_y)
  {
    order_ = order;
    cps_x_ = cps_x;
    cps_y_ = cps_y;
    nurbs_done_ = false;
  }

  inline ON_NurbsSurface&
  getSurface ()
  {
    if (!nurbs_done_)
      initCompute();
    if (!nurbs_done_)
      PCL_THROW_EXCEPTION (InitFailedException,
                           "[pcl::on_nurbs::FittingSurfaceDepthPCA::getNurbsSurface] FittingSurfaceDepthPCA ON_NurbsSurface failed");
    return m_nurbs;
  }

  ON_NurbsSurface
  getSurface3D();

private:
  inline bool
  initCompute ();

  pcl::PCA<PointT> pca;
  int order_;
  int cps_x_;
  int cps_y_;
  bool nurbs_done_;
};

} // namespace on_nurbs
} // namespace pcl

#endif // FITTING_SURFACE_DEPTH_PCA_H
