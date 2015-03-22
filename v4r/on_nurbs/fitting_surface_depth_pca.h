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
class FittingSurfaceDepthPCA : public FittingSurfaceDepth
{
public:
  typedef PCLBase <PointT> Base;
  typedef typename Base::PointCloud PointCloud;
  typedef typename Base::PointCloudPtr PointCloudPtr;
  typedef typename Base::PointCloudConstPtr PointCloudConstPtr;
  typedef typename Base::PointIndicesPtr PointIndicesPtr;
  typedef typename Base::PointIndicesConstPtr PointIndicesConstPtr;

  using FittingSurfaceDepth::getSurface;

  FittingSurfaceDepthPCA() : order_(3), cps_x_(3), cps_y_(3), nurbs_done_(false), proj_done_(false){}

  void setInputCloud(const PointCloudConstPtr &cloud);

  virtual void setIndices(const IndicesPtr &indices);

  virtual void setIndices(const IndicesConstPtr &indices);

  virtual void setIndices(const PointIndicesConstPtr &indices);

  virtual void setIndices(size_t row_start, size_t col_start, size_t nb_rows, size_t nb_cols);

  void setParameter(int order, int cps_x, int cps_y);

  void flipEigenSpace();

  void project (const PointT& input, PointT& projection);

  void reconstruct(const PointT& projection, PointT& input);

  Eigen::Vector4f getMean();

  Eigen::Matrix3f getEigenVectors();

  ON_NurbsSurface& getSurface();

  ON_NurbsSurface getSurface3D();

  Eigen::VectorXd getError ();

  const ROI& getROI();

  const PointCloud& getProjectedCloud();

private:
  bool computePCA();
  bool computeProjection();
  bool computeNurbsSurface();

  Eigen::MatrixXd points_;
  pcl::PCA<PointT> pca;
  Eigen::Matrix3f eigenvectors_;
  Eigen::Vector4f mean_;
  PointCloud cloud_pc_;
  ROI roi_pc_;
  int order_;
  int cps_x_;
  int cps_y_;
  bool pca_done_;
  bool nurbs_done_;
  bool proj_done_;
};

} // namespace on_nurbs
} // namespace pcl

#endif // FITTING_SURFACE_DEPTH_PCA_H
