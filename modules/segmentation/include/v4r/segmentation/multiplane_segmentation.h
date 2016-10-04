/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/


#pragma once

#include <v4r/common/plane_model.h>
#include <v4r/core/macros.h>

namespace v4r
{
  template<typename PointT>
  class V4R_EXPORTS MultiPlaneSegmentation
  {
    private:
      typedef pcl::PointCloud<PointT> PointTCloud;
      typedef typename pcl::PointCloud<PointT>::Ptr PointTCloudPtr;
      typedef typename pcl::PointCloud<PointT>::ConstPtr PointTCloudConstPtr;
      PointTCloudPtr input_;
      int min_plane_inliers_;
      std::vector<PlaneModel<PointT> > models_;
      float resolution_;
      bool merge_planes_;
      pcl::PointCloud<pcl::Normal>::Ptr normal_cloud_;
      bool normals_set_;

    public:
      MultiPlaneSegmentation()
      {
        min_plane_inliers_ = 1000;
        resolution_ = 0.001f;
        merge_planes_ = false;
        normals_set_ = false;
      }

      void
      setMergePlanes(bool b)
      {
        merge_planes_ = b;
      }

      void setResolution(float f)
      {
        resolution_ = f;
      }

      void setMinPlaneInliers(int t)
      {
        min_plane_inliers_ = t;
      }

      void setInputCloud(const typename pcl::PointCloud<PointT>::Ptr & input)
      {
        input_ = input;
      }

      void segment(bool force_unorganized=false);

      void setNormals(const pcl::PointCloud<pcl::Normal>::Ptr &normal_cloud)
      {
          normal_cloud_ = normal_cloud;
          normals_set_ = true;
      }

      std::vector<PlaneModel<PointT> > getModels() const
      {
        return models_;
      }
  };
}
