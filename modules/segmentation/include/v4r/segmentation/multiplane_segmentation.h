/*
 * multiplane_segmentation.h
 *
 *  Created on: Sep 25, 2013
 *      Author: aitor
 */

#ifndef FAAT_PCL_3D_REC_FRAMEWORK_MULTIPLANE_SEGMENTATION_H_
#define FAAT_PCL_3D_REC_FRAMEWORK_MULTIPLANE_SEGMENTATION_H_

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

#endif /* FAAT_PCL_3D_REC_FRAMEWORK_MULTIPLANE_SEGMENTATION_H_ */
