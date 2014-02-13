/*
 * voxel_dist_transform.h
 *
 *  Created on: Oct 20, 2012
 *      Author: aitor
 */

#ifndef FAAT_PCL_VOXEL_DIST_TRANSFORMSS_H_
#define FAAT_PCL_VOXEL_DIST_TRANSFORMSS_H_

//#include <faat_pcl/3d_rec_framework/defines/faat_3d_rec_framework_defines.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <faat_pcl/utils/EDT/3rdparty/distance_field.h>

namespace faat_pcl
{
  namespace rec_3d_framework
  {
    template<typename PointT>
    class Voxel
    {
    public:
      bool label_;
      Voxel<PointT> * neigh_;
      float dist_;
      PointT avg_;
      int n_;
      int idx_;
    };

    template<typename PointT>
      class VoxelGridDistanceTransform
      {
        typename std::vector<Voxel<PointT> > grid_;
        float resolution_;
        typename pcl::PointCloud<PointT>::ConstPtr cloud_;
        bool initialized_;
        int gs_x_, gs_y_, gs_z_;
        PointT min_pt_all, max_pt_all;
        typename pcl::PointCloud<PointT>::Ptr voxelized_cloud_;

        inline void
        visualizeGrid ();

      public:
        VoxelGridDistanceTransform (float res = 0.005f)
        {
          resolution_ = res;
          initialized_ = false;
          std::cout << "Resolution:" << res << std::endl;
        }

        ~VoxelGridDistanceTransform ()
        {
        }

        void
        setInputCloud (typename pcl::PointCloud<PointT>::ConstPtr & cloud)
        {
          cloud_ = cloud;
        }

        void
        getVoxelizedCloud (typename pcl::PointCloud<PointT>::Ptr & cloud)
        {
          cloud = voxelized_cloud_;
        }

        bool
        isInitialized ()
        {
          return initialized_;
        }

        void
        compute ();

        void
        getCorrespondence (const PointT & p, int * idx, float * dist, float sigma, float * color_distance);
      };
  }
}

#endif /* VOXEL_DIST_TRANSFORM_H_ */
