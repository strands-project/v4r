/*
 * lm_icp.h
 *
 *  Created on: Jun 12, 2013
 *      Author: aitor
 */

#ifndef FAAT_LM_ICP_H_
#define FAAT_LM_ICP_H_

#include <levmar-2.6/levmar.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree.h>
#include <faat_pcl/utils/EDT/3rdparty/propagation_distance_field.h>
#include <faat_pcl/utils/voxel_dist_transform.h>

namespace faat_pcl
{
  namespace registration
  {
    class NonLinearICP
    {
      Eigen::Matrix4d final_transform_;

    public:
      float dt_vx_size_;

      typedef pcl::PointXYZ PointT;
      typedef pcl::PointCloud<PointT>::Ptr PointTCloudPtr;
      typedef pcl::PointCloud<PointT>::ConstPtr PointTCloudConstPtr;

      pcl::PointCloud<pcl::PointXYZ>::Ptr input_;
      pcl::PointCloud<pcl::PointXYZ>::Ptr input_transformed_;
      pcl::PointCloud<pcl::PointXYZ>::Ptr target_;
      pcl::PointCloud<pcl::Normal>::Ptr target_normals_;
      float inlier_threshold_;
      int iterations_;
      bool use_octree_;

      //target distance transform
      boost::shared_ptr<distance_field::PropagationDistanceField<PointT> > dist_trans_;
      //boost::shared_ptr<faat_pcl::rec_3d_framework::VoxelGridDistanceTransform<PointT> > dist_trans_;
      boost::shared_ptr<pcl::octree::OctreePointCloudSearch<PointT> > octree_;

      NonLinearICP(float vx=0.01f)
      {
        dt_vx_size_ = vx;
        inlier_threshold_ = 0.01f;
        iterations_ = 50;
        use_octree_ = false;
      }

      void setUseOctree(bool b)
      {
          use_octree_ = b;
      }

      void setIterations(int i)
      {
          iterations_ = i;
      }

      void setInlierThreshold(float f)
      {
          inlier_threshold_ = f;
      }

      void getFinalTransformation(Eigen::Matrix4d & trans)
      {
        trans = final_transform_;
      }

      void setTargetNormals(pcl::PointCloud<pcl::Normal>::Ptr & cloud)
      {
        target_normals_ = cloud;
      }

      void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud)
      {
        input_ = cloud;
      }

      void setTargetCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud)
      {
        target_ = cloud;
      }

      void compute();
    };
  }
}

#endif /* LM_ICP_H_ */
