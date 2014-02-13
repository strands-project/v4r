/*
 * lm_icp.h
 *
 *  Created on: Jun 12, 2013
 *      Author: aitor
 */

#ifndef FAAT_MV_LM_ICP_H_
#define FAAT_MV_LM_ICP_H_

#include <levmar-2.6/levmar.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree.h>
#include <faat_pcl/utils/EDT/3rdparty/propagation_distance_field.h>
#include <faat_pcl/utils/voxel_dist_transform.h>
//#include "pcl/visualization/pcl_visualizer.h"

namespace faat_pcl
{
  namespace registration
  {
    template<typename PointT>
    class MVNonLinearICP
    {

      int max_iterations_;
      bool sparse_solver_;
      std::vector<Eigen::Matrix4f> transformations_;
    public:
      bool vis_intermediate_;
      float dt_vx_size_;
      float inlier_threshold_;
      //typedef pcl::PointXYZ PointT;
      typedef typename pcl::PointCloud<PointT>::Ptr PointTCloudPtr;
      typedef typename pcl::PointCloud<PointT>::ConstPtr PointTCloudConstPtr;

      std::vector<PointTCloudPtr> input_clouds_;
      std::vector<pcl::PointCloud<pcl::Normal>::Ptr> input_normals_;
      std::vector< std::vector<bool> > adjacency_matrix_;
      std::vector< std::pair<int, int> > S_;
      std::vector< std::vector<float> > * weights_;
      bool weights_available_;
      //target distance transform
      std::vector<boost::shared_ptr<distance_field::PropagationDistanceField<PointT> > > dist_transforms_;
      std::vector<boost::shared_ptr<pcl::octree::OctreePointCloudSearch<PointT> > > octrees_;

      pcl::visualization::PCLVisualizer * vis_;
      double last_error;
      bool normals_available_;
      float max_correspondence_distance_;
      float min_dot_;

      MVNonLinearICP(float vx=0.01f)
      {
        //vis_ = new pcl::visualization::PCLVisualizer();
        dt_vx_size_ = vx;
        last_error = std::numeric_limits<double>::infinity();
        normals_available_ = false;
        max_correspondence_distance_ = std::numeric_limits<float>::infinity();
        max_iterations_ = 20;
        vis_intermediate_ = true;
        sparse_solver_ = true;
        weights_available_ = false;
        min_dot_ = 0.9f;
      }

      void setMinDot(float d)
      {
        min_dot_ = d;
      }

      void setPointsWeight(std::vector< std::vector<float> > & weights)
      {
        weights_ = &(weights);
        weights_available_ = true;
      }

      void setInlierThreshold(float f)
      {
        inlier_threshold_ = f;
      }

      void setSparseSolver(bool b)
      {
        sparse_solver_ = b;
      }

      void setVisIntermediate(bool b)
      {
        vis_intermediate_ = b;
      }

      void setMaxIterations(int n)
      {
        max_iterations_ = n;
      }

      void setMaxCorrespondenceDistance(float f)
      {
        max_correspondence_distance_ = f;
      }

      void setAdjacencyMatrix(std::vector< std::vector<bool> > & mat)
      {
        adjacency_matrix_ = mat;
      }

      void setClouds(std::vector<typename pcl::PointCloud<PointT>::Ptr> & clouds)
      {
        input_clouds_ = clouds;
      }

      void setInputNormals(std::vector<pcl::PointCloud<pcl::Normal>::Ptr> & clouds)
      {
        input_normals_ = clouds;
        normals_available_ = true;
      }

      void getTransformation(std::vector<Eigen::Matrix4f> & transforms)
      {
        transforms = transformations_;
      }

      void compute();
    };
  }
}

#endif /* LM_ICP_H_ */
