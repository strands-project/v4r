/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc
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
 *
 * $Id: icp.hpp 8409 2013-01-03 10:02:16Z aaldoma $
 *
 */

#ifndef FAAT_PCL_REGISTRATION_IMPL_ICP_HPP_
#define FAAT_PCL_REGISTRATION_IMPL_ICP_HPP_

#include <pcl/registration/boost.h>
#include <pcl/correspondence.h>
#include <faat_pcl/recognition/cg/graph_geometric_consistency.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <algorithm>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/octree/octree.h>
#include <faat_pcl/utils/EDT/3rdparty/propagation_distance_field.h>
#include <faat_pcl/registration/visibility_reasoning.h>
#include <faat_pcl/registration/uniform_sampling.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>

///////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget, typename Scalar>
  void
  faat_pcl::IterativeClosestPointWithGC<PointSource, PointTarget, Scalar>::transformCloud (const PointCloudSource &input, PointCloudSource &output,
                                                                                           const Matrix4 &transform)
  {
    Eigen::Vector4f pt (0.0f, 0.0f, 0.0f, 1.0f), pt_t;
    Eigen::Matrix4f tr = transform.template cast<float> ();

    // XYZ is ALWAYS present due to the templatization, so we only have to check for normals
    if (source_has_normals_)
    {
      Eigen::Vector3f nt, nt_t;
      Eigen::Matrix3f rot = tr.block<3, 3> (0, 0);

      for (size_t i = 0; i < input.size (); ++i)
      {
        const uint8_t* data_in = reinterpret_cast<const uint8_t*> (&input[i]);
        uint8_t* data_out = reinterpret_cast<uint8_t*> (&output[i]);
        memcpy (&pt[0], data_in + x_idx_offset_, sizeof(float));
        memcpy (&pt[1], data_in + y_idx_offset_, sizeof(float));
        memcpy (&pt[2], data_in + z_idx_offset_, sizeof(float));

        if (!pcl_isfinite (pt[0]) || !pcl_isfinite (pt[1]) || !pcl_isfinite (pt[2]))
          continue;

        pt_t = tr * pt;

        memcpy (data_out + x_idx_offset_, &pt_t[0], sizeof(float));
        memcpy (data_out + y_idx_offset_, &pt_t[1], sizeof(float));
        memcpy (data_out + z_idx_offset_, &pt_t[2], sizeof(float));

        memcpy (&nt[0], data_in + nx_idx_offset_, sizeof(float));
        memcpy (&nt[1], data_in + ny_idx_offset_, sizeof(float));
        memcpy (&nt[2], data_in + nz_idx_offset_, sizeof(float));

        if (!pcl_isfinite (nt[0]) || !pcl_isfinite (nt[1]) || !pcl_isfinite (nt[2]))
          continue;

        nt_t = rot * nt;

        memcpy (data_out + nx_idx_offset_, &nt_t[0], sizeof(float));
        memcpy (data_out + ny_idx_offset_, &nt_t[1], sizeof(float));
        memcpy (data_out + nz_idx_offset_, &nt_t[2], sizeof(float));
      }
    }
    else
    {
      std::cout << "Source does not have normals..." << std::endl;
      for (size_t i = 0; i < input.size (); ++i)
      {
        const uint8_t* data_in = reinterpret_cast<const uint8_t*> (&input[i]);
        uint8_t* data_out = reinterpret_cast<uint8_t*> (&output[i]);
        memcpy (&pt[0], data_in + x_idx_offset_, sizeof(float));
        memcpy (&pt[1], data_in + y_idx_offset_, sizeof(float));
        memcpy (&pt[2], data_in + z_idx_offset_, sizeof(float));

        if (!pcl_isfinite (pt[0]) || !pcl_isfinite (pt[1]) || !pcl_isfinite (pt[2]))
          continue;

        pt_t = tr * pt;

        memcpy (data_out + x_idx_offset_, &pt_t[0], sizeof(float));
        memcpy (data_out + y_idx_offset_, &pt_t[1], sizeof(float));
        memcpy (data_out + z_idx_offset_, &pt_t[2], sizeof(float));
      }
    }

  }

template<typename PointSource, typename PointTarget, typename Scalar>
  void
  faat_pcl::IterativeClosestPointWithGC<PointSource, PointTarget, Scalar>::drawCorrespondences (PointCloudTargetConstPtr scene_cloud,
                                                                                                PointCloudSourcePtr model_cloud,
                                                                                                PointCloudTargetConstPtr keypoints_pointcloud,
                                                                                                PointCloudSourcePtr keypoints_model,
                                                                                                pcl::Correspondences & correspondences,
                                                                                                pcl::visualization::PCLVisualizer & vis_corresp_,
                                                                                                int viewport)
  {
    pcl::visualization::PointCloudColorHandlerCustom<PointSource> random_handler (scene_cloud, 255, 0, 0);
    vis_corresp_.addPointCloud<PointSource> (scene_cloud, random_handler, "points", viewport);

    PointCloudSourcePtr model_cloud_translated (new PointCloudSource);
    Eigen::Matrix4f translate;
    translate.setIdentity ();
    translate (0, 3) = -0.1f;
    translate (1, 3) = -0.1f;
    translate (2, 3) = -0.1f;
    pcl::transformPointCloud (*model_cloud, *model_cloud_translated, translate);
    pcl::visualization::PointCloudColorHandlerCustom<PointTarget> random_handler_sampled (model_cloud_translated, 0, 0, 255);
    vis_corresp_.addPointCloud<PointTarget> (model_cloud_translated, random_handler_sampled, "sampled", viewport);

    for (size_t kk = 0; kk < correspondences.size (); kk++)
    {
      pcl::PointXYZ p;
      p.getVector4fMap () = model_cloud_translated->points[correspondences[kk].index_query].getVector4fMap ();
      pcl::PointXYZ p_scene;
      p_scene.getVector4fMap () = keypoints_pointcloud->points[correspondences[kk].index_match].getVector4fMap ();

      std::stringstream line_name;
      line_name << "line_" << kk;

      vis_corresp_.addLine<pcl::PointXYZ, pcl::PointXYZ> (p_scene, p, line_name.str (), viewport);
    }
  }

//drawCorrespondences(target_, input_transformed, target_, input_transformed, *correspondences_);
//Brute force correspondences :)
/*correspondences_->clear();
 correspondences_->reserve(input_transformed->points.size() * target_->points.size());

 int valid = 0;
 int step=1;
 for(size_t i=0; i < input_transformed->points.size(); i+=step)
 {
 for(size_t j=(i+1); j < target_->points.size(); j+=step)
 {
 pcl::Correspondence p;
 p.index_query = i;
 p.index_match = j;
 p.distance = (input_transformed->points[i].getVector3fMap() - target_->points[j].getVector3fMap()).norm();

 if(p.distance <= corr_dist_threshold_)
 {
 correspondences_->push_back(p);
 valid++;
 }
 }
 }
 correspondences_->resize(valid);*/

template<typename PointSource, typename PointTarget, typename Scalar>
void
faat_pcl::IterativeClosestPointWithGC<PointSource, PointTarget, Scalar>::visualizeICPNodes
  (std::vector<boost::shared_ptr<ICPNode> > & nodes,
   pcl::visualization::PCLVisualizer & icp_vis,
   std::string wname)
{
  int k = 0, l = 0, viewport = 0;
  int y_s = 0, x_s = 0;
  double x_step = 0, y_step = 0;
  y_s = static_cast<int>(floor (sqrt (static_cast<float>(nodes.size ()))));
  x_s = y_s + static_cast<int>(ceil (double (nodes.size ()) / double (y_s) - y_s));
  x_step = static_cast<double>(1.0 / static_cast<double>(x_s));
  y_step = static_cast<double>(1.0 / static_cast<double>(y_s));

  for(size_t i=0; i < nodes.size(); i++)
  {
    icp_vis.createViewPort (k * x_step, l * y_step, (k + 1) * x_step, (l + 1) * y_step, viewport);
    k++;
    if (k >= x_s)
    {
      k = 0;
      l++;
    }

    PointCloudSourcePtr input_transformed (new PointCloudSource);
    pcl::transformPointCloud(*input_, *input_transformed, nodes[i]->accum_transform_);
    std::stringstream cloud_name;
    cloud_name << "input_" << i;

    //pcl::visualization::PointCloudColorHandlerCustom<PointSource> color_handler (input_transformed, 255, 0, 0);
    //icp_vis.addPointCloud (input_transformed, color_handler, cloud_name.str (), viewport);
    PointCloudSourcePtr pp = input_transformed;
    float rgb_m;
    bool exists_m;

    typedef pcl::PointCloud<PointSource> CloudM;
    typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

    pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (pp->points[0], "rgb", exists_m, rgb_m));
    if (exists_m)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::copyPointCloud(*pp,*cloud_rgb);
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (cloud_rgb);
      icp_vis.addPointCloud<pcl::PointXYZRGB> (cloud_rgb, handler_rgb, cloud_name.str (), viewport);
    }
    else
    {
      pcl::visualization::PointCloudColorHandlerCustom<PointSource> handler_rgb (pp, 255, 0, 0);
      icp_vis.addPointCloud<PointSource> (pp, handler_rgb, cloud_name.str (), viewport);
    }

    cloud_name << "_text";
    char cVal[32];
    sprintf(cVal,"%f", nodes[i]->reg_error_);

    std::cout << "OSV:" << nodes[i]->osv_fraction_ << std::endl;
    std::cout << "FSV:" << nodes[i]->fsv_fraction_ << std::endl;
    icp_vis.addText (cVal, 5, 5, 10, 1.0, 1.0, 1.0, cloud_name.str(), viewport);
  }

  PointCloudTargetConstPtr pp = target_;
  float rgb_m;
  bool exists_m;

  typedef pcl::PointCloud<PointTarget> CloudM;
  typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

  pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (pp->points[0], "rgb", exists_m, rgb_m));
  if (exists_m)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*pp,*cloud_rgb);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (cloud_rgb);
    icp_vis.addPointCloud<pcl::PointXYZRGB> (cloud_rgb, handler_rgb, "target");
  }
  else
  {
    pcl::visualization::PointCloudColorHandlerCustom<PointTarget> handler_rgb (pp, 255, 0, 0);
    icp_vis.addPointCloud<PointTarget> (pp, handler_rgb, "target");
  }

  //pcl::visualization::PointCloudColorHandlerCustom<PointTarget> color_handler (target_, 255, 255, 0);
  //icp_vis.addPointCloud(target_, color_handler, "target");
  icp_vis.spin();
}

template<typename PointSource, typename PointTarget, typename Scalar>
bool
faat_pcl::IterativeClosestPointWithGC<PointSource, PointTarget, Scalar>::filterHypothesesByPose
    (Eigen::Matrix4f & current,
     std::vector< Eigen::Matrix4f> & accepted_so_far,
     float trans_threshold,
     float min_angle)
{
    bool found = false;
    Eigen::Vector4f origin;
    pcl::compute3DCentroid(*input_, origin);
    origin[3] = 1.f;
    Eigen::Vector4f origin_trans = origin;
    origin_trans = current * origin_trans;
    Eigen::Vector3f trans = origin_trans.block<3,1>(0, 0);

    Eigen::Quaternionf quat(current.block<3,3>(0,0));
    quat.normalize();
    Eigen::Vector4f quat_4f = Eigen::Vector4f(quat.x(),quat.y(),quat.z(),quat.w());

    for(size_t i=0; i < accepted_so_far.size(); i++)
    {
      Eigen::Vector4f origin_node = origin;
      origin_node = accepted_so_far[i] * origin_node;
      Eigen::Vector3f trans_found = origin_node.block<3,1>(0, 0);
      if((trans - trans_found).norm() < trans_threshold)
      {
        Eigen::Quaternionf quat_found(static_cast<Eigen::Matrix4f>(accepted_so_far[i]).block<3,3>(0,0));
        quat_found.normalize();
        Eigen::Vector4f quat_found_4f = Eigen::Vector4f(quat_found.x(),quat_found.y(),quat_found.z(),quat_found.w());
        float for_acos = 2.f * quat_4f.dot(quat_found_4f) * quat_4f.dot(quat_found_4f) - 1;
        float angle = static_cast<float>(acos(for_acos));
        angle = std::abs(pcl::rad2deg(angle));
        if(angle < min_angle)
        {
          found = true;
          break;
        }

      }
    }

    return found;
}

///////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget, typename Scalar>
bool
faat_pcl::IterativeClosestPointWithGC<PointSource, PointTarget, Scalar>::filterHypothesesByPose
  (boost::shared_ptr<ICPNode> & current,
   std::vector<boost::shared_ptr<ICPNode> > & nodes,
   float trans_threshold)
{
  bool found = false;
  //Eigen::Vector4f origin = Eigen::Vector4f(0,0,0,1);
  Eigen::Vector4f origin;
  pcl::compute3DCentroid(*input_, origin);
  origin[3] = 1.f;
  Eigen::Vector4f origin_trans = origin;
  origin_trans = current->accum_transform_ * origin_trans;
  Eigen::Vector3f trans = origin_trans.block<3,1>(0, 0);

  Eigen::Quaternionf quat(static_cast<Eigen::Matrix4f>(current->accum_transform_).block<3,3>(0,0));
  quat.normalize();
  Eigen::Vector4f quat_4f = Eigen::Vector4f(quat.x(),quat.y(),quat.z(),quat.w());
  //Eigen::Quaternionf quat_conj = quat.conjugate();

  /*pcl::visualization::PCLVisualizer vis("angle test");
  int v1,v2;
  vis.createViewPort(0,0,0.5,1,v1);
  vis.createViewPort(0.5,0,1,1,v2);

  typename pcl::PointCloud<PointSource>::Ptr source_trans(new pcl::PointCloud<PointSource>);
  pcl::transformPointCloud(*input_, *source_trans, current->accum_transform_);

  pcl::visualization::PointCloudColorHandlerRGBField<PointSource> handler (source_trans);
  vis.addPointCloud<PointSource>(source_trans, handler, "target_cloud", v1);*/

  for(size_t i=0; i < nodes.size(); i++)
  {
    Eigen::Vector4f origin_node = origin;
    origin_node = nodes[i]->accum_transform_ * origin_node;
    Eigen::Vector3f trans_found = origin_node.block<3,1>(0, 0);
    if((trans - trans_found).norm() < trans_threshold)
    {
      Eigen::Quaternionf quat_found(static_cast<Eigen::Matrix4f>(nodes[i]->accum_transform_).block<3,3>(0,0));
      quat_found.normalize();
      Eigen::Vector4f quat_found_4f = Eigen::Vector4f(quat_found.x(),quat_found.y(),quat_found.z(),quat_found.w());

      /*typename pcl::PointCloud<PointSource>::Ptr source_trans(new pcl::PointCloud<PointSource>);
      pcl::transformPointCloud(*input_, *source_trans, nodes[i]->accum_transform_);
      pcl::visualization::PointCloudColorHandlerRGBField<PointSource> handler (source_trans);
      vis.addPointCloud<PointSource>(source_trans, handler, "source", v2);*/

      //Eigen::Quaternionf quat_prod = quat_found * quat_conj;
      float for_acos = 2.f * quat_4f.dot(quat_found_4f) * quat_4f.dot(quat_found_4f) - 1;
      float angle = static_cast<float>(acos(for_acos));
      angle = std::abs(pcl::rad2deg(angle));
      //std::cout << angle << std::endl;

      //vis.spin();
      //vis.removePointCloud("source");
      if(angle < 15.f)
      {
        found = true;
        break;
      }

    }
  }

  return found;
}

//#define VIS_KEYPOINTS
//#define VIS_SURVIVE
//#define VIS
//#define VIS_FINAL

template<typename PointT>
inline void
getIndicesFromCloudICPWithGC(typename pcl::PointCloud<PointT>::Ptr & processed,
                       typename pcl::PointCloud<PointT>::Ptr & keypoints_pointcloud,
                       std::vector<int> & indices,
                       float res = 0.003f)
{
  pcl::octree::OctreePointCloudSearch<PointT> octree (res);
  octree.setInputCloud (processed);
  octree.addPointsFromInputCloud ();

  std::vector<int> pointIdxNKNSearch;
  std::vector<float> pointNKNSquaredDistance;

  for(size_t j=0; j < keypoints_pointcloud->points.size(); j++)
  {
   if (octree.nearestKSearch (keypoints_pointcloud->points[j], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
   {
     indices.push_back(pointIdxNKNSearch[0]);
   }
  }
}

template<typename PointT>
inline void
computeISSKeypoints(typename pcl::PointCloud<PointT>::Ptr & cloud,
                       typename pcl::PointCloud<PointT>::Ptr & keypoints, std::vector<int> & indices)
{
  double iss_salient_radius_;
  double iss_non_max_radius_;
  double iss_normal_radius_;
  double iss_border_radius_;
  double iss_gamma_21_ (0.8);
  double iss_gamma_32_ (0.8);
  double iss_min_neighbors_ (5);
  int iss_threads_ (4);

  double model_resolution = 0.001;

  // Compute model_resolution
  iss_salient_radius_ = 0.03f;
  iss_non_max_radius_ = 0.003f;
  iss_normal_radius_ = 0.015f;

  pcl::ISSKeypoint3D<PointT, PointT> iss_detector;

  keypoints.reset(new pcl::PointCloud<PointT>);
  typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
  iss_detector.setSearchMethod (tree);
  iss_detector.setSalientRadius (iss_salient_radius_);
  iss_detector.setNonMaxRadius (iss_non_max_radius_);

  iss_detector.setNormalRadius (iss_normal_radius_);

  iss_detector.setThreshold21 (iss_gamma_21_);
  iss_detector.setThreshold32 (iss_gamma_32_);
  iss_detector.setMinNeighbors (iss_min_neighbors_);
  iss_detector.setNumberOfThreads (iss_threads_);
  iss_detector.setInputCloud (cloud);
  iss_detector.compute (*keypoints);

  std::cout << "ISS keypoints:" << keypoints->points.size() << std::endl;
  //get indices to cloud
  getIndicesFromCloudICPWithGC<PointT>(cloud, keypoints, indices);
}

template<typename PointT>
inline void
uniformSamplingOfKeypoints(typename pcl::PointCloud<PointT>::Ptr & keypoint_cloud,
                              std::vector<int> & indices_keypoints,
                              std::vector<int> & indices,
                              float radius)
{

  pcl::UniformSampling<PointT> keypoint_extractor;
  keypoint_extractor.setRadiusSearch (radius);
  keypoint_extractor.setInputCloud (keypoint_cloud);
  pcl::PointCloud<int> keypoints;
  keypoint_extractor.compute (keypoints);

  indices.resize(keypoints.points.size());
  for(size_t i=0; i < keypoints.points.size(); i++)
  {
    indices[i] = indices_keypoints[keypoints[i]];
  }
}

/*template<>
struct SIFTKeypointFieldSelector<PointNormal>
{
  inline float
  operator () (const PointNormal & p) const
  {
    return p.curvature;
  }
};*/

/*template<typename T>
inline void transformToGPUCloudFormat(const typename pcl::PointCloud<T> & pcl_cloud,
                                           std::vector<faat_pcl::xyz_p> & gpu_cloud)
{
  gpu_cloud.resize(pcl_cloud.points.size());
  for(size_t i=0; i < gpu_cloud.size(); i++)
  {
    gpu_cloud[i].x = pcl_cloud.points[i].x;
    gpu_cloud[i].y = pcl_cloud.points[i].y;
    gpu_cloud[i].z = pcl_cloud.points[i].z;
  }
}*/

inline void transformVoxelGridToGPUVxGrid(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel> * vx,
                                                faat_pcl::cuda::registration::VoxelGrid<faat_pcl::PropDistanceFieldVoxel> * gpu_vx)
{
  gpu_vx->setOrigin(vx->getOrigin(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel>::DIM_X),
                    vx->getOrigin(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel>::DIM_Y),
                    vx->getOrigin(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel>::DIM_Z) );

  gpu_vx->setSize(vx->getSize(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel>::DIM_X),
                  vx->getSize(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel>::DIM_Y),
                  vx->getSize(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel>::DIM_Z) );

  int cells_x, cells_y, cells_z;
  cells_x = vx->getNumCells(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel>::DIM_X);
  cells_y = vx->getNumCells(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel>::DIM_Y);
  cells_z = vx->getNumCells(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel>::DIM_Z);
  gpu_vx->setNumCells(vx->getNumCells(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel>::DIM_X),
                      vx->getNumCells(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel>::DIM_Y),
                      vx->getNumCells(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel>::DIM_Z) );

  gpu_vx->setResolution(vx->getResolution(distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel>::DIM_X));
  faat_pcl::PropDistanceFieldVoxel * data = new faat_pcl::PropDistanceFieldVoxel[cells_x * cells_y * cells_z];
  distance_field::PropDistanceFieldVoxel * data_vx = vx->getDataPointer();
  for(size_t i=0; i < (cells_x * cells_y * cells_z); i++)
  {
    faat_pcl::PropDistanceFieldVoxel v;
    for(size_t j=0; j < 3; j++)
    {
      v.closest_point_[j] = data_vx[i].closest_point_[j];
      v.location_[j] = data_vx[i].location_[j];
    }
    v.distance_square_ = data_vx[i].distance_square_;
    v.occupied_ = data_vx[i].occupied_;
    if(data_vx[i].occupied_)
      v.idx_to_input_cloud_ = data_vx[i].idx_to_input_cloud_;

    data[i] = v;
  }

  gpu_vx->setCells(data);
  delete[] data;
}
///////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget, typename Scalar>
void
faat_pcl::IterativeClosestPointWithGC<PointSource, PointTarget, Scalar>::computeTransformation
  (PointCloudSource &output, const Matrix4 &guess)
  {
    pcl::ScopeTime t("total time: IterativeClosestPointWithGC::computeTransformation");
    faat_pcl::cuda::registration::ICPWithGC<xyz_rgb, faat_pcl::PropDistanceFieldVoxel> gpu_gc_icp_;
    // Point cloud containing the correspondences of each point in <input, indices>
    PointCloudSourcePtr input_transformed (new PointCloudSource);
    pcl::PointCloud<pcl::Normal>::Ptr input_transformed_normals_ (new pcl::PointCloud<pcl::Normal>);

    nr_iterations_ = 0;
    converged_ = false;
    float downsample_leaf_size = dt_vx_size_;
    float uniform_sampling_radius = downsample_leaf_size * 2.f;
    float gc_size_ = downsample_leaf_size * 1.5f;
    float ransac_threshold = gc_size_;

    // Initialise final transformation to the guessed one
    final_transformation_ = guess;

    // If the guessed transformation is non identity
    if (guess != Matrix4::Identity ())
      // Apply guessed transformation prior to search for neighbours
      transformPointCloud (*input_, *input_transformed, guess);
    else
      *input_transformed = *input_;

    if(!input_transformed || input_transformed->points.size() == 0)
    {
      PCL_WARN("Input cloud is empty!\n");
    }

    if(!target_ || target_->points.size() == 0)
    {
      PCL_WARN("Input cloud is empty!\n");
    }

    PointCloudTargetConstPtr target_icp(new pcl::PointCloud<PointTarget>(*target_));
    PointCloudTargetPtr target_range_image(new pcl::PointCloud<PointTarget>(*target_));
    PointCloudSourcePtr source_range_image(new pcl::PointCloud<PointSource>(*input_transformed));

    PointCloudTargetPtr target_range_image_downsampled(new pcl::PointCloud<PointTarget>());
    PointCloudSourcePtr source_range_image_downsampled(new pcl::PointCloud<PointSource>());

    {
      pcl::UniformSampling<PointSource> keypoint_extractor;
      keypoint_extractor.setInputCloud (target_range_image);
      keypoint_extractor.setRadiusSearch(downsample_leaf_size);
      pcl::PointCloud<int> keypoints;
      keypoint_extractor.compute (keypoints);
      pcl::copyPointCloud(*target_range_image, keypoints.points, *target_range_image_downsampled);

      keypoint_extractor.setInputCloud (source_range_image);
      keypoint_extractor.compute (keypoints);
      pcl::copyPointCloud(*source_range_image, keypoints.points, *source_range_image_downsampled);
    }

    //int max_points = std::min(static_cast<int>(input_transformed->points.size()), static_cast<int>(target_icp->points.size()));
    int max_points = std::min(static_cast<int>(source_range_image_downsampled->points.size()), static_cast<int>(target_range_image_downsampled->points.size()));

    //GPU hypotheses evaluation - transform target and input_transformed cloud
    std::vector<faat_pcl::xyz_rgb> * target_gpu = new std::vector<faat_pcl::xyz_rgb>();
    std::vector<faat_pcl::xyz_rgb> * input_transformed_gpu = new std::vector<faat_pcl::xyz_rgb>();

    getFieldValueTarget_(*target_, *target_gpu);
    getFieldValueSource_(*source_range_image_downsampled, *input_transformed_gpu);

    gpu_gc_icp_.setInputCloud(input_transformed_gpu);
    gpu_gc_icp_.setTargetCloud(target_gpu);

    if(!range_images_provided_ && use_range_images_)
    {
      float min_fl = std::numeric_limits<float>::max ();
      float max_fl = -1;
      int cx, cy;
      cx = cy = 150;
      faat_pcl::registration::VisibilityReasoning<PointSource> vr_s (0, 0, 0);
      float fl_source = vr_s.computeFocalLength (cx, cy, input_transformed);

      faat_pcl::registration::VisibilityReasoning<PointTarget> vr_t (0, 0, 0);
      float fl_target = vr_t.computeFocalLength (cx, cy, target_range_image);

      min_fl = std::min(fl_source, fl_target);
      max_fl = std::max(fl_source, fl_target);

      vr_t.computeRangeImage (cx, cy, min_fl, target_range_image, range_image_target_);
      vr_s.computeRangeImage (cx, cy, min_fl, input_transformed, range_image_source_);

      fl_ = min_fl;
      cx_ = cx;
      cy_ = cy;
    }

    typedef pcl::SHOT352 DescriptorType;

    pcl::PointCloud<DescriptorType>::Ptr source_descriptors (new pcl::PointCloud<DescriptorType> ());
    pcl::PointCloud<DescriptorType>::Ptr target_descriptors (new pcl::PointCloud<DescriptorType> ());

    faat_pcl::registration::UniformSamplingSharedVoxelGrid<PointSource> keypoint_extractor;
    keypoint_extractor.setRadiusSearch (uniform_sampling_radius);
    float descr_rad_ = 0.04f;
    Eigen::Vector4f min_b, max_b;

    pcl::IndicesPtr ind_src(new std::vector<int>);
    pcl::IndicesPtr ind_tgt(new std::vector<int>);

    {
      pcl::SHOTEstimationOMP<PointSource, PointSource, DescriptorType> descr_est;
      descr_est.setRadiusSearch (descr_rad_);

      typename pcl::PointCloud<PointSource>::Ptr source_keypoints(new pcl::PointCloud<PointSource>);
      typename pcl::PointCloud<PointTarget>::Ptr tgt_keypoints(new pcl::PointCloud<PointTarget>);

      //computeISSkEYPOINTS
      if(ind_src_ && ind_tgt_ && (ind_src_->size() > 0) && (ind_tgt_->size() > 0))
      {
        ind_src.reset(new std::vector<int>(*ind_src_));
        ind_tgt.reset(new std::vector<int>(*ind_tgt_));
      }
      else
      {
        std::vector<int> indices_src_iss, indices_tgt_iss;
        computeISSKeypoints<PointSource>(input_transformed, source_keypoints, indices_src_iss);
        computeISSKeypoints<PointTarget>(target_range_image, tgt_keypoints, indices_tgt_iss);

        //UNIFORM SAMPLING OF ISS keypoints for target, keep voxelgrid values
        {
          keypoint_extractor.setInputCloud (tgt_keypoints);
          pcl::PointCloud<int> keypoints;
          keypoint_extractor.compute (keypoints);
          keypoint_extractor.getVoxelGridValues(min_b, max_b);

          std::vector<int> indices;
          indices.resize(keypoints.points.size());
          for(size_t i=0; i < keypoints.points.size(); i++)
          {
            indices[i] = indices_tgt_iss[keypoints[i]];
          }

          ind_tgt->clear();
          ind_tgt.reset(new std::vector<int>(indices));
        }

        //UNIFORM SAMPLING OF ISS keypoints for input
        {
          keypoint_extractor.setInputCloud (source_keypoints);
          pcl::PointCloud<int> keypoints;
          keypoint_extractor.compute (keypoints);

          std::vector<int> indices;
          indices.resize(keypoints.points.size());
          for(size_t i=0; i < keypoints.points.size(); i++)
          {
            indices[i] = indices_src_iss[keypoints[i]];
          }

          ind_src->clear();
          ind_src.reset(new std::vector<int>(indices));
        }
      }

      pcl::copyPointCloud(*input_transformed, *ind_src, *source_keypoints);
      pcl::copyPointCloud(*target_icp, *ind_tgt, *tgt_keypoints);

      std::cout << tgt_keypoints->points.size() <<  " " << source_keypoints->points.size() << std::endl;

#ifdef VIS_KEYPOINTS
      pcl::visualization::PCLVisualizer vis ("KEYPOINTS");
      int v2, v3;
      vis.createViewPort (0., 0, 0.5, 1, v2);
      vis.createViewPort (0.5, 0, 1, 1, v3);
      std::cout << tgt_keypoints->points.size() <<  " " << source_keypoints->points.size() << std::endl;

      {
        pcl::visualization::PointCloudColorHandlerCustom<PointTarget> handler_rgb (target_icp, 255, 255, 255);
        vis.addPointCloud<PointTarget> (target_icp, handler_rgb, "target", v3);
      }

      {
        pcl::visualization::PointCloudColorHandlerCustom<PointSource> handler_rgb (input_transformed, 255, 255, 255);
        vis.addPointCloud<PointSource> (input_transformed, handler_rgb, "input", v2);
      }

      pcl::visualization::PointCloudColorHandlerCustom<PointSource> handler_rgb (source_keypoints, 255, 0, 0);
      vis.addPointCloud<PointSource> (source_keypoints, handler_rgb, "source keypoints", v2);

      {
        pcl::visualization::PointCloudColorHandlerCustom<PointTarget> handler_rgb (tgt_keypoints, 255, 0, 0);
        vis.addPointCloud<PointTarget> (tgt_keypoints, handler_rgb, "target keypoints", v3);
      }

      vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 16, "source keypoints");
      vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 16, "target keypoints");

      vis.spin();
#endif

      if(use_shot_)
      {
          descr_est.setInputCloud (source_keypoints);
          descr_est.setInputNormals (input_transformed);
          descr_est.setSearchSurface (input_transformed);
          descr_est.compute (*source_descriptors);

          descr_est.setInputCloud (tgt_keypoints);
          descr_est.setInputNormals (target_icp);
          descr_est.setSearchSurface (target_icp);
          descr_est.compute (*target_descriptors);
      }
    }

    pcl::CorrespondencesPtr source_target_corrs (new pcl::Correspondences ());

    if(use_shot_)
    {
        pcl::KdTreeFLANN<DescriptorType> match_search;
        match_search.setInputCloud (source_descriptors);

        if(source_descriptors->size() > 0 && target_descriptors->size() > 0)
        {
          int k=1;
          std::vector<int> neigh_indices (k);
          std::vector<float> neigh_sqr_dists (k);

          for (size_t i = 0; i < target_descriptors->size (); ++i)
          {
            if (!pcl_isfinite (target_descriptors->at (i).descriptor[0])) //skipping NaNs
              continue;

            int found_neighs = match_search.nearestKSearch (target_descriptors->at (i), k, neigh_indices, neigh_sqr_dists);
            if(found_neighs >= 1) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
            {
              for(size_t j=0; j < found_neighs; j++)
              {
                pcl::Correspondence corr (neigh_indices[j], static_cast<int> (i), neigh_sqr_dists[j]);
                source_target_corrs->push_back (corr);
              }
            }
          }

          //reverse shot matching...
          match_search.setInputCloud (target_descriptors);
          for (size_t i = 0; i < source_descriptors->size (); ++i)
          {
            if (!pcl_isfinite (source_descriptors->at (i).descriptor[0])) //skipping NaNs
              continue;

            int found_neighs = match_search.nearestKSearch (source_descriptors->at (i), k, neigh_indices, neigh_sqr_dists);
            if(found_neighs >= 1) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
            {
              for(size_t j=0; j < found_neighs; j++)
              {
                pcl::Correspondence corr (static_cast<int> (i), neigh_indices[j], neigh_sqr_dists[j]);
                source_target_corrs->push_back (corr);
              }
            }
          }
        }
    }

    //extend source_target_corrs with correspondences given from outside
    if(source_target_corrs_user_)
    {
        for(size_t k=0; k < source_target_corrs_user_->size(); k++)
        {
            source_target_corrs->push_back (source_target_corrs_user_->at(k));
        }
    }

    //std::cout << "going to compute distance transform..." << target_icp->points.size() << " " << dt_vx_size_ << std::endl;

    typename boost::shared_ptr<distance_field::PropagationDistanceField<PointTarget> > dist_trans;
    dist_trans.reset(new distance_field::PropagationDistanceField<PointTarget>(dt_vx_size_));
    dist_trans->setInputCloud(target_icp);
    dist_trans->compute();

    distance_field::VoxelGrid<distance_field::PropDistanceFieldVoxel> * vx = dist_trans->extractVoxelGrid();
    faat_pcl::cuda::registration::VoxelGrid<faat_pcl::PropDistanceFieldVoxel> * vx_gpu =
        new faat_pcl::cuda::registration::VoxelGrid<faat_pcl::PropDistanceFieldVoxel>;

    transformVoxelGridToGPUVxGrid(vx, vx_gpu);
    gpu_gc_icp_.setVoxelGrid(vx_gpu);
    transformation_ = Matrix4::Identity ();

    pcl::PointCloud<pcl::Normal>::Ptr target_normals_ (new pcl::PointCloud<pcl::Normal>);
    pcl::copyPointCloud (*target_, *target_normals_);

    /*convergence_criteria_->setMaximumIterations (max_iterations_);
    convergence_criteria_->setRelativeMSE (euclidean_fitness_epsilon_);
    convergence_criteria_->setTranslationThreshold (transformation_epsilon_);
    convergence_criteria_->setRotationThreshold (1.0 - transformation_epsilon_);*/

#ifdef VIS_SURVIVE
    pcl::visualization::PCLVisualizer survived_vis ("survived VIS...");
#endif

#ifdef VIS
    pcl::visualization::PCLVisualizer icp_vis ("icp VIS...");

    int v1, v2, v3;
    icp_vis.createViewPort (0, 0, 0.33, 1, v1);
    icp_vis.createViewPort (0.33, 0, 0.66, 1, v2);
    icp_vis.createViewPort (0.66, 0, 1, 1, v3);

    {
      pcl::visualization::PointCloudColorHandlerCustom<PointTarget> handler (target_, 0, 255, 0);
      icp_vis.addPointCloud<PointTarget> (target_, handler, "target_cloud");
    }
#endif

    float best_registration_error = -1.f;
    float threshold_best_ = 0.5f;
    int max_keep = 8;
    int best_overlap = 0;
    Eigen::Matrix4f best_transformation_;
    best_transformation_.setIdentity ();

    std::vector<boost::shared_ptr<ICPNode> > alive_nodes_; //contains nodes that need to be processed at a certain ICP iteration
    std::vector<boost::shared_ptr<ICPNode> > nodes_survived_after_filtering;

    //create root_node
    if(initial_poses_.size() == 0)
    {
        boost::shared_ptr<ICPNode> root_node_ (new ICPNode (0, true));
        root_node_->accum_transform_ = Eigen::Matrix4f::Identity ();
        alive_nodes_.push_back (root_node_);
        nodes_survived_after_filtering.push_back(root_node_);
    }
    else
    {
      for(size_t i=0; i < initial_poses_.size(); i++)
      {
        boost::shared_ptr<ICPNode> root_node_ (new ICPNode (0, true));
        root_node_->accum_transform_ = initial_poses_[i];
        alive_nodes_.push_back (root_node_);
        nodes_survived_after_filtering.push_back(root_node_);
      }
    }

#ifdef VIS_SURVIVE
        visualizeICPNodes(nodes_survived_after_filtering, survived_vis, "survived");
        survived_vis.removeAllPointClouds();
        survived_vis.removeAllShapes();
#endif

    pcl::IndicesPtr copy_ind_src(new std::vector<int>(*ind_src));
    //pcl::IndicesPtr copy_ind_tgt(new std::vector<int>(*ind_tgt));

    /*keypoint_extractor.setInputCloud (target_range_image);
    keypoint_extractor.setRadiusSearch(0.02f);
    pcl::PointCloud<int> keypoints_idxes_tgt_us;
    keypoint_extractor.compute (keypoints_idxes_tgt_us);
    for(size_t i=0; i < keypoints_idxes_tgt_us.points.size(); i++)
      ind_tgt->push_back(keypoints_idxes_tgt_us.points[i]);

    keypoint_extractor.getVoxelGridValues(min_b, max_b);*/

    // Repeat until convergence
    std::vector<boost::shared_ptr<ICPNode> > converged_branches_;
    do
    {
      //pcl::ScopeTime t_iter("One iteration...");
      //std::cout << "Iterations:" << nr_iterations_ << " " << alive_nodes_.size () << std::endl;

      std::vector<boost::shared_ptr<ICPNode> > next_level_nodes_;
      {
        //pcl::ScopeTime t("Processing alive nodes...");
        std::vector<std::vector<boost::shared_ptr<ICPNode> > > next_level_nodes_by_parent_;
        next_level_nodes_by_parent_.resize(alive_nodes_.size());
#pragma omp parallel for num_threads(8)
        for(size_t an=0; an < alive_nodes_.size(); an++)
        {
          PointCloudSourcePtr input_transformed_local_ (new PointCloudSource);
          pcl::PointCloud<pcl::Normal>::Ptr input_transformed_normals_local_ (new pcl::PointCloud<pcl::Normal>);

          typename boost::shared_ptr<pcl::registration::CorrespondenceEstimation<PointSource, PointTarget, Scalar> > corresp_est;
          corresp_est.reset (new pcl::registration::CorrespondenceEstimation<PointSource, PointTarget, Scalar>);
          corresp_est->setInputSource (input_transformed_local_);
          corresp_est->setInputTarget (target_);

          pcl::CorrespondencesPtr correspondences_alive_node(new pcl::Correspondences);
          boost::shared_ptr<ICPNode> cur_node = alive_nodes_[an];
          pcl::transformPointCloudWithNormals (*input_, *input_transformed_local_, cur_node->accum_transform_);
          pcl::copyPointCloud (*input_transformed_local_, *input_transformed_normals_local_);

          pcl::IndicesPtr ind_src_local;
          ind_src_local.reset(new std::vector<int>(*copy_ind_src));

          //do an additional uniform sampling using the voxelgrid from target on the input cloud in its current pose
          //pcl::UniformSampling<PointSource> keypoint_extractor;
          /*faat_pcl::registration::UniformSamplingSharedVoxelGrid<PointSource> keypoint_extractor;
          //keypoint_extractor.setVoxelGridValues(min_b, max_b);
          keypoint_extractor.setVoxelGridValues (min_b, max_b);
          keypoint_extractor.setRadiusSearch(0.02f);
          keypoint_extractor.setInputCloud (input_transformed_local_);
          pcl::PointCloud<int> keypoints_idxes_src;
          keypoint_extractor.compute (keypoints_idxes_src);

          for(size_t i=0; i < keypoints_idxes_src.points.size(); i++)
            ind_src_local->push_back(keypoints_idxes_src.points[i]);*/

          {
            //pcl::ScopeTime t ("correspondence_estimation_...");
            if (use_reciprocal_correspondence_)
              corresp_est->determineReciprocalCorrespondences (*correspondences_alive_node, corr_dist_threshold_);
            else
            {
              corresp_est->setIndicesSource(ind_src_local);
              corresp_est->determineCorrespondences (*correspondences_alive_node, corr_dist_threshold_);

              corresp_est->setIndicesSource(ind_tgt);
              corresp_est->setInputSource (target_);
              corresp_est->setInputTarget (input_transformed_local_);


              CorrespondencesPtr reverse_correspondences (new Correspondences ());
              corresp_est->determineCorrespondences (*reverse_correspondences, corr_dist_threshold_);

              for(size_t i=0; i < reverse_correspondences->size(); i++)
              {
                pcl::Correspondence rev_corresp;
                rev_corresp.index_match = (*reverse_correspondences)[i].index_query;
                rev_corresp.index_query = (*reverse_correspondences)[i].index_match;
                rev_corresp.distance = (*reverse_correspondences)[i].distance;
                correspondences_alive_node->push_back(rev_corresp);
              }
            }

            //add correspondences from SHOT
            for(size_t i=0; i < source_target_corrs->size(); i++)
            {
              correspondences_alive_node->push_back(source_target_corrs->at(i));
            }
          }

          //correspondence grouping...
          std::vector<pcl::Correspondences> corresp_clusters;
          if(!use_cg_)
          {
            std::cout << "Number of correspondences:" << correspondences_alive_node->size() << std::endl;
            corresp_clusters.push_back(*correspondences_alive_node);
          }
          else
          {
            std::cout << "Number of correspondences:" << correspondences_alive_node->size() << std::endl;
            pcl::ScopeTime t ("corresponde grouping...");
            faat_pcl::GraphGeometricConsistencyGrouping<PointSource, PointTarget> gcg_alg;
            gcg_alg.setGCThreshold (min_number_correspondences_);
            gcg_alg.setGCSize (gc_size_);
            gcg_alg.setDotDistance (0.25f);
            gcg_alg.setRansacThreshold (ransac_threshold);
            gcg_alg.setDistForClusterFactor(0.f);
            gcg_alg.setMaxTaken(1);

            gcg_alg.setCheckNormalsOrientation(false);
            gcg_alg.setSortCliques(true);
            gcg_alg.pruneByCC(false);
            gcg_alg.setUseGraph (true);
            gcg_alg.setMaxTimeForCliquesComputation(100);
            gcg_alg.setPrune (false);
            gcg_alg.setSceneCloud (target_);
            gcg_alg.setInputCloud (input_transformed_local_);
            gcg_alg.setModelSceneCorrespondences (correspondences_alive_node);
            gcg_alg.setInputAndSceneNormals (input_transformed_normals_local_, target_normals_);
            gcg_alg.cluster (corresp_clusters);
          }

          typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointSource>::Ptr
                                                                                              rej (
                                                                                                   new pcl::registration::CorrespondenceRejectorSampleConsensus<
                                                                                                       PointSource> ());

          //go through clusters and compute things
          for (size_t kk = 0; kk < corresp_clusters.size (); kk++)
          {

            //std::cout << corresp_clusters[kk].size() << std::endl;
            CorrespondencesPtr temp_correspondences (new Correspondences (corresp_clusters[kk]));
            CorrespondencesPtr after_rej_correspondences (new Correspondences ());

            rej->setMaximumIterations (50000);
            rej->setInlierThreshold (ransac_threshold);
            rej->setInputTarget (target_);
            rej->setInputSource (input_transformed_local_);
            rej->setInputCorrespondences (temp_correspondences);
            rej->getCorrespondences (*after_rej_correspondences);

            size_t cnt = after_rej_correspondences->size ();
            //std::cout << "after rejection:" << cnt << std::endl;
            if (cnt >= min_number_correspondences_)
            {
              Eigen::Matrix4f transformation = rej->getBestTransformation();
              transformation_estimation_->estimateRigidTransformation (*input_transformed_local_, *target_, *after_rej_correspondences, transformation);
              boost::shared_ptr<ICPNode> child (new ICPNode (nr_iterations_ + 1));
              //child->incr_transform_ = transformation; //goes from cur_node to tgt
              child->accum_transform_ = transformation * cur_node->accum_transform_;
              //child->parent_ = cur_node;
              child->after_rej_correspondences_ = after_rej_correspondences;
              //next_level_nodes_.push_back (child);
              next_level_nodes_by_parent_[an].push_back(child);
            }
          }
        }

        alive_nodes_.clear();
        for(size_t i=0; i < next_level_nodes_by_parent_.size(); i++)
        {
          for(size_t j=0; j < next_level_nodes_by_parent_[i].size(); j++)
          {
            next_level_nodes_.push_back (next_level_nodes_by_parent_[i][j]);
          }
        }
      }

      std::cout << "Next level size:" << next_level_nodes_.size () << std::endl;
      //visualizeICPNodes(next_level_nodes_);

      {
        float max_ov = ov_percentage_ * max_points;
        pcl::ScopeTime tt("hypotheses evaluation...\n");
        //std::cout << "Next level size:" << next_level_nodes_.size () << std::endl;

        //GPU hypotheses evaluation - transform next level nodes
        //icp_with_gc_gpu_utils

        {
          pcl::ScopeTime t("ICP GPU evaluation...");
          faat_pcl::ICPNodeGPU * gpu_icp_nodes = new faat_pcl::ICPNodeGPU[next_level_nodes_.size()];

          for (size_t k = 0; k < next_level_nodes_.size (); k++)
          {
            gpu_icp_nodes[k].overlap_ = 0;
            gpu_icp_nodes[k].reg_error_ = 0.f;
            gpu_icp_nodes[k].color_reg_ = 0.f;
            for(size_t ii=0; ii < 4; ii++)
              for(size_t jj=0; jj < 4; jj++)
                gpu_icp_nodes[k].transform_.mat[ii][jj] = next_level_nodes_[k]->accum_transform_(ii,jj);
          }

          gpu_gc_icp_.setHypothesesToEvaluate(gpu_icp_nodes, static_cast<int>(next_level_nodes_.size()));
          //gpu_gc_icp_.setInliersThreshold(std::max(inliers_threshold_ - 0.0005f * nr_iterations_ , inliers_threshold_));
          gpu_gc_icp_.setInliersThreshold(inliers_threshold_);

          {
            //pcl::ScopeTime t("computeOverlapAndRegistrationError...");
            gpu_gc_icp_.computeOverlapAndRegistrationError();
          }

          for (size_t k = 0; k < next_level_nodes_.size (); k++)
          {
            next_level_nodes_[k]->overlap_ = gpu_icp_nodes[k].overlap_;
            next_level_nodes_[k]->reg_error_ = gpu_icp_nodes[k].reg_error_;
            next_level_nodes_[k]->color_weight_ = gpu_icp_nodes[k].color_reg_;
          }

          delete[] gpu_icp_nodes;

        }


        /*{
          pcl::ScopeTime tt("hypotheses evaluation - overlap & distance weight...\n");

//#pragma omp parallel for num_threads(8)
          for (size_t k = 0; k < next_level_nodes_.size (); k++)
          {
            //pcl::ScopeTime t("computing fitness score...");
            // Tranform the data and compute fitness score...
            PointCloudSourcePtr input_transformed_intern (new PointCloudSource());
            pcl::transformPointCloudWithNormals (*input_, *input_transformed_intern, next_level_nodes_[k]->accum_transform_);

            next_level_nodes_[k]->reg_error_ = 0.f;
            next_level_nodes_[k]->overlap_ = 0;

            for (size_t kkk = 0; kkk < input_transformed_intern->points.size (); kkk++)
            {
              int idx_match;
              float distance;
              float color_distance = -1.f;
              dist_trans->getCorrespondence(input_transformed_intern->points[kkk], &idx_match, &distance, -1.f, &color_distance);
              if((idx_match) >= 0 && (distance < inliers_threshold_)) //ATTENTION: check this!! is distance squared?
              {
                float d_weight = -(distance / (inliers_threshold_)) + 1;
                //Eigen::Vector3f scene_p_normal = input_transformed_intern->points[kkk].getNormalVector3fMap ();
                //Eigen::Vector3f model_p_normal = target_->points[idx_match].getNormalVector3fMap ();
                //float dotp = scene_p_normal.dot (model_p_normal);
                //dotp = 1.f;
                next_level_nodes_[k]->overlap_++;
                next_level_nodes_[k]->reg_error_ += d_weight; // * dotp;
              }
            }
          }
        }*/

        if(use_range_images_)
        {
          //TODO: Port this to the gpu too!
          //pcl::ScopeTime t("FSV computation...");
          float osv_cutoff = 0.1f;
          float fsv_cutoff = 0.01f;
  #pragma omp parallel for num_threads(8)
          for (size_t k = 0; k < next_level_nodes_.size (); k++)
          {
            if (next_level_nodes_[k]->overlap_ > 0)
            {
              //compute FSV fraction
              faat_pcl::registration::VisibilityReasoning<PointSource> vr (fl_, cx_, cy_);
              vr.setThresholdTSS (inliers_threshold_);
              /*float fsv_ij = vr.computeFSV (range_image_target, source_range_image, next_level_nodes_[k]->accum_transform_);
              float fsv_ji = vr.computeFSV (range_image_source, target_range_image, next_level_nodes_[k]->accum_transform_.inverse ());*/

              float fsv_ij = vr.computeFSV (range_image_target_, source_range_image_downsampled, next_level_nodes_[k]->accum_transform_);
              float fsv_ji = vr.computeFSV (range_image_source_, target_range_image_downsampled, next_level_nodes_[k]->accum_transform_.inverse ());

              float osv_ij = vr.computeOSV (range_image_target_, source_range_image_downsampled, next_level_nodes_[k]->accum_transform_);
              float osv_ji = vr.computeOSV (range_image_source_, target_range_image_downsampled, next_level_nodes_[k]->accum_transform_.inverse ());
              float osv_fraction = std::max(osv_ij, osv_ji);

              float fsv_fraction = std::max (fsv_ij, fsv_ji);
              float ov = std::min (static_cast<float> (next_level_nodes_[k]->overlap_), max_ov) / static_cast<float> (max_points);
              next_level_nodes_[k]->reg_error_ =
                  (next_level_nodes_[k]->reg_error_ / static_cast<float> (next_level_nodes_[k]->overlap_)) * (ov)
                  * (1.f - ( std::max(fsv_fraction, fsv_cutoff) - fsv_cutoff)); /* * (1.f - ( std::max(osv_fraction, osv_cutoff) - osv_cutoff))*/
                  //* (next_level_nodes_[k]->color_weight_ / static_cast<float> (next_level_nodes_[k]->overlap_));

              next_level_nodes_[k]->osv_fraction_ = osv_fraction;
              next_level_nodes_[k]->fsv_fraction_ = fsv_fraction;
              next_level_nodes_[k]->overlap_ = ov;
            }
            else
            {
              next_level_nodes_[k]->reg_error_ = -1.f;
            }
          }
        }
        else
        {
            for (size_t k = 0; k < next_level_nodes_.size (); k++)
            {
              if (next_level_nodes_[k]->overlap_ > 0)
              {
                float ov = std::min (static_cast<float> (next_level_nodes_[k]->overlap_), max_ov) / static_cast<float> (max_points);
                next_level_nodes_[k]->reg_error_ =
                    (next_level_nodes_[k]->reg_error_ / static_cast<float> (next_level_nodes_[k]->overlap_)) * (ov);
                        //* (next_level_nodes_[k]->color_weight_ / static_cast<float> (next_level_nodes_[k]->overlap_));

                if(use_color_)
                {
                    next_level_nodes_[k]->reg_error_ *= (next_level_nodes_[k]->color_weight_ / static_cast<float> (next_level_nodes_[k]->overlap_));
                }

                next_level_nodes_[k]->osv_fraction_ = 0;
                next_level_nodes_[k]->fsv_fraction_ = 0;
                next_level_nodes_[k]->overlap_ = ov;
              }
              else
              {
                next_level_nodes_[k]->reg_error_ = -1.f;
              }
            }
        }
      }

      //std::cout << "computation of fitness done...:" << next_level_nodes_.size () << std::endl;

      CorrespondencesPtr best_correspondences;
      bool improvement_in_iteration = false;
      int selected_cluster = 0;

      //Some of the best nodes survive...
      if (!survival_of_the_fittest_)
      {
          //pcl::ScopeTime ttt("best nodes selection and filtering...");
          std::sort (next_level_nodes_.begin (), next_level_nodes_.end (),
                     boost::bind (&ICPNode::reg_error_, _1) > boost::bind (&ICPNode::reg_error_, _2));

          std::cout << "best registration error:" << best_registration_error << std::endl;

          for (size_t k = 0; k < next_level_nodes_.size (); k++)
          {
            if (next_level_nodes_[k]->reg_error_ > best_registration_error)
            {
              best_registration_error = next_level_nodes_[k]->reg_error_;
              best_overlap = next_level_nodes_[k]->overlap_;
              best_transformation_ = next_level_nodes_[k]->accum_transform_;
              selected_cluster = k;
              improvement_in_iteration = true;
            }
          }

          std::cout << "best registration error:" << best_registration_error << " " << static_cast<int>(improvement_in_iteration) << std::endl;

          std::vector<boost::shared_ptr<ICPNode> > nodes_survived;
          nodes_survived.reserve(next_level_nodes_.size());
          int surv = 0;
          for (size_t k = 0; k < next_level_nodes_.size (); k++)
          {
            if (next_level_nodes_[k]->reg_error_ > (best_registration_error * threshold_best_))
            {
              nodes_survived.push_back (next_level_nodes_[k]);
              surv++;
            }
          }

          nodes_survived.resize(surv);
          std::cout << "alive nodes:" << nodes_survived.size() << " " << nr_iterations_ << std::endl;

          std::vector<boost::shared_ptr<ICPNode> > nodes_survived_after_filtering;

          //filter hypotheses that are equal...
          for (size_t k = 0; k < nodes_survived.size (); k++)
          {
            bool f = filterHypothesesByPose (nodes_survived[k], nodes_survived_after_filtering, 0.02f);
            if (!f)
              nodes_survived_after_filtering.push_back (nodes_survived[k]);
          }

          nodes_survived_after_filtering.resize (std::min (static_cast<int> (nodes_survived_after_filtering.size ()), max_keep));
          //std::cout << "survived vs total vs after filter:" << nodes_survived.size() << " " << next_level_nodes_.size() << " " << nodes_survived_after_filtering.size() << std::endl;

          //visualizeOrigins(nodes_survived_after_filtering);
          for (size_t k = 0; k < nodes_survived_after_filtering.size (); k++)
            alive_nodes_.push_back (nodes_survived_after_filtering[k]);

          std::cout << "alive nodes:" << alive_nodes_.size() << std::endl;

        if (improvement_in_iteration)
        {
          final_transformation_ = best_transformation_;
          //std::cout << "Overlap best:" << best_overlap << " reg error:" << best_registration_error << std::endl;

          transformCloud (*input_, *input_transformed, next_level_nodes_[selected_cluster]->accum_transform_);
          best_correspondences.reset (new Correspondences (*next_level_nodes_[selected_cluster]->after_rej_correspondences_));

          /*std::cout << "iter:" << nr_iterations_ << std::endl;
          std::cout << "Overlap best:" << best_overlap << " reg error:" << best_registration_error << std::endl;
          std::cout << "Overlap current:" << next_level_nodes_[selected_cluster]->overlap_ << " reg error:" << next_level_nodes_[selected_cluster]->reg_error_ << std::endl;*/

#ifdef VIS
          {
            pcl::visualization::PointCloudColorHandlerCustom<PointSource> handler (input_transformed, 0, 255, 255);
            icp_vis.addPointCloud<PointTarget> (input_transformed, handler, "source_cloud", v1);
          }

          {
            pcl::visualization::PointCloudColorHandlerCustom<PointSource> handler (input_, 0, 255, 255);
            icp_vis.addPointCloud<PointTarget> (input_, handler, "source_cloud_first_iteration", v2);
          }

          drawCorrespondences (target_, input_transformed, target_, input_transformed, *best_correspondences, icp_vis, v3);

          //std::cout << "Selected cluster:" << selected_cluster << " size selected:" << best_correspondences->size () << std::endl;
          //std::cout << "iter:" << nr_iterations_ << std::endl;
          //std::cout << "Overlap best:" << best_overlap << " reg error:" << best_registration_error << std::endl;
          //std::cout << "Overlap current:" << next_level_nodes_[selected_cluster]->overlap_ << " reg error:" << next_level_nodes_[selected_cluster]->reg_error_ << std::endl;
          icp_vis.spin ();
          icp_vis.removePointCloud ("source_cloud");
          icp_vis.removePointCloud ("source_cloud_first_iteration");
          icp_vis.removeAllShapes (v3);
          icp_vis.removeAllPointClouds (v3);
#endif

#ifdef VIS_SURVIVE
        visualizeICPNodes(nodes_survived_after_filtering, survived_vis, "survived");
        survived_vis.removeAllPointClouds();
        survived_vis.removeAllShapes();
#endif

        }
        else
        {
          converged_branches_.push_back(next_level_nodes_[selected_cluster]);
          // Copy all the values
          output = *input_;
          // Transform the XYZ + normals
          transformPointCloud (*input_, output, final_transformation_);
          //return;
        }
      }
      else
      {
        //Best node only...
        for (size_t k = 0; k < next_level_nodes_.size (); k++)
        {
          if (next_level_nodes_[k]->reg_error_ > best_registration_error)
          {
            best_registration_error = next_level_nodes_[k]->reg_error_;
            best_transformation_ = next_level_nodes_[k]->accum_transform_;
            best_overlap = next_level_nodes_[k]->overlap_;
            best_correspondences.reset (new Correspondences (*next_level_nodes_[k]->after_rej_correspondences_));
            improvement_in_iteration = true;
            selected_cluster = k;
          }
        }

        if (improvement_in_iteration)
        {
          transformCloud (*input_, *input_transformed, next_level_nodes_[selected_cluster]->accum_transform_);
          final_transformation_ = next_level_nodes_[selected_cluster]->accum_transform_;

#ifdef VIS
          {
            pcl::visualization::PointCloudColorHandlerCustom<PointSource> handler (input_transformed, 0, 255, 255);
            icp_vis.addPointCloud<PointTarget> (input_transformed, handler, "source_cloud", v1);
          }

          {
            pcl::visualization::PointCloudColorHandlerCustom<PointSource> handler (input_, 0, 255, 255);
            icp_vis.addPointCloud<PointTarget> (input_, handler, "source_cloud_first_iteration", v2);
          }

          drawCorrespondences (target_, input_transformed, target_, input_transformed, *best_correspondences, icp_vis, v3);

          std::cout << "Selected cluster:" << selected_cluster << " size selected:" << best_correspondences->size () << std::endl;
          std::cout << "iter:" << nr_iterations_ << std::endl;
          std::cout << "Overlap:" << best_overlap << " reg error:" << best_registration_error << std::endl;
          icp_vis.spin ();
          icp_vis.removePointCloud ("source_cloud");
          icp_vis.removePointCloud ("source_cloud_first_iteration");
          icp_vis.removeAllShapes (v3);
          icp_vis.removeAllPointClouds (v3);
#endif

          alive_nodes_.push_back (next_level_nodes_[selected_cluster]);
        }
      } //end iteration

      ++nr_iterations_;
      converged_ = static_cast<bool> ((*convergence_criteria_));

    } while (!converged_);

    // Transform the input cloud using the final transformation
    PCL_DEBUG ("Transformation is:\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n\t%5f\t%5f\t%5f\t%5f\n",
        final_transformation_ (0, 0), final_transformation_ (0, 1), final_transformation_ (0, 2), final_transformation_ (0, 3),
        final_transformation_ (1, 0), final_transformation_ (1, 1), final_transformation_ (1, 2), final_transformation_ (1, 3),
        final_transformation_ (2, 0), final_transformation_ (2, 1), final_transformation_ (2, 2), final_transformation_ (2, 3),
        final_transformation_ (3, 0), final_transformation_ (3, 1), final_transformation_ (3, 2), final_transformation_ (3, 3));

    delete vx;
    delete vx_gpu;
    delete target_gpu;
    delete input_transformed_gpu;

    // Copy all the values
    output = *input_;
    // Transform the XYZ + normals
    transformPointCloud (*input_, output, final_transformation_);

    result_.clear();
    for(size_t i=0; i < alive_nodes_.size(); i++)
    {
      result_.push_back(std::make_pair(alive_nodes_[i]->reg_error_, alive_nodes_[i]->accum_transform_));
    }

    for(size_t i=0; i < converged_branches_.size(); i++)
    {
      result_.push_back(std::make_pair(converged_branches_[i]->reg_error_, converged_branches_[i]->accum_transform_));
    }

    std::sort (result_.begin (), result_.end (),
    boost::bind (&std::pair<float, Eigen::Matrix4f>::first, _1) > boost::bind (&std::pair<float, Eigen::Matrix4f>::first, _2));

    //remove similar poses
    std::vector<Eigen::Matrix4f> accepted_so_far;
    std::vector<int> unique_results;
    for(size_t i=0; i < result_.size(); i++)
    {
        if(accepted_so_far.size() == 0)
        {
            accepted_so_far.push_back(result_[i].second);
            unique_results.push_back((int)i);
        }
        else
        {
            bool f = filterHypothesesByPose(result_[i].second, accepted_so_far, 0.01, 5.f);
            if(!f)
            {
                accepted_so_far.push_back(result_[i].second);
                unique_results.push_back((int)i);
            }
        }
    }

    for(size_t i=0; i < unique_results.size(); i++)
        result_[i] = result_[unique_results[i]];

    result_.resize(unique_results.size());

    if(VIS_FINAL_)
    {

      pcl::visualization::PCLVisualizer icp_alive("alive nodes at last level...");
      visualizeICPNodes(alive_nodes_, icp_alive, "alive nodes at last level...");
      icp_alive.removeAllPointClouds();
      icp_alive.close();
      icp_alive.spinOnce();
    }
  }

///////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget, typename Scalar>
  void
  faat_pcl::IterativeClosestPointWithGCWithNormals<PointSource, PointTarget, Scalar>::transformCloud (const PointCloudSource &input,
                                                                                                      PointCloudSource &output,
                                                                                                      const Matrix4 &transform)
  {
    pcl::transformPointCloudWithNormals (input, output, transform);
  }

#endif /* PCL_REGISTRATION_IMPL_ICP_HPP_ */
