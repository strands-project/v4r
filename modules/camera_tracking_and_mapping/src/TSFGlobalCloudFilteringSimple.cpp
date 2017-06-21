/**
 * $Id$
 * 
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl
 *
 */


#include <v4r/camera_tracking_and_mapping/TSFGlobalCloudFilteringSimple.h>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/reconstruction/impl/projectPointToImage.hpp>
#include <v4r/common/convertImage.h>
#include "pcl/common/transforms.h"
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/octree/octree_pointcloud.h>
#include <v4r/camera_tracking_and_mapping/OctreeVoxelCentroidContainerXYZRGBNormal.hpp>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include "opencv2/highgui/highgui.hpp"



namespace v4r
{


using namespace std;



/************************************************************************************
 * Constructor/Destructor
 */
TSFGlobalCloudFilteringSimple::TSFGlobalCloudFilteringSimple(const Parameter &p)
  : base_transform(Eigen::Matrix4f::Identity()), bb_min(Eigen::Vector3f(-FLT_MAX,-FLT_MAX,-FLT_MAX)), bb_max(Eigen::Vector3f(FLT_MAX,FLT_MAX,FLT_MAX))
{ 
  setParameter(p);
}

TSFGlobalCloudFilteringSimple::~TSFGlobalCloudFilteringSimple()
{
}


/**
 * @brief TSFGlobalCloudFilteringSimple::getMask
 * @param sf_cloud
 * @param mask
 */
void TSFGlobalCloudFilteringSimple::getMask(const v4r::DataMatrix2D<v4r::Surfel> &sf_cloud, cv::Mat_<unsigned char> &mask)
{
  tmp_mask = cv::Mat_<unsigned char>::ones(sf_cloud.rows, sf_cloud.cols)*255;

  for (int v=0; v<sf_cloud.rows; v++)
  {
    for (int u=0; u<sf_cloud.cols; u++)
    {
      const v4r::Surfel &sf = sf_cloud(v,u);
      if (isnan(sf.pt[0]) || isnan(sf.pt[1]) || isnan(sf.pt[2]))
        tmp_mask(v,u) = 0;
    }
  }

  cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2*param.erosion_size + 1, 2*param.erosion_size+1 ), cv::Point( param.erosion_size, param.erosion_size ) );
  cv::erode( tmp_mask, mask, element );
//  cv::imshow("tmp_mask", tmp_mask);
//  cv::imshow("mask", mask);
//  cv::waitKey(0);
}

/**
 * @brief TSFGlobalCloudFilteringSimple::filterCluster
 * @param cloud
 * @param cloud
 */
void TSFGlobalCloudFilteringSimple::filterCluster(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud_filt)
{
  pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
  tree->setInputCloud (cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGBNormal> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (100);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud);
  ec.extract (cluster_indices);

  int idx=-1;
  int max=0;
  for (unsigned i=0; i<cluster_indices.size(); i++)
  {
    if (((int)cluster_indices[i].indices.size())>max)
    {
      max = cluster_indices[i].indices.size();
      idx = i;
    }
  }

  cloud_filt.clear();

  if (idx==-1)
    return;

  const pcl::PointCloud<pcl::PointXYZRGBNormal> &ref = *cloud;
  pcl::PointIndices &inds = cluster_indices[idx];
  cloud_filt.points.resize(inds.indices.size());
  for (unsigned i=0; i<inds.indices.size(); i++)
    cloud_filt.points[i] = ref.points[inds.indices[i]];
  cloud_filt.width = cloud_filt.points.size();
  cloud_filt.height = 1;
  cloud_filt.is_dense = true;
}


/***************************************************************************************/

/**
 * @brief TSFGlobalCloudFilteringSimple::getGlobalCloudFiltered
 * @param frames
 * @param cloud
 */
void TSFGlobalCloudFilteringSimple::getGlobalCloudFiltered(const std::vector< v4r::TSFFrame::Ptr > &frames, pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud)
{
  cloud.clear();
  Eigen::Matrix4f inv_pose;
  Eigen::Vector3f pt;
  cv::Point2f im_pt;
  Eigen::Matrix3f R, inv_R;
  Eigen::Vector3f t, inv_t;
  pcl::PointXYZRGBNormal pcl_pt;
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
  pcl::PointCloud<pcl::PointXYZRGBNormal> &ref = *tmp_cloud;
  cv::Mat_<unsigned char> mask;

  pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGBNormal,pcl::octree::OctreeVoxelCentroidContainerXYZRGBNormal<pcl::PointXYZRGBNormal> >::Ptr octree;
  typedef pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGBNormal,pcl::octree::OctreeVoxelCentroidContainerXYZRGBNormal<pcl::PointXYZRGBNormal> >::AlignedPointTVector AlignedPointXYZRGBNormalVector;
  boost::shared_ptr< AlignedPointXYZRGBNormalVector > oc_cloud;
  oc_cloud.reset(new AlignedPointXYZRGBNormalVector () );
  octree.reset(new pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGBNormal,pcl::octree::OctreeVoxelCentroidContainerXYZRGBNormal<pcl::PointXYZRGBNormal> >(param.voxel_size));
  octree->setResolution(param.voxel_size);

  double cos_rad_thr_delta_angle = cos(param.thr_delta_angle*M_PI/180.);

  for (unsigned i=0; i<frames.size(); i++)
  {
    ref.clear();
    const v4r::TSFFrame &frame = *frames[i];
    R = frame.delta_cloud_rgb_pose.topLeftCorner<3, 3>();
    t = frame.delta_cloud_rgb_pose.block<3,1>(0, 3);
    v4r::invPose(frame.pose*base_transform, inv_pose);
    inv_R = inv_pose.topLeftCorner<3, 3>();
    inv_t = inv_pose.block<3,1>(0, 3);

    getMask(frame.sf_cloud, mask);

    for (int v=0; v<frame.sf_cloud.rows; v++)
    {
      for (int u=0; u<frame.sf_cloud.cols; u++)
      {
        if (mask(v,u)<128)
          continue;
        const v4r::Surfel &sf = frame.sf_cloud(v,u);
        if (isnan(sf.pt[0]) || isnan(sf.n[0]))
          continue;
        pt = R*sf.pt + t;
        if (dist_coeffs.empty())
          v4r::projectPointToImage(&pt[0],&intrinsic(0,0),&im_pt.x);
        else v4r::projectPointToImage(&pt[0], &intrinsic(0,0), &dist_coeffs(0), &im_pt.x);
        if (im_pt.x>=0 && im_pt.y>=0 && im_pt.x<frame.sf_cloud.cols && im_pt.y<frame.sf_cloud.rows)
        {
          pcl_pt.getVector3fMap() = inv_R*sf.pt+inv_t;
          if (pcl_pt.x<bb_min[0] || pcl_pt.x>bb_max[0] || pcl_pt.y<bb_min[1] || pcl_pt.y>bb_max[1] || pcl_pt.z<bb_min[2] || pcl_pt.z>bb_max[2])
            continue;
          if (sf.weight>=param.thr_weight && sf.n.dot(-sf.pt.normalized()) > cos_rad_thr_delta_angle )
          {
            pcl_pt.getNormalVector3fMap() = inv_R*sf.n;
            getInterpolatedRGB(frame.sf_cloud, im_pt, pcl_pt.r, pcl_pt.g, pcl_pt.b);
            ref.push_back(pcl_pt);
          }
        }
      }
    }
    ref.width = ref.points.size();
    ref.height = 1;
    ref.is_dense = true;
    octree->setInputCloud(tmp_cloud);
    octree->addPointsFromInputCloud();
  }

  // set point cloud
  octree->getVoxelCentroids(*oc_cloud);
  const AlignedPointXYZRGBNormalVector &oc = *oc_cloud;
  ref.resize(oc.size());
  for (unsigned i=0; i<oc.size(); i++)
      ref.points[i] = oc[i];
  ref.height=1;
  ref.width=ref.points.size();
  ref.is_dense = true;

  //filter largest custer
  if (param.filter_largest_cluster) filterCluster(tmp_cloud, cloud);
  else pcl::copyPointCloud(*tmp_cloud, cloud);
}

/**
 * @brief TSFGlobalCloudFilteringSimple::getMesh
 * @param cloud
 * @param mesh
 */
void TSFGlobalCloudFilteringSimple::getMesh(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, pcl::PolygonMesh &mesh)
{
  PoissonTriangulation poisson(param.poisson_depth, param.samples_per_node, param.crop_mesh);
  poisson.reconstruct(cloud, mesh);
}



/**
 * @brief TSFGlobalCloudFilteringSimple::setBaseTransform
 * @param transform
 */
void TSFGlobalCloudFilteringSimple::setBaseTransform(const Eigen::Matrix4f &transform)
{
  base_transform = transform;
}

/**
 * @brief TSFGlobalCloudFilteringSimple::setROI
 * @param bb_lowerleft
 * @param bb_upper_right
 */
void TSFGlobalCloudFilteringSimple::setROI(const Eigen::Vector3f &bb_lowerleft, const Eigen::Vector3f &bb_upperright)
{
  bb_min = bb_lowerleft;
  bb_max = bb_upperright;
}

/**
 * @brief TSFGlobalCloudFilteringSimple::setParameter
 * @param p
 */
void TSFGlobalCloudFilteringSimple::setParameter(const Parameter &p)
{
  param = p;
}


/**
 * @brief TSFGlobalCloudFilteringSimple::setCameraParameter
 * @param _intrinsic
 * @param _dist_coeffs
 */
void TSFGlobalCloudFilteringSimple::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  dist_coeffs = cv::Mat_<double>();
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;
  if (!_dist_coeffs.empty())
  {
    dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols*_dist_coeffs.rows; i++)
      dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }
}


}












