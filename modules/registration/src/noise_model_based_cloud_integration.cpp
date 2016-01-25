/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma, Thomas Faeulhammer
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

#include <pcl/common/angles.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <v4r/registration/noise_model_based_cloud_integration.h>
#include <v4r/common/organized_edge_detection.h>

#include <glog/logging.h>

namespace v4r
{

template<typename PointT>
void
NMBasedCloudIntegration<PointT>::collectInfo ()
{
  size_t total_point_count = 0;
  for(size_t i = 0; i < input_clouds_.size(); i++)
    total_point_count += indices_.empty() ? input_clouds_[i]->size() : indices_[i].size();
  big_cloud_info_.resize(total_point_count);

  size_t point_count = 0;
  for(size_t i=0; i < input_clouds_.size(); i++)
  {
    pcl::PointCloud<PointT> cloud_aligned;
    pcl::transformPointCloud(*input_clouds_[i], cloud_aligned, transformations_to_global_[i]);
    pcl::PointCloud<pcl::Normal> normals_aligned;
    transformNormals(*input_normals_[i], normals_aligned, transformations_to_global_[i]);

    size_t kept_new_pts = 0;

    if (indices_.empty())
    {
      for(size_t jj=0; jj<cloud_aligned.points.size(); jj++)
      {
        if ( !pcl::isFinite(cloud_aligned.points[jj]))
          continue;

        PointInfo &pt = big_cloud_info_[point_count + kept_new_pts];
        pt.pt = cloud_aligned.points[jj];
        pt.normal = normals_aligned.points[jj];
        pt.sigma_lateral = pt_properties_[i][jj][0];
        pt.sigma_axial = pt_properties_[i][jj][1];
        pt.distance_to_depth_discontinuity = pt_properties_[i][jj][2];
        kept_new_pts++;
      }
    }
    else
    {
      for(const auto idx : indices_[i])
      {
        if(!pcl::isFinite(cloud_aligned.points[idx]))
          continue;

        PointInfo &pt = big_cloud_info_[point_count + kept_new_pts];
        pt.pt = cloud_aligned.points[idx];
        pt.normal = normals_aligned.points[ idx ];
        pt.sigma_lateral = pt_properties_[i][idx][0];
        pt.sigma_axial = pt_properties_[i][idx][1];
        pt.distance_to_depth_discontinuity = pt_properties_[i][idx][2];
        kept_new_pts++;
      }
    }

    // compute and store remaining information
    for(size_t jj=0; jj<kept_new_pts; jj++)
    {
      PointInfo &pt = big_cloud_info_ [point_count + jj];
      pt.origin = i;

      Eigen::Matrix3f sigma = Eigen::Matrix3f::Zero(), sigma_aligned = Eigen::Matrix3f::Zero();
      sigma(0,0) = pt.sigma_lateral;
      sigma(1,1) = pt.sigma_lateral;
      sigma(2,2) = pt.sigma_axial;

      const Eigen::Matrix4f &tf = transformations_to_global_[ i ];
      Eigen::Matrix3f rotation = tf.block<3,3>(0,0); // or inverse?
      sigma_aligned = rotation * sigma * rotation.transpose();

      pt.probability = 1/ sqrt(2 * M_PI * sigma_aligned.determinant());
    }

    point_count += kept_new_pts;
  }

  big_cloud_info_.resize(point_count);
}

template<typename PointT>
void
NMBasedCloudIntegration<PointT>::reasonAboutPts ()
{
    const int width = input_clouds_[0]->width;
    const int height = input_clouds_[0]->height;
    const float cx = static_cast<float> (width) / 2.f;// - 0.5f;
    const float cy = static_cast<float> (height) / 2.f;// - 0.5f;

//    pcl::visualization::PCLVisualizer vis;
    for (size_t i=0; i<big_cloud_info_.size(); i++)
    {
        PointInfo &pt = big_cloud_info_[i];
        const PointT &ptt = pt.pt;

        for (size_t cloud=0; cloud<input_clouds_.size(); cloud++)
        {
            if( pt.origin == cloud)  // we don't have to reason about the point with respect to the original cloud
                continue;

            // reproject point onto the cloud's image plane and check if its within FOV and if so, if it can be seen or is occluded
            const Eigen::Matrix4f &tf = transformations_to_global_[cloud].inverse() * transformations_to_global_[pt.origin];
            float x = static_cast<float> (tf (0, 0) * ptt.x + tf (0, 1) * ptt.y + tf (0, 2) * ptt.z + tf (0, 3));
            float y = static_cast<float> (tf (1, 0) * ptt.x + tf (1, 1) * ptt.y + tf (1, 2) * ptt.z + tf (1, 3));
            float z = static_cast<float> (tf (2, 0) * ptt.x + tf (2, 1) * ptt.y + tf (2, 2) * ptt.z + tf (2, 3));

            int u = static_cast<int> (param_.focal_length_ * x / z + cx);
            int v = static_cast<int> (param_.focal_length_ * y / z + cy);

            std::cout << "u: " <<u << ", v: " << v << std::endl;

            PointT ptt_aligned;
            ptt_aligned.x = x;
            ptt_aligned.y = y;
            ptt_aligned.z = z;


            if( u<0 || v <0 || u>=width || v >= height )
                pt.occluded_++;
            else
            {
                float thresh = param_.threshold_explained_;
                if( z > 1.f )
                    thresh+= param_.threshold_explained_ * (z-1.f) * (z-1.f);

               const float z_c = input_clouds_[cloud]->points[ v*width + u ].z;
               if ( std::abs(z_c - z) < thresh )
                   pt.explained_++;
               else if (z_c > z )
               {
                   pt.violated_ ++;

//                   vis.removeAllShapes();
//                   vis.removeAllPointClouds();
//                   vis.addPointCloud(input_clouds_[cloud]);
//                   vis.addSphere(ptt_aligned, 0.03f, 1.f, 0.f, 0.f);
//                   vis.spin();
               }
               else
                   pt.occluded_ ++;
            }

        }
    }
}

template<typename PointT>
void
NMBasedCloudIntegration<PointT>::compute (PointTPtr & output)
{
    if(input_clouds_.empty()) {
        std::cerr << "No input clouds set for cloud integration!" << std::endl;
        return;
    }

    big_cloud_info_.clear();

    collectInfo();

    if(param_.reason_about_points_)
        reasonAboutPts();

    pcl::octree::OctreePointCloudPointVector<PointT> octree( param_.octree_resolution_ );
    PointTPtr big_cloud ( new pcl::PointCloud<PointT>());
    big_cloud->width = big_cloud_info_.size();
    big_cloud->height = 1;
    big_cloud->points.resize( big_cloud_info_.size() );
    for(size_t i=0; i < big_cloud_info_.size(); i++)
        big_cloud->points[i] = big_cloud_info_[i].pt;
    octree.setInputCloud( big_cloud );
    octree.addPointsFromInputCloud();

    typename pcl::octree::OctreePointCloudPointVector<PointT>::LeafNodeIterator leaf_it;
    const typename pcl::octree::OctreePointCloudPointVector<PointT>::LeafNodeIterator it2_end = octree.leaf_end();

    output->points.resize( big_cloud_info_.size() );
    output_normals_.reset(new pcl::PointCloud<pcl::Normal>);
    output_normals_->points.resize( big_cloud_info_.size());

    size_t kept = 0;
    size_t total_used = 0;

    for (leaf_it = octree.leaf_begin(); leaf_it != it2_end; ++leaf_it)
    {
        pcl::octree::OctreeContainerPointIndices& container = leaf_it.getLeafContainer();

        // add points from leaf node to indexVector
        std::vector<int> indexVector;
        container.getPointIndices (indexVector);

        if(indexVector.empty() || indexVector.size() < param_.min_points_per_voxel_)
            continue;

        std::vector<PointInfo> voxel_pts ( indexVector.size() );

        for(size_t k=0; k < indexVector.size(); k++)
            voxel_pts[k] = big_cloud_info_ [indexVector[k]];

        PointT p;
        pcl::Normal n;

        if(param_.average_)
        {
            p.getVector3fMap() = Eigen::Vector3f::Zero();
            p.r = p.g = p.b = 0.f;
            n.getNormalVector3fMap() = Eigen::Vector3f::Zero();
            n.curvature = 0.f;

            for(const PointInfo &pt_tmp : voxel_pts)
            {
                p.getVector3fMap() = p.getVector3fMap() +  pt_tmp.pt.getVector3fMap();
                p.r += pt_tmp.pt.r;
                p.g += pt_tmp.pt.g;
                p.b += pt_tmp.pt.b;

                Eigen::Vector3f normal = pt_tmp.normal.getNormalVector3fMap();
                normal.normalize();
                n.getNormalVector3fMap() = n.getNormalVector3fMap() + normal;
                n.curvature += pt_tmp.normal.curvature;
            }

            p.getVector3fMap() = p.getVector3fMap() / indexVector.size();
            p.r /= indexVector.size();
            p.g /= indexVector.size();
            p.b /= indexVector.size();

            n.getNormalVector3fMap() = n.getNormalVector3fMap() / indexVector.size();
            n.curvature /= indexVector.size();

            total_used += indexVector.size();
        }
        else // take only point with max probability
        {
            std::sort(voxel_pts.begin(), voxel_pts.end());

            bool found_good_pt = false;

            for(const PointInfo &pt_tmp : voxel_pts)
            {
                if (pt_tmp.distance_to_depth_discontinuity > param_.edge_radius_px_)
                {
                    p.getVector3fMap() = pt_tmp.pt.getVector3fMap();
                    p.r = pt_tmp.pt.r;
                    p.g = pt_tmp.pt.g;
                    p.b = pt_tmp.pt.b;

                    n.getNormalVector3fMap() = pt_tmp.normal.getNormalVector3fMap();
                    n.curvature = pt_tmp.normal.curvature;
                    found_good_pt = true;
                    break;
                }
            }

            if( !found_good_pt )
                continue;

            total_used++;
        }

        output->points[kept] = p;
        output_normals_->points[kept] = n;
        kept++;
    }

    std::cout << "Number of points in final model:" << kept << " used:" << total_used << std::endl;

    output->points.resize(kept);
    output_normals_->points.resize(kept);
    output->width = output_normals_->width = kept;
    output->height = output_normals_->height = 1;
    output->is_dense = output_normals_->is_dense = true;

    cleanUp();
}

template class V4R_EXPORTS NMBasedCloudIntegration<pcl::PointXYZRGB>;
}
