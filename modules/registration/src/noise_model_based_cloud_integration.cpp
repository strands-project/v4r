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
#include <v4r/registration/noise_model_based_cloud_integration.h>

#include <glog/logging.h>
#include <omp.h>

namespace v4r
{

template<typename PointT>
void
NMBasedCloudIntegration<PointT>::collectInfo ()
{
  size_t total_point_count = 0;
  for(size_t i = 0; i < input_clouds_.size(); i++)
    total_point_count += (indices_.empty() || indices_[i].empty()) ? input_clouds_[i]->size() : indices_[i].size();
  big_cloud_info_.resize(total_point_count);

  std::vector<pcl::PointCloud<PointT> > input_clouds_aligned (input_clouds_.size());
  std::vector<pcl::PointCloud<pcl::Normal> > input_normals_aligned (input_clouds_.size());

#pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i < input_clouds_.size(); i++)
  {
      pcl::transformPointCloud(*input_clouds_[i], input_clouds_aligned[i], transformations_to_global_[i]);
      transformNormals(*input_normals_[i], input_normals_aligned[i], transformations_to_global_[i]);
  }

  size_t point_count = 0;
  for(size_t i=0; i < input_clouds_.size(); i++)
  {
    const pcl::PointCloud<PointT> &cloud_aligned = input_clouds_aligned[i];
    const pcl::PointCloud<pcl::Normal> &normals_aligned = input_normals_aligned[i];

    size_t kept_new_pts = 0;
    if (indices_.empty() || indices_[i].empty())
    {
      for(size_t jj=0; jj<cloud_aligned.points.size(); jj++)
      {
        if ( !pcl::isFinite(cloud_aligned.points[jj]) || !pcl::isFinite(normals_aligned.points[jj]) )
          continue;

        PointInfo &pt = big_cloud_info_[point_count + kept_new_pts];
        pt.pt = cloud_aligned.points[jj];
        pt.normal = normals_aligned.points[jj];
        pt.sigma_lateral = pt_properties_[i][jj][0];
        pt.sigma_axial = pt_properties_[i][jj][1];
        pt.distance_to_depth_discontinuity = pt_properties_[i][jj][2];
        pt.pt_idx = jj;
        kept_new_pts++;
      }
    }
    else
    {
      for(const auto idx : indices_[i])
      {
        if ( !pcl::isFinite(cloud_aligned.points[idx]) || !pcl::isFinite(normals_aligned.points[idx]) )
           continue;

        PointInfo &pt = big_cloud_info_[point_count + kept_new_pts];
        pt.pt = cloud_aligned.points[idx];
        pt.normal = normals_aligned.points[ idx ];
        pt.sigma_lateral = pt_properties_[i][idx][0];
        pt.sigma_axial = pt_properties_[i][idx][1];
        pt.distance_to_depth_discontinuity = pt_properties_[i][idx][2];
        pt.pt_idx = idx;
        kept_new_pts++;
      }
    }

    // compute and store remaining information
#pragma omp parallel for schedule (dynamic) firstprivate(i, point_count, kept_new_pts)
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
      double det = sigma_aligned.determinant();

//      if( std::isfinite(det) && det>0)
//          pt.probability = 1 / sqrt(2 * M_PI * det);
//      else
//          pt.probability = std::numeric_limits<float>::min();

      if( std::isfinite(det) && det>0)
          pt.weight = det;
      else
          pt.weight = std::numeric_limits<float>::max();
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

    size_t kept = 0;
    size_t total_used = 0;

    std::vector<PointInfo> filtered_cloud_info ( big_cloud_info_.size() );

    for (leaf_it = octree.leaf_begin(); leaf_it != it2_end; ++leaf_it)
    {
        pcl::octree::OctreeContainerPointIndices& container = leaf_it.getLeafContainer();

        // add points from leaf node to indexVector
        std::vector<int> indexVector;
        container.getPointIndices (indexVector);

        if(indexVector.empty())
            continue;

        std::vector<PointInfo> voxel_pts ( indexVector.size() );

        for(size_t k=0; k < indexVector.size(); k++)
            voxel_pts[k] = big_cloud_info_ [indexVector[k]];

        PointInfo p;

        size_t num_good_pts = 0;
        if(param_.average_)
        {
            for(const PointInfo &pt_tmp : voxel_pts)
            {
                if (pt_tmp.distance_to_depth_discontinuity > param_.edge_radius_px_)
                {
                    p.moving_average( pt_tmp );
                    num_good_pts++;
                }
            }

            if( !num_good_pts || num_good_pts < param_.min_points_per_voxel_ )
                continue;

            total_used += num_good_pts;
        }
        else // take only point with min weight
        {
            for(const PointInfo &pt_tmp : voxel_pts)
            {
                if ( pt_tmp.distance_to_depth_discontinuity > param_.edge_radius_px_)
                {
                    num_good_pts++;
                    if ( pt_tmp.weight < p.weight || num_good_pts == 1)
                        p = pt_tmp;
                }
            }

            if( !num_good_pts || num_good_pts < param_.min_points_per_voxel_ )
                continue;

            total_used++;
        }
        filtered_cloud_info[kept++] = p;
    }

    std::cout << "Number of points in final model:" << kept << " used:" << total_used << std::endl;


    if(!output)
        output.reset(new pcl::PointCloud<PointT>);

    if(!output_normals_)
        output_normals_.reset( new pcl::PointCloud<pcl::Normal>);

    filtered_cloud_info.resize(kept);
    output->points.resize(kept);
    output_normals_->points.resize(kept);
    output->width = output_normals_->width = kept;
    output->height = output_normals_->height = 1;
    output->is_dense = output_normals_->is_dense = true;

    PointT na;
    na.x = na.y = na.z = std::numeric_limits<float>::quiet_NaN();

    input_clouds_used_.resize( input_clouds_.size() );
    for(size_t i=0; i<input_clouds_used_.size(); i++) {
        input_clouds_used_[i].reset( new pcl::PointCloud<PointT> );
        input_clouds_used_[i]->points.resize( input_clouds_[i]->points.size(), na);
        input_clouds_used_[i]->width =  input_clouds_[i]->width;
        input_clouds_used_[i]->height =  input_clouds_[i]->height;
    }

    for(size_t i=0; i<kept; i++) {
        output->points[i] = filtered_cloud_info[i].pt;
        output_normals_->points[i] = filtered_cloud_info[i].normal;
        int origin = filtered_cloud_info[i].origin;
        input_clouds_used_[origin]->points[filtered_cloud_info[i].pt_idx] = filtered_cloud_info[i].pt;
    }

    cleanUp();
}

template class V4R_EXPORTS NMBasedCloudIntegration<pcl::PointXYZRGB>;
}
