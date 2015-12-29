/*
 * noise_models.cpp
 *
 *  Created on: Oct 28, 2013
 *      Author: aitor
 */

#include <pcl/common/angles.h>
#include <v4r/common/organized_edge_detection.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "v4r/common/noise_models.h"
#include "v4r/common/organized_edge_detection.h"

#include <opencv2/opencv.hpp>
#include <v4r/common/pcl_opencv.h>
#include <fstream>

namespace v4r {

template<typename PointT>
noise_models::NguyenNoiseModel<PointT>::NguyenNoiseModel (const Parameter &param)
{
    param_ = param;
    pose_set_ = false;
    pose_to_plane_RF_ = Eigen::Matrix4f::Identity();
}

template<typename PointT>
void
noise_models::NguyenNoiseModel<PointT>::compute ()
{
    weights_.clear();
    sigmas_combined_.clear();
    sigmas_.clear();
    weights_.resize(input_->points.size(), 1.f);
    sigmas_combined_.resize(input_->points.size(), 0.f);
    sigmas_.resize(input_->points.size());
    discontinuity_edges_.indices.clear();

    //compute depth discontinuity edges
    OrganizedEdgeBase<PointT, pcl::Label> oed;
    oed.setDepthDisconThreshold (0.05f); //at 1m, adapted linearly with depth
    oed.setMaxSearchNeighbors(100);
    oed.setEdgeType (  OrganizedEdgeBase<PointT,           pcl::Label>::EDGELABEL_OCCLUDING
                     | OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDED
                     | OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_NAN_BOUNDARY
                     );
    oed.setInputCloud (input_);

    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
    std::vector<pcl::PointIndices> edge_indices;
    oed.compute (*labels, edge_indices);

    for (size_t j = 0; j < edge_indices.size (); j++)
    {
        for (size_t i = 0; i < edge_indices[j].indices.size (); i++)
            discontinuity_edges_.indices.push_back(edge_indices[j].indices[i]);
    }

    for(size_t i=0; i < input_->points.size(); i++)
    {
        float sigma_lateral = 0.f;
        float sigma_axial = 0.f;
        const PointT &pt = input_->points[i];
        const pcl::Normal &n = normals_->points[i];
        const Eigen::Vector3f & np = n.getNormalVector3fMap();

        sigmas_[i].resize(3, std::numeric_limits<float>::max());
        sigmas_combined_[i] = std::numeric_limits<float>::max();
        weights_[i] = 0;

        if( !pcl::isFinite(pt) || !pcl::isFinite(n) )
            continue;

        //origin to pint
        //Eigen::Vector3f o2p = input_->points[i].getVector3fMap() * -1.f;
        Eigen::Vector3f o2p = Eigen::Vector3f::UnitZ() * -1.f;

        o2p.normalize();
        float angle = pcl::rad2deg(acos(o2p.dot(np)));

        sigma_lateral = (0.8 + 0.034 * angle / (90.f - angle)) * pt.z / param_.focal_length_;
        sigma_axial = 0.0012 + 0.0019 * ( pt.z - 0.4 ) * ( pt.z - 0.4 ) + 0.0001 * angle * angle / ( sqrt(pt.z) * (90 - angle) * (90 - angle));

        sigmas_[i][0] = sigma_lateral;
        sigmas_[i][1] = sigma_axial;
        sigmas_combined_[i] = sqrt(sigma_lateral * sigma_lateral + sigma_lateral * sigma_lateral + sigma_axial * sigma_axial); // lateral is two-dimensional, axial only one dimension

        if(angle > param_.max_angle_)
        {
            weights_[i] = 1.f - (angle - param_.max_angle_) / (90.f - param_.max_angle_);
        }
        else
        {
            //weights_[i] = 1.f - 0.2f * ((std::max(angle, 30.f) - 30.f) / (max_angle_ - 30.f));
        }

        //std::cout << angle << " " << weights_[i] << std::endl;
        //weights_[i] = 1.f - ( angle )
    }

    //compute distance (in pixels) to edge for each pixel
    if (param_.use_depth_edges_)
    {
        std::vector<float> dist_to_edge_3d(input_->points.size(), std::numeric_limits<float>::infinity());
        std::vector<float> dist_to_edge_px(input_->points.size(), std::numeric_limits<float>::infinity());

        int wdw_size = 5;

        for (const auto &idx_start : discontinuity_edges_.indices) {
            dist_to_edge_3d[idx_start] = 0.f;
            dist_to_edge_px[idx_start] = 0.f;

            int row_start = idx_start / input_->width;
            int col_start = idx_start % input_->width;

            for (int row_k = (row_start - wdw_size); row_k <= (row_start + wdw_size); row_k++)
            {
                for (int col_k = (col_start - wdw_size); col_k <= (col_start + wdw_size); col_k++)
                {
                    if( col_k<0 || row_k < 0 || col_k >= input_->width || row_k >= input_->height || row_k == row_start || col_k == col_start)
                        continue;

                    int idx_k = row_k * input_->width + col_k;

                    float dist_3d = dist_to_edge_3d[idx_start] + (input_->points[idx_start].getVector3fMap () - input_->points[idx_k].getVector3fMap ()).norm ();
                    float dist_px = dist_to_edge_px[idx_start] + sqrt( (col_k-col_start)*(col_k-col_start) + (row_k-row_start)*(row_k-row_start));

                    if( dist_px < dist_to_edge_px[idx_k] )
                        dist_to_edge_px[idx_k] = dist_px;

                    if( dist_3d < dist_to_edge_3d[idx_k] )
                        dist_to_edge_3d[idx_k] = dist_3d;
                }
            }
        }

//        std::ofstream f ("/tmp/test.txt");
        for (int i = 0; i < input_->points.size (); i++) {
            sigmas_[i][2] = dist_to_edge_px[i];
//            f << dist_to_edge_px[i] << " ";
        }
//        f.close();
    }

//    for(size_t i=0; i < input_->points.size(); i++)
//    {
//        if(weights_[i] < 0.f)
//            weights_[i] = 0.f;

//        else
//        {
//            if(pose_set_)
//            {
//                Eigen::Vector4f p = input_->points[i].getVector4fMap();
//                p = pose_to_plane_RF_ * p;
//                weights_[i] *= 1.f - 0.25f * std::max(0.f, (0.01f - p[2]) / 0.01f);
//                //std::cout << p[2] << " " << 0.25f * std::max(0.f, 0.01f - p[2]) << std::endl;
//            }
//        }
//    }
}

/*template<typename PointT>
void
noise_models::NguyenNoiseModel<PointT>::getFilteredCloud(PointTPtr & filtered, float w_t)
{
  Eigen::Vector3f nan3f(std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN());
  filtered.reset(new pcl::PointCloud<PointT>(*input_));
  for(size_t i=0; i < input_->points.size(); i++)
  {
    if(weights_[i] < w_t)
    {
      //filtered->points[i].getVector3fMap() = nan3f;
      filtered->points[i].r = 255;
      filtered->points[i].g = 0;
      filtered->points[i].b = 0;
    }

    if(!pcl_isfinite( input_->points[i].z))
    {
      filtered->points[i].r = 255;
      filtered->points[i].g = 255;
      filtered->points[i].b = 0;
    }
  }
}*/

template<typename PointT>
void
noise_models::NguyenNoiseModel<PointT>::getFilteredCloudRemovingPoints(PointTPtr & filtered, float w_t)
{
    Eigen::Vector3f nan3f(std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::quiet_NaN());

    filtered.reset(new pcl::PointCloud<PointT>(*input_));
    for(size_t i=0; i < input_->points.size(); i++)
    {
        if(weights_[i] < w_t)
        {
            filtered->points[i].x = std::numeric_limits<float>::quiet_NaN();
            filtered->points[i].y = std::numeric_limits<float>::quiet_NaN();
            filtered->points[i].z = std::numeric_limits<float>::quiet_NaN();
        }
    }
}

template<typename PointT>
void
noise_models::NguyenNoiseModel<PointT>:: getFilteredCloudRemovingPoints(PointTPtr & filtered, float w_t, std::vector<int> & kept)
{
    Eigen::Vector3f nan3f(std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::quiet_NaN());

    filtered.reset(new pcl::PointCloud<PointT>(*input_));
    for(size_t i=0; i < input_->points.size(); i++)
    {
        if(weights_[i] < w_t)
        {
            filtered->points[i].getVector3fMap() = nan3f;
        }
        else
        {
            kept.push_back(i);
        }
    }
}

template class V4R_EXPORTS noise_models::NguyenNoiseModel<pcl::PointXYZRGB>;
//template class V4R_EXPORTS noise_models::NguyenNoiseModel<pcl::PointXYZ>;

}
