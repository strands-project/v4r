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
NguyenNoiseModel<PointT>::NguyenNoiseModel (const Parameter &param)
{
    param_ = param;
}

template<typename PointT>
void
NguyenNoiseModel<PointT>::compute ()
{
    pt_properties_.resize(input_->points.size());
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

//    pcl::visualization::PCLVisualizer vis;
//    vis.addPointCloud(input_);

#pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i < input_->points.size(); i++)
    {
        float sigma_lateral = 0.f;
        float sigma_axial = 0.f;
        const PointT &pt = input_->points[i];
        const pcl::Normal &n = normals_->points[i];
        const Eigen::Vector3f & np = n.getNormalVector3fMap();

        pt_properties_[i].resize(3, std::numeric_limits<float>::max());

        if( !pcl::isFinite(pt) || !pcl::isFinite(n) )
            continue;

        //origin to pint
        //Eigen::Vector3f o2p = input_->points[i].getVector3fMap() * -1.f;
        Eigen::Vector3f o2p = Eigen::Vector3f::UnitZ() * -1.f;

        o2p.normalize();
        float angle = pcl::rad2deg(acos(o2p.dot(np)));

        sigma_lateral = (0.8 + 0.034 * angle / (90.f - angle)) * pt.z / param_.focal_length_;
        sigma_axial = 0.0012 + 0.0019 * ( pt.z - 0.4 ) * ( pt.z - 0.4 ) + 0.0001 * angle * angle / ( sqrt(pt.z) * (90 - angle) * (90 - angle));

//        std::cout << "angle: " << angle << ", sigma_lateral: " << sigma_lateral << ", sigma axial: " << sigma_axial << std::endl;
//        vis.removeAllShapes();
//        vis.addSphere(input_->points[i], 0.03f, 1,0,0);
//        vis.spin();
        pt_properties_[i][0] = sigma_lateral;
        pt_properties_[i][1] = sigma_axial;
   }

    //compute distance (in pixels) to edge for each pixel
    if (param_.use_depth_edges_)
    {
        std::vector<float> dist_to_edge_3d(input_->points.size(), std::numeric_limits<float>::infinity());
        std::vector<float> dist_to_edge_px(input_->points.size(), std::numeric_limits<float>::infinity());

#pragma omp parallel for schedule(dynamic)
        for (size_t i=0; i <discontinuity_edges_.indices.size(); i++) {
            const int &idx_start = discontinuity_edges_.indices[i];
            dist_to_edge_3d[idx_start] = 0.f;
            dist_to_edge_px[idx_start] = 0.f;

            int row_start = idx_start / input_->width;
            int col_start = idx_start % input_->width;

            for (int row_k = (row_start - param_.edge_radius_); row_k <= (row_start + param_.edge_radius_); row_k++)
            {
                for (int col_k = (col_start - param_.edge_radius_); col_k <= (col_start + param_.edge_radius_); col_k++)
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
            pt_properties_[i][2] = dist_to_edge_px[i];
//            f << dist_to_edge_px[i] << std::endl;
        }
//        f.close();
    }
}

template class V4R_EXPORTS NguyenNoiseModel<pcl::PointXYZRGB>;
//template class V4R_EXPORTS NguyenNoiseModel<pcl::PointXYZ>;
}
