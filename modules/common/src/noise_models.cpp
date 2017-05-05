#include <pcl_1_8/features/organized_edge_detection.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/pcl_opencv.h>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <omp.h>

namespace v4r
{


template<typename PointT>
bool
NguyenNoiseModel<PointT>::computeNoiseLevel(const PointT &pt, const pcl::Normal &n, float &sigma_lateral, float &sigma_axial, float focal_length)
{
    const Eigen::Vector3f &np = n.getNormalVector3fMap();

     if( !pcl::isFinite(pt) || !pcl::isFinite(n) )
     {
         sigma_lateral = sigma_axial = std::numeric_limits<float>::max();
         return false;
     }

     //origin to pint
     //Eigen::Vector3f o2p = input_->points[i].getVector3fMap() * -1.f;
     Eigen::Vector3f o2p = Eigen::Vector3f::UnitZ() * -1.f;
     o2p.normalize();
     float angle = pcl::rad2deg(acos(o2p.dot(np)));

     if (angle > 85.f)
         angle = 85.f;

     float sigma_lateral_px = (0.8f + 0.034f * angle / (90.f - angle)) * pt.z / focal_length; // in pixel
     sigma_lateral = sigma_lateral_px * pt.z * 1.f; // in metres
     sigma_axial = 0.0012f + 0.0019f * ( pt.z - 0.4f ) * ( pt.z - 0.4f ) + 0.0001f * angle * angle / ( sqrt(pt.z) * (90.f - angle) * (90.f - angle));  // in metres

     return true;
}


template<typename PointT>
void
NguyenNoiseModel<PointT>::compute ()
{
    CHECK( input_->isOrganized() );
    pt_properties_.resize(input_->points.size(), std::vector<float>(3));

#pragma omp parallel for schedule(dynamic)
    for(size_t i=0; i < input_->points.size(); i++)
    {
        const PointT &pt = input_->points[i];
        const pcl::Normal &n = normals_->points[i];
        computeNoiseLevel( pt, n, pt_properties_[i][0], pt_properties_[i][1], param_.focal_length_);
        pt_properties_[i][2] = std::numeric_limits<float>::max();
   }

    //compute distance (in pixels) to edge for each pixel
    if (param_.use_depth_edges_)
    {
        //compute depth discontinuity edges
        pcl_1_8::OrganizedEdgeBase<PointT, pcl::Label> oed;
        oed.setDepthDisconThreshold (0.05f); //at 1m, adapted linearly with depth
        oed.setMaxSearchNeighbors(100);
        oed.setEdgeType (  pcl_1_8::OrganizedEdgeBase<PointT,           pcl::Label>::EDGELABEL_OCCLUDING
                         | pcl_1_8::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDED
                         | pcl_1_8::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_NAN_BOUNDARY
                         );
        oed.setInputCloud (input_);

        pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
        std::vector<pcl::PointIndices> edge_indices;
        oed.compute (*labels, edge_indices);

        cv::Mat_<uchar> pixel_is_edge (input_->height, input_->width); // depth edges
        pixel_is_edge.setTo(255);

        for (size_t j = 0; j < edge_indices.size (); j++)
        {
            for (int edge_px_id : edge_indices[j].indices)
            {
                int row = edge_px_id / input_->width;
                int col = edge_px_id % input_->width;
                pixel_is_edge.at<uchar>(row,col) = 0;
            }
        }

        cv::Mat img_boundary_distance;
        cv::distanceTransform(pixel_is_edge, img_boundary_distance, CV_DIST_L2, 5);

        for (size_t idx = 0; idx < input_->points.size (); idx++)
        {
            int row = idx / input_->width;
            int col = idx % input_->width;
            pt_properties_[idx][2] = img_boundary_distance.at<float>(row, col);
        }
    }
}

template class V4R_EXPORTS NguyenNoiseModel<pcl::PointXYZRGB>;
//template class V4R_EXPORTS NguyenNoiseModel<pcl::PointXYZ>;
}
