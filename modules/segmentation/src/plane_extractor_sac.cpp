#include <v4r/segmentation/plane_extractor_sac.h>

#include <pcl/impl/instantiate.hpp>
#include <glog/logging.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/kdtree/kdtree.h>

namespace v4r
{

template<typename PointT>
void
SACPlaneExtractor<PointT>::compute()
{
    CHECK ( cloud_  ) << "Input cloud is not organized!";

    all_planes_.clear();
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations ( param_.max_iterations_ );
    seg.setDistanceThreshold ( param_.distance_threshold_ );

    typename pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud( *cloud_, *filtered_cloud );

    do
    {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud ( filtered_cloud );
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        seg.segment (*inliers, *coefficients);

        if ( inliers->indices.size() < param_.min_num_plane_inliers_ )
            break;

        all_planes_.push_back( Eigen::Vector4f(coefficients->values[0], coefficients->values[1],
                coefficients->values[2], coefficients->values[3]) ) ;

        for( int idx : inliers->indices )
        {
            pcl::PointXYZ &p = filtered_cloud->points[ idx ];
            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
        }

    }while( param_.compute_all_planes_ );

}

#define PCL_INSTANTIATE_SACPlaneExtractor(T) template class V4R_EXPORTS SACPlaneExtractor<T>;
PCL_INSTANTIATE(SACPlaneExtractor, PCL_XYZ_POINT_TYPES )

}
