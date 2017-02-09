#include <v4r/segmentation/euclidean_segmenter.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/kdtree/kdtree.h>

namespace v4r
{

template <typename PointT>
void
EuclideanSegmenter<PointT>::computeTablePlanes()
{
    // Create the segmentation object for the planar model and set all the parameters
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (param_.sensor_noise_max_);

    while(true)
    {
        pcl::copyPointCloud(*scene_xyz_, *cloud_filtered);
        for(size_t i=0; i<filter_mask_.size(); i++)
        {
            if( !filter_mask_[i] )
            {
                pcl::PointXYZ &p = cloud_filtered->points[i];
                p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
            }
        }

        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);

        if ( (int)inliers->indices.size() < param_.num_plane_inliers_ )
            break;

        typename PlaneModel<PointT>::Ptr pm (new PlaneModel<PointT>);
        pm->coefficients_ = Eigen::Vector4f(coefficients->values[0], coefficients->values[1],
                coefficients->values[2], coefficients->values[3]);
        all_planes_.push_back( pm ) ;

        for(size_t i=0; i<inliers->indices.size(); i++)
            filter_mask_[ inliers->indices[i] ] = false;
    }
}

template <typename PointT>
void
EuclideanSegmenter<PointT>::segment()
{
    clusters_.clear();
    filter_mask_.clear();
    filter_mask_.resize(scene_->points.size(), true);

    scene_xyz_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_filtered.reset (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*scene_, *scene_xyz_);
    computeTablePlanes();

    // remove nan points
    for(size_t i=0; i<scene_xyz_->points.size(); i++)
    {
        if (!filter_mask_[i])
            continue;

        if( !pcl::isFinite( scene_xyz_->points[i] ) || scene_xyz_->points[i].z > param_.chop_z_ )
            filter_mask_[i] = false;
    }

    pcl::copyPointCloud(*scene_xyz_, filter_mask_, *cloud_filtered);


    std::vector<int> indices2originalMap (scene_xyz_->points.size());    // maps points from filtered point cloud to original cloud
    size_t kept=0;
    for(size_t i=0; i<filter_mask_.size(); i++)
    {
        if( filter_mask_[i] )
            indices2originalMap[kept++] = i;
    }
    indices2originalMap.resize(kept);

    typename pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (param_.cluster_tolerance_);
    ec.setMinClusterSize (param_.min_cluster_size_);
    ec.setMaxClusterSize (param_.max_cluster_size_);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (clusters_);

    // transform to original indices
    for(size_t i=0; i < clusters_.size(); i++)
    {
        pcl::PointIndices &cluster = clusters_[i];
        for(size_t pt_id=0; pt_id<cluster.indices.size(); pt_id++)
        {
            cluster.indices[pt_id] = indices2originalMap [ cluster.indices[pt_id] ];
        }
    }
}

#define PCL_INSTANTIATE_EuclideanSegmenter(T) template class V4R_EXPORTS EuclideanSegmenter<T>;
PCL_INSTANTIATE(EuclideanSegmenter, PCL_XYZ_POINT_TYPES )

}
