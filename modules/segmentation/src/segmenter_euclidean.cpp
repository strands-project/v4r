#include <v4r/segmentation/segmenter_euclidean.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>

namespace v4r
{

template <typename PointT>
void
EuclideanSegmenter<PointT>::segment()
{
    // NaN points cause segmentation fault in kdtree search
    typename pcl::PointCloud<PointT>::Ptr scene_wo_nans (new pcl::PointCloud<PointT> );
    scene_wo_nans->points.resize( scene_->points.size() );

    std::vector<int> indices_converter ( scene_->points.size() );
    size_t kept=0;
    for(size_t i=0; i<scene_->points.size(); i++)
    {
        const PointT &p = scene_->points[i];
        if( pcl::isFinite(p)  )
        {
            scene_wo_nans->points[kept] = p;
            indices_converter[kept] = i;
            kept++;
        }
    }
    indices_converter.resize(kept);
    scene_wo_nans->points.resize(kept);
    scene_wo_nans->width = kept;
    scene_wo_nans->height = 1;

    typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud ( scene_wo_nans );
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (param_.cluster_tolerance_);
    ec.setMinClusterSize (param_.min_cluster_size_);
    ec.setMaxClusterSize (param_.max_cluster_size_);
    ec.setSearchMethod (tree);
    ec.setInputCloud ( scene_wo_nans );
    std::vector<pcl::PointIndices> clusters_pcl;
    ec.extract (clusters_pcl);

    clusters_.clear();
    clusters_.resize( clusters_pcl.size() );
    for(size_t i=0; i<clusters_pcl.size(); i++)
    {
        clusters_[i].reserve( clusters_pcl[i].indices.size() );
        for( int idx_wo_nan : clusters_pcl[i].indices )
            clusters_[i].push_back( indices_converter[ idx_wo_nan ] );
    }

}

#define PCL_INSTANTIATE_EuclideanSegmenter(T) template class V4R_EXPORTS EuclideanSegmenter<T>;
PCL_INSTANTIATE(EuclideanSegmenter, PCL_XYZ_POINT_TYPES )

}
