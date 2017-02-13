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
    typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud (scene_);
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (param_.cluster_tolerance_);
    ec.setMinClusterSize (param_.min_cluster_size_);
    ec.setMaxClusterSize (param_.max_cluster_size_);
    ec.setSearchMethod (tree);
    ec.setInputCloud (scene_);
    std::vector<pcl::PointIndices> clusters_pcl;
    ec.extract (clusters_pcl);

    clusters_.resize( clusters_pcl.size() );
    for(size_t i=0; i<clusters_.size(); i++)
        clusters_[i] = clusters_pcl[i].indices;

}

#define PCL_INSTANTIATE_EuclideanSegmenter(T) template class V4R_EXPORTS EuclideanSegmenter<T>;
PCL_INSTANTIATE(EuclideanSegmenter, PCL_XYZ_POINT_TYPES )

}
