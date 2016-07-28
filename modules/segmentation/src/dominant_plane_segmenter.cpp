#include <v4r/segmentation/dominant_plane_segmenter.h>

namespace v4r
{

template<typename PointT>
void
DominantPlaneSegmenter<PointT>::segment()
{
    clusters_.clear();
    pcl::apps::DominantPlaneSegmentation<PointT> dps;
    dps.setInputCloud (scene_);
    dps.setMaxZBounds (param_.chop_z_);
    dps.setObjectMinHeight (param_.object_min_height_);
    dps.setObjectMaxHeight (param_.object_max_height_);
    dps.setMinClusterSize (param_.min_cluster_size_);
    dps.setWSize (param_.w_size_px_);
    dps.setDistanceBetweenClusters (param_.min_distance_between_clusters_);
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    dps.setDownsamplingSize ( param_.downsampling_size_ );
    if(param_.compute_table_plane_only_)
    {
        dps.compute_table_plane();
    }
    else
    {
        dps.compute_fast( clusters);
        dps.getIndicesClusters (clusters_);
    }
    dps.getTableCoefficients (dominant_plane_);

    if(visualize_)
        this->visualize();
}

template class V4R_EXPORTS DominantPlaneSegmenter<pcl::PointXYZRGB>;
}
