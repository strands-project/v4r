#pragma once
#include <v4r/core/macros.h>
#include <pcl/point_cloud.h>


namespace v4r
{

/**
 * @brief visualize clustering output
 * @param input cloud
 * @param indices of points belonging to the individual clusters
 */
template<typename PointT>
V4R_EXPORTS
void
visualizeCluster(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector<int> &cluster_indices, const std::string &window_title = "Segmentation results" );


/**
 * @brief visualize clustering output
 * @param input cloud
 * @param indices of points belonging to the individual clusters
 */
template<typename PointT>
V4R_EXPORTS
void
visualizeClusters(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector< std::vector<int> > &cluster_indices, const std::string &window_title = "Segmentation results");

}

