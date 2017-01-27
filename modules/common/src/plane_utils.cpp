#include <v4r/common/plane_utils.h>

#include <pcl/point_cloud.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/convex_hull.h>

namespace v4r
{

template<typename PointT>
V4R_EXPORTS
typename pcl::PointCloud<PointT>::Ptr
getConvexHullCloud(const typename pcl::PointCloud<PointT>::ConstPtr cloud)
{
    typename pcl::PointCloud<PointT>::Ptr convex_hull_cloud (new pcl::PointCloud<PointT>);

    pcl::ConvexHull<PointT> convex_hull;
    convex_hull.setInputCloud ( cloud );
    convex_hull.setDimension (2);
    convex_hull.setComputeAreaVolume (false);
    pcl::PolygonMeshPtr mesh_out(new pcl::PolygonMesh);

    std::vector<pcl::Vertices> polygons;
    convex_hull.reconstruct (*convex_hull_cloud, polygons);
//    convex_hull.reconstruct (*mesh_out);
//    convex_hull_ = mesh_out;

    return convex_hull_cloud;
}

template<typename PointT>
V4R_EXPORTS
std::vector<int>
get_largest_connected_plane_inliers(const pcl::PointCloud<PointT> &cloud, const Eigen::Vector4f &plane, float plane_inlier_threshold, float cluster_tolerance, int min_cluster_size)
{
    std::vector<int> all_plane_indices = get_all_plane_inliers(cloud, plane, plane_inlier_threshold);
    return get_largest_connected_inliers(cloud, all_plane_indices, cluster_tolerance, min_cluster_size);
}

template<typename PointT>
V4R_EXPORTS
std::vector<int>
get_largest_connected_inliers(const pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices, float cluster_tolerance, int min_cluster_size)
{
    if ( indices.empty() )
    {
        std::cerr << "Given indices for connected components segmentation are emtpy!" << std::endl;
        return indices;
    }

    typename pcl::PointCloud<PointT>::Ptr all_plane_cloud(new pcl::PointCloud<PointT>());
    pcl::copyPointCloud( cloud, indices, *all_plane_cloud);

    // segment all points that are fulfilling the plane equation and cluster them.
    // Table plane corresponds to biggest cluster, remaining points do not belong to plane
    typename pcl::search::KdTree<PointT>::Ptr tree_plane (new pcl::search::KdTree<PointT>);
    tree_plane->setInputCloud (all_plane_cloud);
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance ( cluster_tolerance );
    ec.setMinClusterSize ( min_cluster_size );
    ec.setMaxClusterSize ( std::numeric_limits<int>::max() );
    ec.setSearchMethod ( tree_plane );
    ec.setInputCloud (all_plane_cloud);
    std::vector<pcl::PointIndices> plane_cluster_indices;
    ec.extract (plane_cluster_indices);

    std::sort(plane_cluster_indices.begin(), plane_cluster_indices.end(),
              [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
        return a.indices.size() > b.indices.size();
    });

    if( !plane_cluster_indices.empty() )
    {
        std::vector<int> largest_connected_plane_indices (plane_cluster_indices[0].indices.size());
        for(size_t p_id=0; p_id<largest_connected_plane_indices.size(); p_id++)
            largest_connected_plane_indices[p_id] = indices[ plane_cluster_indices[0].indices[p_id] ];
       return largest_connected_plane_indices;
    }

    return indices;
}


#define PCL_INSTANTIATE_get_largest_connected_plane_inliers(T) template V4R_EXPORTS  std::vector<int> get_largest_connected_plane_inliers<T>(const pcl::PointCloud<T> &, const Eigen::Vector4f &, float, float, int);
PCL_INSTANTIATE(get_largest_connected_plane_inliers, PCL_XYZ_POINT_TYPES )

#define PCL_INSTANTIATE_get_largest_connected_inliers(T) template V4R_EXPORTS  std::vector<int> get_largest_connected_inliers<T>(const pcl::PointCloud<T> &, const std::vector<int> &, float, int);
PCL_INSTANTIATE(get_largest_connected_inliers, PCL_XYZ_POINT_TYPES )

#define PCL_INSTANTIATE_getConvexHullCloud(T) template V4R_EXPORTS  pcl::PointCloud<T>::Ptr getConvexHullCloud<T>(const pcl::PointCloud<T>::ConstPtr);
PCL_INSTANTIATE(getConvexHullCloud, PCL_XYZ_POINT_TYPES )

}
