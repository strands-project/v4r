#include <v4r/segmentation/plane_utils.h>

#include <pcl/point_cloud.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/visualization/pcl_visualizer.h>

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


template<typename PointT>
V4R_EXPORTS
void
visualizePlane(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const Eigen::Vector4f &plane, float inlier_threshold, const std::string &window_title )
{
    typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud( *cloud, *plane_cloud );

    std::vector<int> plane_inliers = get_all_plane_inliers( *cloud, plane, inlier_threshold );

    for(pcl::PointXYZRGB &p :plane_cloud->points)
        p.r = p.g = p.b = 0.f;

    for(int idx : plane_inliers)
        plane_cloud->points[idx].g = 255.f;

    int vp1, vp2;

    static pcl::visualization::PCLVisualizer::Ptr vis;

    if(!vis)
        vis.reset ( new pcl::visualization::PCLVisualizer );

    vis->setWindowName(window_title);
    vis->removeAllPointClouds();
    vis->removeAllShapes();
    vis->createViewPort(0,0,0.5,1,vp1);
    vis->createViewPort(0.5,0,1,1,vp2);
    vis->addPointCloud<PointT>( cloud, "input", vp1 );
    vis->addPointCloud<pcl::PointXYZRGB>( plane_cloud, "plane_inliers_cloud", vp2);
    vis->addText("input", 10, 10, 15, 1, 1, 1, "input_txt", vp1);
    vis->addText("plane inliers", 10, 10, 15, 1, 1, 1, "plane_inliers_txt", vp2);
    vis->resetCamera();
    vis->spin();
    vis->close();
}

template<typename PointT>
V4R_EXPORTS
void
visualizePlanes(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > &planes, float inlier_threshold, const std::string &window_title )
{
    typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr planes_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud( *cloud, *planes_cloud );
    for(pcl::PointXYZRGB &p :planes_cloud->points)
        p.r = p.g = p.b = 0.f;

    for(size_t i=0; i<planes.size(); i++)
    {
        float r = rand()%255;
        float g = rand()%255;
        float b = rand()%255;

        std::vector<int> plane_inliers = get_all_plane_inliers( *cloud, planes[i], inlier_threshold );

        for(int idx : plane_inliers)
        {
            pcl::PointXYZRGB &p = planes_cloud->points[idx];
            p.r = r;
            p.g = g;
            p.b = b;
        }
    }

    int vp1, vp2;

    static pcl::visualization::PCLVisualizer::Ptr vis;

    if(!vis)
        vis.reset ( new pcl::visualization::PCLVisualizer );

    vis->setWindowName(window_title);
    vis->removeAllPointClouds();
    vis->removeAllShapes();
    vis->createViewPort(0,0,0.5,1,vp1);
    vis->createViewPort(0.5,0,1,1,vp2);
    vis->addPointCloud<PointT>( cloud, "input", vp1 );
    vis->addPointCloud<pcl::PointXYZRGB>( planes_cloud, "plane_inliers_txt", vp2);
    vis->addText("input", 10, 10, 15, 1, 1, 1, "input_txt", vp1);
    vis->addText("plane inliers", 10, 10, 15, 1, 1, 1, "plane_inliers_cloud", vp2);
    vis->resetCamera();
    vis->spin();
    vis->close();
}

#define PCL_INSTANTIATE_get_largest_connected_plane_inliers(T) template V4R_EXPORTS  std::vector<int> get_largest_connected_plane_inliers<T>(const pcl::PointCloud<T> &, const Eigen::Vector4f &, float, float, int);
PCL_INSTANTIATE(get_largest_connected_plane_inliers, PCL_XYZ_POINT_TYPES )

#define PCL_INSTANTIATE_get_largest_connected_inliers(T) template V4R_EXPORTS  std::vector<int> get_largest_connected_inliers<T>(const pcl::PointCloud<T> &, const std::vector<int> &, float, int);
PCL_INSTANTIATE(get_largest_connected_inliers, PCL_XYZ_POINT_TYPES )

#define PCL_INSTANTIATE_getConvexHullCloud(T) template V4R_EXPORTS  pcl::PointCloud<T>::Ptr getConvexHullCloud<T>(const pcl::PointCloud<T>::ConstPtr);
PCL_INSTANTIATE(getConvexHullCloud, PCL_XYZ_POINT_TYPES )

#define PCL_INSTANTIATE_visualizePlane(T) template V4R_EXPORTS void visualizePlane<T>(const pcl::PointCloud<T>::ConstPtr &, const Eigen::Vector4f &, float, const std::string &);
PCL_INSTANTIATE(visualizePlane, PCL_XYZ_POINT_TYPES )

#define PCL_INSTANTIATE_visualizePlanes(T) template V4R_EXPORTS void visualizePlanes<T>(const pcl::PointCloud<T>::ConstPtr &, const std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > &, float, const std::string &);
PCL_INSTANTIATE(visualizePlanes, PCL_XYZ_POINT_TYPES )

}
