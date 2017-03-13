#include <v4r/segmentation/segmentation_utils.h>

#include <pcl/impl/instantiate.hpp>
#include <pcl/visualization/pcl_visualizer.h>

namespace v4r
{

template<typename PointT>
void
visualizeClusters(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector< std::vector<int> > &cluster_indices, const std::string &window_title )
{
    int vp1, vp2;
    static pcl::visualization::PCLVisualizer::Ptr vis;

    if(!vis)
        vis.reset ( new pcl::visualization::PCLVisualizer );

    vis->setWindowName(window_title);
    vis->createViewPort(0,0,0.5,1,vp1);
    vis->createViewPort(0.5,0,1,1,vp2);
    vis->removeAllPointClouds();
    vis->removeAllShapes();
    vis->addPointCloud<PointT>( cloud, "input", vp1 );

    typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
    for(size_t i=0; i < cluster_indices.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB> cluster;
        pcl::copyPointCloud(*cloud, cluster_indices[i], cluster);

        const uint8_t r = rand()%255;
        const uint8_t g = rand()%255;
        const uint8_t b = rand()%255;
        for(size_t pt_id=0; pt_id<cluster.points.size(); pt_id++)
        {
            cluster.points[pt_id].r = r;
            cluster.points[pt_id].g = g;
            cluster.points[pt_id].b = b;
        }
        *colored_cloud += cluster;

        std::stringstream txt; txt << std::setfill(' ') << std::setw(7) << cluster_indices[i].size() << " pts";
        std::stringstream label; label << "cluster_" << i;
        vis->addText(txt.str(), 10, 15+14*i, 15, r/255.f, g/255.f, b/255.f, label.str(), vp2);
    }
    vis->addPointCloud(colored_cloud,"segments", vp2);

    vis->addText("input", 10, 10, 15, 1, 1, 1, "input", vp1);
    vis->addText("segments", 10, 10, 15, 1, 1, 1, "segments", vp2);
    vis->resetCamera();
    vis->spin();
}

template<typename PointT>
void
visualizeCluster(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector<int> &cluster_indices, const std::string &window_title )
{
    int vp1, vp2;

    static pcl::visualization::PCLVisualizer::Ptr vis;

    if(!vis)
        vis.reset ( new pcl::visualization::PCLVisualizer );

    vis->setWindowName(window_title);
    vis->createViewPort(0,0,0.5,1,vp1);
    vis->createViewPort(0.5,0,1,1,vp2);
    vis->removeAllPointClouds();
    vis->removeAllShapes();
    vis->addPointCloud<PointT>( cloud, "input", vp1 );

    // generate random colors for each cluster

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud, cluster_indices, *cluster);
    const uint8_t r = rand()%255;
    const uint8_t g = rand()%255;
    const uint8_t b = rand()%255;
    for(size_t pt_id=0; pt_id<cluster->points.size(); pt_id++)
    {
        cluster->points[pt_id].r = r;
        cluster->points[pt_id].g = g;
        cluster->points[pt_id].b = b;
    }
    vis->addPointCloud(cluster,"segment", vp2);
    vis->addText("input", 10, 10, 15, 1, 1, 1, "input", vp1);
    vis->addText("segments", 10, 10, 15, 1, 1, 1, "segments", vp2);
    vis->resetCamera();
    vis->spin();
    vis->close();
}

#define PCL_INSTANTIATE_visualizeClusters(T) template V4R_EXPORTS void visualizeClusters<T>(const pcl::PointCloud<T>::ConstPtr &, const std::vector<std::vector<int> > &, const std::string &);
PCL_INSTANTIATE(visualizeClusters, PCL_XYZ_POINT_TYPES )

#define PCL_INSTANTIATE_visualizeCluster(T) template V4R_EXPORTS void visualizeCluster<T>(const pcl::PointCloud<T>::ConstPtr &, const std::vector<int> &, const std::string &);
PCL_INSTANTIATE(visualizeCluster, PCL_XYZ_POINT_TYPES )

}
