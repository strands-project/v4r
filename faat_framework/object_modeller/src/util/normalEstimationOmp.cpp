
#include "util/normalEstimationOmp.h"

#include <pcl/features/normal_3d_omp.h>

namespace object_modeller
{
namespace util
{

void NormalEstimationOmp::applyConfig(Config &config)
{

}

std::vector<pcl::PointCloud<pcl::Normal>::Ptr> NormalEstimationOmp::process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds)
{

    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normal_clouds;

    for(size_t i=0; i < pointClouds.size(); i++)
    {
        pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);

        pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
        ne.setInputCloud (pointClouds[i]);
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
        ne.setSearchMethod (tree);
        ne.setRadiusSearch (0.02);
        ne.compute (*normal_cloud);

        normal_clouds.push_back(normal_cloud);
    }

    return normal_clouds;
}

}
}
