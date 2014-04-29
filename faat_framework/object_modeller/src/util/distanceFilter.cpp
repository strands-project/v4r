
#include "util/distanceFilter.h"

#include <pcl/filters/passthrough.h>

namespace object_modeller
{
namespace util
{

void DistanceFilter::applyConfig(Config &config)
{
    this->maxDist = config.getFloat("distanceFilter.zDist", 2.0f);
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> DistanceFilter::process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> result;

    std::cout << "input cloud size: " << pointClouds.size() << std::endl;

    for (size_t i = 0; i < pointClouds.size (); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud (new pcl::PointCloud<pcl::PointXYZRGB>);

        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setFilterLimits (0.f, maxDist);
        pass.setFilterFieldName ("z");
        pass.setInputCloud (pointClouds[i]);
        pass.setKeepOrganized (true);
        pass.filter (*pointCloud);

        result.push_back(pointCloud);
    }

    std::cout << "result size: " << result.size() << std::endl;

    return result;
}

}
}
