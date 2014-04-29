
#include "util/transform.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

namespace object_modeller
{
namespace util
{

void Transform::applyConfig(Config &config)
{

}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> Transform::process(boost::tuples::tuple<
             std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
             std::vector<Eigen::Matrix4f> > input)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds = input.get<0>();
    std::vector<Eigen::Matrix4f> poses = input.get<1>();

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> result;

    for (size_t i = 0; i < pointClouds.size (); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud (new pcl::PointCloud<pcl::PointXYZRGB>);

        pcl::copyPointCloud(*pointClouds[i], *pointCloud);
        pcl::transformPointCloud(*pointCloud, *pointCloud, poses[i]);

        result.push_back(pointCloud);
    }

    return result;
}


}
}
