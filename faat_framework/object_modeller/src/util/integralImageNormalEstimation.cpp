
#include "util/integralImageNormalEstimation.h"

#include <pcl/features/integral_image_normal.h>

namespace object_modeller
{
namespace util
{

void IntegralImageNormalEstimation::applyConfig(Config &config)
{

}

std::vector<pcl::PointCloud<pcl::Normal>::Ptr> IntegralImageNormalEstimation::process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds)
{

    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normal_clouds;

    for(size_t i=0; i < pointClouds.size(); i++)
    {
        pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);

        pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
        ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
        ne.setMaxDepthChangeFactor (0.02f);
        ne.setNormalSmoothingSize (20.0f);
        ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::BORDER_POLICY_MIRROR);
        ne.setInputCloud (pointClouds[i]);
        ne.compute (*normal_cloud);

        normal_clouds.push_back(normal_cloud);
    }

    return normal_clouds;
}

}
}
