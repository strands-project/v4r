
#include "util/distanceFilter.h"

#include <pcl/filters/passthrough.h>
#include <pcl/io/io.h>

namespace object_modeller
{
namespace util
{

void DistanceFilter::applyConfig(Config &config)
{
    this->maxDist = config.getFloat(getConfigName(), "zDist", 2.0f);
    this->euclideanDistance = config.getBool(getConfigName(), "euclideanDistance", false);
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> DistanceFilter::process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> result;

    std::cout << "input cloud size: " << pointClouds.size() << std::endl;

    for (size_t i = 0; i < pointClouds.size (); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud (new pcl::PointCloud<pcl::PointXYZRGB>);

        if(!euclideanDistance)
        {
            pcl::PassThrough<pcl::PointXYZRGB> pass;
            pass.setFilterLimits (0.f, maxDist);
            pass.setFilterFieldName ("z");
            pass.setInputCloud (pointClouds[i]);
            pass.setKeepOrganized (true);
            pass.filter (*pointCloud);
        }
        else
        {

            pcl::copyPointCloud(*pointClouds[i], *pointCloud);
            float bad_value = std::numeric_limits<float>::quiet_NaN();

            for(size_t k=0; k < pointClouds[i]->points.size(); k++)
            {
                const Eigen::Vector3f & p = pointClouds[i]->points[k].getVector3fMap();
                if(pcl_isnan(p[0]) || pcl_isnan(p[1]) || pcl_isnan(p[2]))
                    continue;

                if(p.norm() > maxDist)
                {
                    //set to NaN
                    pointCloud->points[k].x = bad_value;
                    pointCloud->points[k].y = bad_value;
                    pointCloud->points[k].z = bad_value;
                }
            }
        }

        result.push_back(pointCloud);
    }

    std::cout << "result size: " << result.size() << std::endl;

    return result;
}

}
}
