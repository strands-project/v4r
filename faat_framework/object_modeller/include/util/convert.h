
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/common/common.h>

#include "boost/tuple/tuple.hpp"

namespace object_modeller
{
namespace util
{

class ConvertPointCloud :
        public InOutModule<std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>,
                            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{

public:
    ConvertPointCloud(std::string config_name="convert") : InOutModule(config_name)
    {}

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> input)
    {
        std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr > result;

        for (int i=0;i<input.size();i++)
        {
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr new_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr p = input[i];

            new_cloud->resize(p->points.size());

            for (size_t j = 0; j < p->points.size(); j++) {
                new_cloud->points[j].x = p->points[j].x;
                new_cloud->points[j].y = p->points[j].y;
                new_cloud->points[j].z = p->points[j].z;

                new_cloud->points[j].r = p->points[j].r;
                new_cloud->points[j].g = p->points[j].g;
                new_cloud->points[j].b = p->points[j].b;
            }

            result.push_back(new_cloud);
        }

        return result;
    }

    virtual void applyConfig(Config &config) {}

    std::string getName()
    {
        return "Convert point clouds";
    }
};

}
}
