
#include "inputModule.h"
#include "outputModule.h"
#include "ioModule.h"
#include "module.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace util
{

class DistanceFilter :
        public InOutModule<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{
private:
    float maxDist;
    bool euclideanDistance;

public:
    DistanceFilter(std::string config_name="distanceFilter") : InOutModule(config_name)
    {
        registerParameter("zDist", "Maximum Distance", &maxDist, 1.5f);
    }

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> input);

    virtual void applyConfig(Config::Ptr &config);

    std::string getName()
    {
        return "Distance filter";
    }
};

}
}
