
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

public:
    virtual void applyConfig(Config &config);

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> input);

    std::string getName()
    {
        return "Distance filter point clouds";
    }
};

}
}
