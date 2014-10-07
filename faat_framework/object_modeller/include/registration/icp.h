
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace registration
{

class Icp :
        public InOutModule<std::vector<Eigen::Matrix4f>,
                           std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{
private:

public:
    Icp(std::string config_name="icp") : InOutModule(config_name)
    {}

    virtual void applyConfig(Config &config);

    std::vector<Eigen::Matrix4f> process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds);

    std::string getName()
    {
        return "ICP";
    }
};

}
}
