
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "boost/tuple/tuple.hpp"

namespace object_modeller
{
namespace util
{

class Transform :
        public InOutModule<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                        boost::tuples::tuple<
                            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                            std::vector<Eigen::Matrix4f> > >
{

public:
    Transform(std::string config_name="transform") : InOutModule(config_name)
    {}

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> process(boost::tuples::tuple<
                 std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                 std::vector<Eigen::Matrix4f> > input);

    virtual void applyConfig(Config &config);

    std::string getName()
    {
        return "Transform point clouds";
    }
};

}
}
