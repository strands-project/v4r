
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

class NormalEstimationOmp :
        public InOutModule<std::vector<pcl::PointCloud<pcl::Normal>::Ptr>,
                           std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{
public:
    NormalEstimationOmp(std::string config_name="normalEstimationOmp") : InOutModule(config_name)
    {}

    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> input);

    std::string getName()
    {
        return "Normal estimation OMP";
    }
};

}
}
