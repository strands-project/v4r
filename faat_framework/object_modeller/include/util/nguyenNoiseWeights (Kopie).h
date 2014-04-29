
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

class NguyenNoiseWeights :
        public InOutModule<std::vector<std::vector<float> >,
                           boost::tuples::tuple<
                            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                            std::vector<pcl::PointCloud<pcl::Normal>::Ptr> > >
{
private:
    bool depth_edges;
    float max_angle;
    float lateral_sigma;
public:
    virtual void applyConfig(Config &config);

    std::vector<std::vector<float> > process(boost::tuples::tuple<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> , std::vector<pcl::PointCloud<pcl::Normal>::Ptr> > input);

    std::string getName()
    {
        return "Nguyen noise weights";
    }
};

}
}
