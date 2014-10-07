
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
    NguyenNoiseWeights(std::string config_name="nguyenNoiseWeights") : InOutModule(config_name)
    {
        depth_edges = true;
        max_angle = 60.f;
        lateral_sigma = 0.002f;
    }

    std::vector<std::vector<float> > process(boost::tuples::tuple<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> , std::vector<pcl::PointCloud<pcl::Normal>::Ptr> > input);
    std::vector<std::vector<float> > process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & pointClouds , std::vector<pcl::PointCloud<pcl::Normal>::Ptr> & normal_clouds);

    std::string getName()
    {
        return "Nguyen noise weights";
    }
};

}
}
