
#include "outputModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace object_modeller
{
namespace output
{

template<class TPointType>
class PointCloudWriter : public OutModule<std::vector<typename pcl::PointCloud<TPointType>::Ptr> >
{
private:
    std::string outputPath;
    std::string pattern;

public:
    PointCloudWriter(std::string config_name="pointCloudWriter")
        : OutModule<std::vector<typename pcl::PointCloud<TPointType>::Ptr> >(config_name)
    {}

    virtual void applyConfig(Config &config);

    void process(std::vector<typename pcl::PointCloud<TPointType>::Ptr> input);

    std::string getName()
    {
        return "Point cloud Writer";
    }
};

}
}
