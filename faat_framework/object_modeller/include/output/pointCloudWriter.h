
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
    {
        ConfigItem::registerParameter("outputPath", "Output path", &outputPath, std::string("./out"));
        ConfigItem::registerParameter("pattern", "Pattern", &pattern, std::string("cloud_*.pcd"));
    }

    void process(std::vector<typename pcl::PointCloud<TPointType>::Ptr> input);

    std::string getName()
    {
        return "Point cloud Writer";
    }
};

}
}
