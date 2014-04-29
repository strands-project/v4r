
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

class PointCloudWriter : public OutModule<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{
private:
    std::string outputPath;

public:
    virtual void applyConfig(Config &config);

    void process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> input);

    std::string getName()
    {
        return "Point cloud Writer";
    }
};

}
}
