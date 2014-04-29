
#include "inputModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace reader
{

class FileReader : public InModule<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{
private:
    std::string pattern;
    std::string inputPath;
    int step;

public:

    virtual std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> process();

    virtual void applyConfig(Config &config);

    std::string getName()
    {
        return "PCD File Reader";
    }
};

}
}
