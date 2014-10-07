
#include "inputModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace reader
{

class MemoryFriendlyFileReader : public InModule<std::vector<std::string> >
{
private:
    std::string pattern;
    std::string inputPath;
    int step;
    std::vector<std::string> files_;
    int max_files_;
public:
    MemoryFriendlyFileReader(std::string config_name="reader") : InModule(config_name)
    {
        max_files_ = std::numeric_limits<int>::infinity();
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getCloud(int i);

    virtual std::vector<std::string> process();

    virtual void applyConfig(Config::Ptr config);

    std::string getName()
    {
        return "PCD File Reader (memory friendly)";
    }
};

}
}
