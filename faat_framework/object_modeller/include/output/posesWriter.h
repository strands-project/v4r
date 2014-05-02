
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

class PosesWriter : public OutModule<std::vector<Eigen::Matrix4f> >
{
private:
    std::string outputPath;
    std::string pattern;

public:
    PosesWriter(std::string config_name="posesWriter") : OutModule(config_name)
    {}

    virtual void applyConfig(Config &config);

    void process(std::vector<Eigen::Matrix4f> poses);
    bool writeMatrixToFile (std::string file, Eigen::Matrix4f & matrix);

    std::string getName()
    {
        return "Poses Writer";
    }
};

}
}
