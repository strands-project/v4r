
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

class IndicesWriter : public OutModule<std::vector<std::vector<int> > >
{
private:
    std::string outputPath;
    std::string pattern;

public:
    IndicesWriter(std::string config_name="indicesWriter") : OutModule(config_name)
    {}

    virtual void applyConfig(Config &config);

    void process(std::vector<std::vector<int> > indices);

    std::string getName()
    {
        return "Indices Writer";
    }
};

}
}
