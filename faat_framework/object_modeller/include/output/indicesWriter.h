
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
    {
        registerParameter("outputPath", "Output path", &outputPath, std::string("./out"));
        registerParameter("pattern", "Pattern", &pattern, std::string("object_indices_*.txt"));
    }

    void process(std::vector<std::vector<int> > indices);

    std::string getName()
    {
        return "Indices Writer";
    }
};

}
}
