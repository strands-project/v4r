
#include "outputModule.h"
#include "module.h"
#include "rendererArgs.h"
#include "renderer.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace object_modeller
{
namespace output
{

class PclRenderer : public Renderer
{
private:
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis;

public:
    PclRenderer();

    virtual void applyConfig(Config &config);
    void process(boost::tuples::tuple<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>, std::string, bool> input);
    virtual void renderMesh(pcl::PolygonMesh::Ptr, std::string, bool);

    std::string getName()
    {
        return "Pcl Renderer";
    }
};

}
}
