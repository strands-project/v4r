
#include "outputModule.h"
#include "rendererArgs.h"
#include "module.h"
#include "renderer.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <v4r/TomGinePCL/tgTomGineThreadPCL.h>

namespace object_modeller
{
namespace output
{

class TomGineRenderer : public Renderer
{
private:
    boost::shared_ptr<TomGine::tgTomGineThreadPCL> win;

public:
    TomGineRenderer();

    virtual void applyConfig(Config &config);
    void process(boost::tuples::tuple<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>, std::string, bool> input);
    virtual void renderMesh(pcl::PolygonMesh::Ptr, std::string, bool);

    std::string getName()
    {
        return "TomGine Renderer";
    }
};

}
}
