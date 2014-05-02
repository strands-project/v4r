
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

    TomGineRenderer()
    {
        win.reset(new TomGine::tgTomGineThreadPCL(800,600));
    }

    virtual void applyConfig(Config &config, std::string base_path);

    template<class TPointType>
    void renderPointCloudsImpl(std::vector<typename pcl::PointCloud<TPointType>::Ptr> point_clouds, std::string name, bool step);

    virtual void renderPointClouds(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> point_clouds, std::string name, bool step)
    {
        renderPointCloudsImpl<pcl::PointXYZRGB>(point_clouds, name, step);
    }

    virtual void renderPointClouds(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> point_clouds, std::string name, bool step)
    {
        renderPointCloudsImpl<pcl::PointXYZRGBNormal>(point_clouds, name, step);
    }

    virtual void renderMesh(pcl::PolygonMesh::Ptr, std::string, bool);

    std::string getName()
    {
        return "TomGine Renderer";
    }
};

}
}
