#include "output/pclRenderer.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

namespace object_modeller
{
namespace output
{

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
{
    if (event.getKeySym () == "space" && event.keyDown ())
    {
        boost::shared_ptr<pcl::visualization::PCLVisualizer> vis = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
        vis->spinOnce();
    }
}

void PclRenderer::applyConfig(Config &config, std::string base_path)
{

}

PclRenderer::PclRenderer()
{
    vis.reset(new pcl::visualization::PCLVisualizer(""));
    vis->registerKeyboardCallback (keyboardEventOccurred, (void*) &vis);
}

template<class TPointType>
void PclRenderer::renderPointCloudsImpl(std::vector<typename pcl::PointCloud<TPointType>::Ptr> point_clouds, std::string text, bool step)
{
    int v;
    vis->createViewPort(0,0,1,1,v);
    vis->removeAllPointClouds();

    for (int i=0;i<point_clouds.size();i++)
    {
        pcl::visualization::PointCloudColorHandlerRGBField<TPointType> handler_rgb (point_clouds[i]);
        std::stringstream name;
        name << text << i;
        vis->addPointCloud<TPointType> (point_clouds[i], handler_rgb, name.str(), v);
    }

    if (step)
    {
        vis->spin();
    }
    else
    {
        vis->spinOnce();
    }
}

void PclRenderer::renderMesh(pcl::PolygonMesh::Ptr mesh, std::string text, bool step)
{
    int v;
    vis->createViewPort(0,0,1,1,v);
    vis->removeAllPointClouds();
    vis->removeAllShapes();

    vis->addPolygonMesh(*mesh, text, v);

    if (step)
    {
        vis->spin();
    }
    else
    {
        vis->spinOnce();
    }
}

}
}
