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

void PclRenderer::applyConfig(Config &config)
{

}

PclRenderer::PclRenderer()
{
    vis.reset(new pcl::visualization::PCLVisualizer(""));
    vis->registerKeyboardCallback (keyboardEventOccurred, (void*) &vis);
}

void PclRenderer::process(boost::tuples::tuple<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>, std::string, bool> input)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds = input.get<0>();
    std::string text = input.get<1>();
    bool step = input.get<2>();

    int v;
    vis->createViewPort(0,0,1,1,v);
    vis->removeAllPointClouds();

    for (int i=0;i<pointClouds.size();i++)
    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (pointClouds[i]);
        std::stringstream name;
        name << text << i;
        vis->addPointCloud<pcl::PointXYZRGB> (pointClouds[i], handler_rgb, name.str(), v);
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
