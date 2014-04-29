#include "output/tgRenderer.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

namespace object_modeller
{
namespace output
{

void TomGineRenderer::applyConfig(Config &config)
{

}

TomGineRenderer::TomGineRenderer()
{
    win.reset(new TomGine::tgTomGineThreadPCL(800,600));
}

void TomGineRenderer::process(boost::tuples::tuple<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>, std::string, bool> input)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds = input.get<0>();
    std::string text = input.get<1>();
    bool step = input.get<2>();

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*pointClouds[0], centroid);
    win->SetRotationCenter(centroid[0], centroid[1], centroid[2]);

    win->Clear();

    for (int i=0;i<pointClouds.size();i++)
    {
        win->AddPointCloudPCL(*pointClouds[i]);
    }

    win->AddLabel2D(text, 10, 10, 580);

    win->Update();

    if (step)
    {
        win->AddLabel2D("Press SPACE to continue", 10, 10, 10);
        win->WaitForEvent(TomGine::TMGL_Press, TomGine::TMGL_Space);
    }
}

void TomGineRenderer::renderMesh(pcl::PolygonMesh::Ptr mesh, std::string text, bool step)
{
    //Eigen::Vector4f centroid;
    //pcl::compute3DCentroid(*pointClouds[0], centroid);
    //win->SetRotationCenter(centroid[0], centroid[1], centroid[2]);

    win->Clear();

    win->AddModelPCL(*mesh);

    win->AddLabel2D(text, 10, 10, 580);

    win->Update();

    if (step)
    {
        win->AddLabel2D("Press SPACE to continue", 10, 10, 10);
        win->WaitForEvent(TomGine::TMGL_Press, TomGine::TMGL_Space);
    }
}

}
}
