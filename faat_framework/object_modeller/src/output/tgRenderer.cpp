#include "output/tgRenderer.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

namespace object_modeller
{
namespace output
{

void TomGineRenderer::applyConfig(Config &config, std::string base_path)
{

}

template<class TPointType>
void TomGineRenderer::renderPointCloudsImpl(std::vector<typename pcl::PointCloud<TPointType>::Ptr> point_clouds, std::string name, bool step)
{
    std::cout << "Render point cloud with tom gine" << std::endl;

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*point_clouds[0], centroid);
    win->SetRotationCenter(centroid[0], centroid[1], centroid[2]);

    win->Clear();

    for (int i=0;i<point_clouds.size();i++)
    {
        win->AddPointCloudPCL(*point_clouds[i]);
    }

    win->AddLabel2D(name, 10, 10, 580);

    win->Update();

    if (step)
    {
        std::cout << "Wait for key press" << std::endl;

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
