#include "output/meshRenderer.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

namespace object_modeller
{
namespace output
{

void MeshRenderer::applyConfig(Config &config)
{

}

MeshRenderer::MeshRenderer(boost::shared_ptr<Renderer> baseRenderer, std::string config_name) : OutModule(config_name)
{
    this->baseRenderer = baseRenderer;
}

void MeshRenderer::process(boost::tuples::tuple<pcl::PolygonMesh::Ptr, std::string, bool> input)
{
    pcl::PolygonMesh::Ptr mesh = input.get<0>();
    std::string text = input.get<1>();
    bool step = input.get<2>();

    baseRenderer->renderMesh(mesh, text, step);
}

}
}
