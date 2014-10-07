#include "output/texturedMeshRenderer.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

namespace object_modeller
{
namespace output
{

void TexturedMeshRenderer::applyConfig(Config &config)
{

}

TexturedMeshRenderer::TexturedMeshRenderer(boost::shared_ptr<Renderer> baseRenderer, std::string config_name) : OutModule(config_name)
{
    this->baseRenderer = baseRenderer;
}

void TexturedMeshRenderer::process(boost::tuples::tuple<object_modeller::output::TexturedMesh::Ptr, std::string, bool> input)
{
    object_modeller::output::TexturedMesh::Ptr mesh = input.get<0>();
    std::string text = input.get<1>();
    bool step = input.get<2>();

    baseRenderer->addTexturedMesh(0, mesh);
    baseRenderer->update();
}

}
}
