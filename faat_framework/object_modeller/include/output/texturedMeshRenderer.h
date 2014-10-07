#pragma once

#include "outputModule.h"
#include "module.h"
#include "rendererArgs.h"
#include "renderer.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "texturing/pclTexture.h"

namespace object_modeller
{
namespace output
{

class TexturedMeshRenderer : public OutModule<boost::tuples::tuple<object_modeller::output::TexturedMesh::Ptr, std::string, bool> >
{
private:
    boost::shared_ptr<Renderer> baseRenderer;

public:
    TexturedMeshRenderer(boost::shared_ptr<Renderer> baseRenderer, std::string config_name="renderer");

    virtual void applyConfig(Config &config);

    void process(boost::tuples::tuple<object_modeller::output::TexturedMesh::Ptr, std::string, bool> input);

    std::string getName()
    {
        return "Mesh Renderer";
    }
};

}
}
