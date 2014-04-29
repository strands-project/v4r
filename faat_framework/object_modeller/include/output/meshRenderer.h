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

namespace object_modeller
{
namespace output
{

class MeshRenderer : public OutModule<boost::tuples::tuple<pcl::PolygonMesh::Ptr, std::string, bool> >
{
private:
    boost::shared_ptr<Renderer> baseRenderer;

public:

    virtual void applyConfig(Config &config);
    MeshRenderer(boost::shared_ptr<Renderer> baseRenderer);
    void process(boost::tuples::tuple<pcl::PolygonMesh::Ptr, std::string, bool> input);

    std::string getName()
    {
        return "Mesh Renderer";
    }
};

}
}
