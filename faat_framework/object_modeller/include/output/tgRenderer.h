#pragma once

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
        win->StartEventListener(1);
    }

    output::Renderer::Event waitForEvent();

    virtual void pipelineStateChanged(State state, std::string activeStepName)
    {
        if (state == EventManager::PAUSED)
        {
            update();

            Event e = waitForEvent();
            trigger(e);
        }
    }

    boost::shared_ptr<TomGine::tgModel> convert (pcl::PolygonMesh &mesh);

    virtual void update();

    virtual void updateImage() {}

    virtual void enableRoiMode(Eigen::Vector3f *dim, Eigen::Vector3f *translation, Eigen::Quaternionf *rotation)
    {

    }

    std::string getName()
    {
        return "TomGine Renderer";
    }
};

}
}
