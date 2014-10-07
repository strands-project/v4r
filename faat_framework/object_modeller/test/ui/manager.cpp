
#include "manager.h"

#include <string>

#include "result.h"

#include "pipelineFactory.h"

#include <QVTKWidget.h>


Manager::Manager()
{
    std::cout << "crating manager" << std::endl;
    config.reset(new AutoSaveConfig());
}

void Manager::setupPipeline(object_modeller::output::Renderer::Ptr renderer)
{
    std::cout << "setup pipeline" << std::endl;

    this->renderer = renderer;

    object_modeller::PipelineFactory factory;
    pipeline = factory.create(factory.getPipelines().back(), config, renderer);

    applyConfig(config);
}


object_modeller::Pipeline::Ptr Manager::getPipeline()
{
    return pipeline;
}
