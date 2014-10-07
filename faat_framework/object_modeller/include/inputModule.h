#pragma once

#include "module.h"
#include "result.h"

#include "output/renderer.h"

#include <boost/shared_ptr.hpp>

namespace object_modeller
{

template<class T>
class ResultPointer;

template<class RETURNTYPE>
class InModule : public Module
{
private:
    output::RenderDelegate<RETURNTYPE> renderDelegate;
    bool stopAfterRender;
    std::string renderName;
public:

    InModule(std::string config_name) : Module(config_name)
    {
        stopAfterRender = false;

        if (canRender())
        {
            registerParameter("stopAfterRender", "Stop after Rendering", &stopAfterRender, false);
            registerParameter("renderName", "Rendering Display Name", &renderName, getName());
        }
    }

    bool getStopAfterRender()
    {
        return stopAfterRender;
    }

    virtual int getNrOutputSequences()
    {
        return -1;
    }

    bool canRender()
    {
        return renderDelegate.canRender();
    }

    void renderResult(object_modeller::output::Renderer::Ptr renderer, int sequenceId, RETURNTYPE result)
    {
        renderer->setRenderName(renderName);
        renderDelegate.render(renderer, sequenceId, result);
    }

    typedef ResultPointer<RETURNTYPE> ResultType;

    typedef RETURNTYPE ReturnType;

    virtual RETURNTYPE process() = 0;
};

}
