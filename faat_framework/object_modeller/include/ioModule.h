#pragma once

#include "module.h"
#include "result.h"

#include "output/renderer.h"

#include <boost/shared_ptr.hpp>

namespace object_modeller
{

template<class T>
class ResultPointer;

template<class RETURNTYPE, class ARG>
class InOutModule : public Module
{
private:
    object_modeller::output::RenderDelegate<RETURNTYPE> renderDelegate;
protected:
    int nrInputSequences;
    bool stopAfterRender;
    std::string renderName;
public:

    InOutModule(std::string config_name, VisualConfigBase *visualConfig = NULL) : Module(config_name, visualConfig)
    {
        nrInputSequences = 1;
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

    void setNrInputSequences(int nrInputSequences)
    {
        this->nrInputSequences = nrInputSequences;
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

    typedef RETURNTYPE ReturnType;
    typedef ResultPointer<RETURNTYPE> ResultType;

    virtual RETURNTYPE process(ARG arg) = 0;
};

}
