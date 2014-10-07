
#include "visualConfig.h"

#include "pipeline.h"

using namespace std;

namespace object_modeller
{

void Pipeline::setState(EventManager::State state, std::string activeStepName)
{
    this->state = state;
    renderer->pipelineStateChanged(state, activeStepName);
}

void Pipeline::process(bool autorun)
{
    EventManager::Event event = EventManager::NONE;
    if (autorun)
    {
        setState(EventManager::RUNNING);
        event = EventManager::RUN;
    }
    else
    {
        setState(EventManager::INITIALIZED);
    }

    while (state != EventManager::QUIT)
    {
        if (state != EventManager::RUNNING)
        {
            event = renderer->consumeBlocking();
        } else {
            event = renderer->consumeNonBlocking();
        }

        if (event == EventManager::CLOSE)
        {
            setState(EventManager::QUIT);
            break;
        }

        if (event == EventManager::PAUSE)
        {
            setState(EventManager::PAUSED);
        }

        if (event == EventManager::STEP)
        {
            setState(EventManager::RUNNING, modules[currentStep]->getName());
            step();
            setState(EventManager::PAUSED);
        }

        if (event == EventManager::RUN)
        {
            setState(EventManager::RUNNING, modules[currentStep]->getName());
        }

        if (event == EventManager::RESET)
        {
            currentStep = 0;
            renderer->clear();
            setState(EventManager::INITIALIZED);
        }

        if (state == EventManager::RUNNING)
        {
            /*
            EventManager::Event e = EventManager::NONE;

            while (e != EventManager::UPDATE_COMPLETE)
            {
                e = renderer->consumeBlocking();
            }*/

            setState(EventManager::RUNNING, modules[currentStep]->getName());
            step();
        }

        if (state == EventManager::RUNNING)
        {
            if (currentStep == modules.size())
            {
                setState(EventManager::FINISHED);
            }
            else
            {
                if (modules[currentStep-1]->stopAfterRender())
                {
                    std::cout << "stopAfterRender" << std::endl;
                    setState(EventManager::PAUSED);
                }
            }
        }

        /*
        int i = currentStep;
        for (i = 0; i < modules.size (); i++)
        {
            step();
        }
        */
    }
}

bool PipelineStepSingle::getConfigureBeforeProcessing()
{
    if (module->getVisualConfig() != NULL)
    {
        return module->getVisualConfig()->getConfigureBeforeProcessing();
    }

    return false;
}

void Pipeline::step()
{
    bool canRender = false;

    while (!canRender)
    {
        std::cout << "------------------------ Processing module " << modules[currentStep]->getName() << std::endl;

        modules[currentStep]->applyConfig(config);

        if (modules[currentStep]->getConfigureBeforeProcessing())
        {

            modules[currentStep]->startConfig(renderer);
            setState(EventManager::CONFIGURE);
            renderer->trigger(EventManager::UPDATE_RENDERER);

            EventManager::Event e = EventManager::NONE;

            while (e != EventManager::CONTINUE)
            {
                e = renderer->consumeBlocking();

                if (e == EventManager::CLOSE || e == EventManager::RESET)
                {
                    renderer->trigger(e);
                    return;
                }
            }

            setState(EventManager::RUNNING);
        }

        canRender = modules[currentStep]->process();

        std::cout << "------------------------ Finished module " << modules[currentStep]->getName() << std::endl;

        currentStep++;
    }

    renderer->trigger(EventManager::UPDATE_RENDERER);
}

}
