#pragma once

#include "module.h"
#include "output/renderer.h"

namespace object_modeller
{
    class VisualConfigBase
    {
    private:
        Module *module;
        bool configureBeforeProcessing;
    protected:
        VisualConfigBase(Module *module)
        {
            std::cout << "create visual config" << std::endl;
            this->module = module;
        }

    public:
        void init()
        {
            std::cout << "register visual config param" << std::endl;
            module->registerParameter("configureBeforeProcessing", "Configure before processing", &configureBeforeProcessing, true);
        }

        bool getConfigureBeforeProcessing()
        {
            return configureBeforeProcessing;
        }

        virtual void startConfig(output::Renderer::Ptr renderer) = 0;
    };

    /*
    template<class T>
    class ConfigHandler
    {
    public:
        virtual void startConfig(output::Renderer::Ptr renderer) = 0;
    };

    template<class T>
    class VisualConfigModule : VisualConfigBase
    {
    private:
        ConfigHandler<T> configHandler;
    public:
        VisualConfigModule(Module *module) : VisualConfigBase(module)
        {

        }

        virtual void startConfig(output::Renderer::Ptr renderer)
        {
            configHandler.startConfig(renderer);
        }
    };
    */

}
