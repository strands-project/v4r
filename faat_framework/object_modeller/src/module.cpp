
#include "module.h"

#include "visualConfig.h"

namespace object_modeller
{

    Module::Module(std::string config_name, VisualConfigBase *visualConfig) : ConfigItem(config_name)
    {
        std::cout << "module ctor" << std::endl;
        this->visualConfig = visualConfig;

        if (this->visualConfig != NULL)
        {
            this->visualConfig->init();
        }
    }
}
