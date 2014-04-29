#pragma once

#include <string>

namespace object_modeller
{
namespace output
{

class RendererArgs
{
public:
    std::string name;
    bool step;

    RendererArgs() {}

    RendererArgs(std::string name, bool step)
    {
        this->name = name;
        this->step = step;
    }
};

}
}
