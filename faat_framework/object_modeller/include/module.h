#pragma once

#include <string>
#include <iostream>

#include <boost/tuple/tuple.hpp>

#include "result.h"
#include "config.h"

namespace object_modeller
{

class Module
{
private:
    std::string config_name;
public:

    Module(std::string config_name) : config_name(config_name)
    {}

    virtual void applyConfig(Config &config) = 0;

    std::string getConfigName()
    {
        return config_name;
    }

    virtual std::string getName()
    {
        return "base";
    }
};

}
