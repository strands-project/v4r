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
public:
    virtual void applyConfig(Config &config) = 0;

    virtual std::string getName()
    {
        return "base";
    }
};

}
