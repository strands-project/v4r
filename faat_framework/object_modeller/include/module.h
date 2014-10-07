#pragma once

#include <string>
#include <iostream>

#include <boost/tuple/tuple.hpp>

#include "result.h"
#include "config.h"
#include "multisequence.h"

namespace object_modeller
{

class VisualConfigBase;

class Module : public ConfigItem
{
protected:
    int activeSequence;
    VisualConfigBase *visualConfig;
public:

    Module(std::string config_name, VisualConfigBase *visualConfig = NULL);

    VisualConfigBase *getVisualConfig()
    {
        return visualConfig;
    }

    void setActiveSequence(int sequence)
    {
        this->activeSequence = sequence;
    }

    virtual std::string getName()
    {
        return "base";
    }
};

}
