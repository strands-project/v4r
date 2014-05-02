#pragma once

#include "module.h"

namespace object_modeller
{

template<class RETURNTYPE, class ARG>
class InOutModule : public Module
{
public:

    InOutModule(std::string config_name) : Module(config_name)
    {}

    typedef Result<RETURNTYPE> ResultType;
    virtual RETURNTYPE process(ARG arg) = 0;
};

}
