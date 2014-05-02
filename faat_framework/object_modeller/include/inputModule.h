#pragma once

#include "module.h"
#include "result.h"

#include <boost/shared_ptr.hpp>

namespace object_modeller
{

template<class RETURNTYPE>
class InModule : public Module
{
public:

    InModule(std::string config_name) : Module(config_name)
    {}

    typedef Result<RETURNTYPE> ResultType;

    virtual RETURNTYPE process() = 0;
};

}
