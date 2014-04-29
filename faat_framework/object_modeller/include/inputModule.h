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
    typedef Result<RETURNTYPE> ResultType;

    virtual RETURNTYPE process() = 0;
};

}
