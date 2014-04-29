#pragma once

#include "module.h"

namespace object_modeller
{

template<class RETURNTYPE, class ARG>
class InOutModule : public Module
{
public:
    typedef Result<RETURNTYPE> ResultType;
    virtual RETURNTYPE process(ARG arg) = 0;
};

}
