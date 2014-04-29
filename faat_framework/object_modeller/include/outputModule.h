#pragma once

#include "module.h"

#include <boost/shared_ptr.hpp>

namespace object_modeller
{

template<class ARG>
class OutModule : public Module
{
public:
    typedef OutModule<ARG> Type;

    virtual void process(ARG arg) = 0;
};

}
