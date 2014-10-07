#pragma once

#include "module.h"

#include <boost/shared_ptr.hpp>

namespace object_modeller
{

template<class ARG>
class OutModule : public Module
{
protected:
    int nrInputSequences;
public:

    OutModule(std::string config_name) : Module(config_name)
    {
        nrInputSequences = 1;
    }

    typedef OutModule<ARG> Type;

    void setNrInputSequences(int nrInputSequences)
    {
        this->nrInputSequences = nrInputSequences;
    }

    virtual void process(ARG arg) = 0;
};

}
