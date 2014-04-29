#pragma once

#include "module.h"
#include "inputModule.h"
#include "outputModule.h"
#include "ioModule.h"
#include "config.h"

#include <vector>

namespace object_modeller
{

class PipelineStep
{
public:
    virtual void process() = 0;
    virtual Module* getModule() = 0;
};

template<class T>
class PipelineStepIn : public PipelineStep
{
private:
    InModule<T> *module;
    Result<T> *result;
public:
    PipelineStepIn(InModule<T> *module, Result<T> *result)
    {
        this->module = module;
        this->result = result;
    }
    void process()
    {
        result->setResult(module->process());
    }
    Module* getModule()
    {
        return module;
    }
};

template<class T>
class PipelineStepOut : public PipelineStep
{
private:
    OutModule<T> *module;
    ResultBase<T> *param;
public:
    PipelineStepOut(OutModule<T> *module, ResultBase<T> *param)
    {
        this->module = module;
        this->param = param;
    }
    void process()
    {
        module->process(param->getResult());
    }
    Module* getModule()
    {
        return module;
    }
};

template<class T, class U>
class PipelineStepInOut : public PipelineStep
{
private:
    InOutModule<T, U> *module;
    Result<T> *result;
    ResultBase<U> *param;
public:
    PipelineStepInOut(InOutModule<T, U> *module, Result<T> *result, ResultBase<U> *param)
    {
        this->module = module;
        this->result = result;
        this->param = param;
    }

    void process()
    {
        result->setResult(module->process(param->getResult()));
    }
    Module* getModule()
    {
        return module;
    }
};


class Pipeline
{
private:
    std::vector<PipelineStep*> modules;
    Config &config;
public:
    Pipeline(Config &config) : config(config)
    {
    }

    template<class T>
    Result<T> *addIn(InModule<T> *inModule)
    {
        Result<T> *result = new Result<T>();
        PipelineStepIn<T> *step = new PipelineStepIn<T>(inModule, result);
        modules.push_back(step);
        return result;
    }

    template<class T>
    void addOut(OutModule<T> *outModule, ResultBase<T> *param)
    {
        PipelineStepOut<T> *step = new PipelineStepOut<T>(outModule, param);
        modules.push_back(step);
    }

    template<class ARG1, class ARG2>
    void addOut(OutModule<boost::tuples::tuple<ARG1, ARG2> > *outModule, ResultBase<ARG1> *arg1, ResultBase<ARG2> *arg2)
    {
        boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>& > *param
                = new boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>& >(boost::ref(*arg1), boost::ref(*arg2));

        ResultAggregator<boost::tuples::tuple<ARG1, ARG2>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2> &> > *wrappedParam = new ResultAggregator<boost::tuples::tuple<ARG1, ARG2>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2> &> >(param);
        PipelineStepOut<boost::tuples::tuple<ARG1, ARG2> > *step = new PipelineStepOut<boost::tuples::tuple<ARG1, ARG2> >(outModule, wrappedParam);
        modules.push_back(step);
    }

    template<class ARG1, class ARG2, class ARG3>
    void addOut(OutModule<boost::tuples::tuple<ARG1, ARG2, ARG3> > *outModule, ResultBase<ARG1> *arg1, ResultBase<ARG2> *arg2, ResultBase<ARG3> *arg3)
    {
        boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>&, ResultBase<ARG3>& > *param
                = new boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>&, ResultBase<ARG3>& >(boost::ref(*arg1), boost::ref(*arg2), boost::ref(*arg3));

        ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2> &, ResultBase<ARG3> &> > *wrappedParam = new ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2> &, ResultBase<ARG3> &> >(param);
        PipelineStepOut<boost::tuples::tuple<ARG1, ARG2, ARG3> > *step = new PipelineStepOut<boost::tuples::tuple<ARG1, ARG2, ARG3> >(outModule, wrappedParam);
        modules.push_back(step);
    }

    template<class T, class U>
    Result<T> *addInOut(InOutModule<T, U> *inOutModule, ResultBase<U> *param)
    {
        Result<T> *result = new Result<T>();
        PipelineStepInOut<T, U> *step = new PipelineStepInOut<T, U>(inOutModule, result, param);
        modules.push_back(step);
        return result;
    }

    template<class T, class ARG1, class ARG2>
    Result<T> *addInOut(InOutModule<T, boost::tuples::tuple<ARG1, ARG2> > *inOutModule, ResultBase<ARG1> *arg1, ResultBase<ARG2> *arg2)
    {
        Result<T> *result = new Result<T>();

        boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>& > *param =
                new boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>& >(boost::ref(*arg1), boost::ref(*arg2));

        ResultAggregator<boost::tuples::tuple<ARG1, ARG2>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>& > > *wrappedParam = new ResultAggregator<boost::tuples::tuple<ARG1, ARG2>, boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>& > >(param);

        PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2> > *step = new PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2> >(inOutModule, result, wrappedParam);
        modules.push_back(step);
        return result;
    }

    template<class T, class ARG1, class ARG2, class ARG3>
    Result<T> *addInOut(InOutModule<T, boost::tuples::tuple<ARG1, ARG2, ARG3> > *inOutModule, ResultBase<ARG1> *arg1, ResultBase<ARG2> *arg2, ResultBase<ARG3> *arg3)
    {
        Result<T> *result = new Result<T>();

        boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>& > *param =
                new boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>& >(boost::ref(*arg1), boost::ref(*arg2), boost::ref(*arg3));

        ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>& > > *wrappedParam = new ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3>, boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>&, ResultBase<ARG3>& > >(param);

        PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2, ARG3> > *step = new PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2, ARG3> >(inOutModule, result, wrappedParam);
        modules.push_back(step);
        return result;
    }

    template<class T, class ARG1, class ARG2, class ARG3, class ARG4, class ARG5>
    Result<T> *addInOut(InOutModule<T, boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4, ARG5> > *inOutModule, ResultBase<ARG1> *arg1, ResultBase<ARG2> *arg2, ResultBase<ARG3> *arg3, ResultBase<ARG4> *arg4, ResultBase<ARG5> *arg5)
    {
        Result<T> *result = new Result<T>();

        boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>& , ResultBase<ARG4>& , ResultBase<ARG5>& > *param =
                new boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>&, ResultBase<ARG4>&, ResultBase<ARG5>& >(boost::ref(*arg1), boost::ref(*arg2), boost::ref(*arg3), boost::ref(*arg4), boost::ref(*arg5));

        ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4, ARG5>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>&, ResultBase<ARG4>&, ResultBase<ARG5>& > > *wrappedParam = new ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4, ARG5>, boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>&, ResultBase<ARG3>&, ResultBase<ARG4>&, ResultBase<ARG5>& > >(param);

        PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4, ARG5> > *step = new PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4, ARG5> >(inOutModule, result, wrappedParam);
        modules.push_back(step);
        return result;
    }

    void process();
};

}
