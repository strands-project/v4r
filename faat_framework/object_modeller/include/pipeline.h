#pragma once

#include "eventManager.h"
#include "module.h"
#include "ioModule.h"
#include "inputModule.h"
#include "outputModule.h"
#include "config.h"
#include "multisequence.h"

#include "resultAggregator.h"

#include "visualConfig.h"

#include <vector>

namespace object_modeller
{

template<class T>
class ResultPointer
{
private:
    ResultBase<T> *result;
public:
    ResultPointer()
    {
        std::cout << "empty ctor" << std::endl;
        result = NULL;
    }

    ResultPointer(ResultBase<T> *result)
    {
        std::cout << "standard ctor" << std::endl;
        this->result = result;
    }

    ResultPointer(const ResultPointer& obj)
    {
        std::cout << "copy ctor" << std::endl;
        result = obj.result;
    }

    ResultBase<T> *get()
    {
        return result;
    }

    ResultPointer<T>& operator= (const ResultPointer<T> &cSource)
    {
        std::cout << "assignment" << std::endl;

        if (result != NULL)
        {
            std::cout << "second assignment" << std::endl;
            cSource.result->setFallback(result);
            std::cout << "second assignment ok" << std::endl;
        }

        result = cSource.result;
        return *this;
    }
};

class PipelineStep
{
protected:
    std::string name;
public:
    PipelineStep(std::string name)
    {
        this->name = name;
    }

    virtual std::vector<ParameterBase*> getParameters() = 0;

    virtual bool process() = 0;

    virtual std::string getName()
    {
        return name;
    }

    virtual bool stopAfterRender()
    {
        return false;
    }

    virtual bool getConfigureBeforeProcessing() = 0;

    virtual bool startConfig(output::Renderer::Ptr renderer)
    {
        std::cout << "Warning: No config handler for module" << std::endl;
        renderer->trigger(EventManager::CONTINUE);
    }

    virtual void applyConfig(Config::Ptr config) = 0;

    virtual void applyParametersToConfig(Config::Ptr config) = 0;
};

class PipelineStepSingle : public PipelineStep
{
private:
    Module* module;
public:
    PipelineStepSingle(Module *module) : PipelineStep(module->getName())
    {
        this->module = module;
    }

    virtual void applyConfig(Config::Ptr config)
    {
        module->applyConfig(config);
    }

    virtual void applyParametersToConfig(Config::Ptr config)
    {
        module->applyParametersToConfig(config);
    }

    virtual bool startConfig(output::Renderer::Ptr renderer)
    {
        VisualConfigBase *v = module->getVisualConfig();

        if (v != NULL)
        {
            v->startConfig(renderer);
        }
        else
        {
            PipelineStep::startConfig(renderer);
        }
    }

    virtual std::vector<ParameterBase*> getParameters()
    {
        return module->getParameters();
    }

    virtual bool getConfigureBeforeProcessing();
};

template<class T>
class PipelineStepIn : public PipelineStepSingle
{
private:
    InModule<T> *module;
    Result<T> *result;
    output::Renderer::Ptr renderer;
public:
    PipelineStepIn(output::Renderer::Ptr renderer, InModule<T> *module, Result<T> *result) : PipelineStepSingle(module)
    {
        this->renderer = renderer;
        this->module = module;
        this->result = result;
    }
    bool process()
    {
        int outSequences = module->getNrOutputSequences();

        if (outSequences == -1) {
            outSequences = 1;
        }

        std::cout << "out sequences " << outSequences << std::endl;

        for (int i=0;i<outSequences;i++)
        {
            std::cout << "++++++++++++++++++++ Sequence " << i << std::endl;
            module->setActiveSequence(i);
            result->setResult(i, module->process());

            outSequences = module->getNrOutputSequences();

            std::cout << "out sequences " << outSequences << std::endl;
        }

        if (outSequences == -1) {
            outSequences = 1;
        }

        std::cout << "out sequences " << outSequences << std::endl;

        if (module->canRender())
        {
            renderer->clear();

            std::cout << "add clouds to renderer" << outSequences << " " << module->getNrOutputSequences() << std::endl;

            for (int i=0;i<outSequences;i++)
            {
                std::cout << "add clouds" << i << std::endl;
                module->renderResult(renderer, i, result->getResult(i));
            }
        }

        return module->canRender();
    }

    virtual bool stopAfterRender()
    {
        return module->getStopAfterRender();
    }
};

template<class T>
class PipelineStepOut : public PipelineStepSingle
{
private:
    OutModule<T> *module;
    ResultBase<T> *param;
public:
    PipelineStepOut(OutModule<T> *module, ResultBase<T> *param) : PipelineStepSingle(module)
    {
        this->module = module;
        this->param = param;
    }
    bool process()
    {
        int nrSequences = param->getNrSequences();

        for (int i=0;i<nrSequences;i++)
        {
            module->setActiveSequence(i);
            module->setNrInputSequences(nrSequences);
            module->process(param->getResult(i));
        }

        return false;
    }
};

template<class T, class U>
class PipelineStepInOut : public PipelineStepSingle
{
private:
    InOutModule<T, U> *module;
    Result<T> *result;
    ResultBase<U> *param;
    output::Renderer::Ptr renderer;
public:
    PipelineStepInOut(output::Renderer::Ptr renderer, InOutModule<T, U> *module, Result<T> *result, ResultBase<U> *param) : PipelineStepSingle(module)
    {
        this->renderer = renderer;
        this->module = module;
        this->result = result;
        this->param = param;
    }

    bool process()
    {
        int outSequences = module->getNrOutputSequences();
        int inSequences = param->getNrSequences();

        if (outSequences == -1) {
            outSequences = inSequences;
        }

        for (int i=0;i<inSequences;i++)
        {
            std::cout << "++++++++++++++++++++ Sequence " << i << " in: " << inSequences << " out: " << outSequences << std::endl;
            module->setActiveSequence(i);
            module->setNrInputSequences(inSequences);
            int targetSequence = std::min(i, outSequences - 1);
            result->setResult(targetSequence, module->process(param->getResult(i)));
        }

        if (outSequences == -1) {
            outSequences = inSequences;
        }

        if (module->canRender())
        {
            renderer->clear();

            for (int i=0;i<inSequences;i++)
            {
                int targetSequence = std::min(i, outSequences - 1);

                module->renderResult(renderer, targetSequence, result->getResult(targetSequence));
            }
        }

        return module->canRender();
    }

    virtual bool stopAfterRender()
    {
        return module->getStopAfterRender();
    }
};

class PipelineStepMulti : public PipelineStep, public ConfigItem
{
private:
    std::vector<PipelineStep*> steps;
    std::string config_name;
public:
    PipelineStepMulti(std::string configName, std::string displayName) : PipelineStep(displayName), ConfigItem(configName)
    {
    }

    std::vector<PipelineStep*> getSteps()
    {
        return steps;
    }

    void add(PipelineStep* step)
    {
        steps.push_back(step);
    }

    bool process()
    {
        bool canRender = false;

        for (int i=0;i<steps.size();i++)
        {
            canRender = steps.at(i)->process();
        }

        return canRender;
    }

    virtual bool stopAfterRender()
    {
        return steps.at(steps.size() - 1)->stopAfterRender();
    }

    virtual void applyConfig(Config::Ptr config)
    {
        ConfigItem::applyConfig(config);

        for (int i=0;i<steps.size();i++)
        {
            steps.at(i)->applyConfig(config);
        }
    }

    virtual void applyParametersToConfig(Config::Ptr config)
    {
        ConfigItem::applyParametersToConfig(config);

        for (int i=0;i<steps.size();i++)
        {
            steps.at(i)->applyParametersToConfig(config);
        }
    }
};

class PipelineStepOptional : public PipelineStepMulti
{
private:
    bool enabled;
public:
    PipelineStepOptional(std::string configName, std::string displayName) : PipelineStepMulti(configName, displayName)
    {
        registerParameter("enabled", "Enabled", &enabled, true);
    }

    bool process()
    {
        bool canRender = false;

        if (enabled)
        {
            for (int i=0;i<getSteps().size();i++)
            {
                canRender = getSteps().at(i)->process();
            }
        }

        return canRender;
    }

    virtual bool getConfigureBeforeProcessing()
    {
        return getSteps().at(getSteps().size() - 1)->getConfigureBeforeProcessing();
    }

    virtual std::vector<ParameterBase*> getParameters()
    {
        return ConfigItem::getParameters();
    }
};

class PipelineStepChoiceBase : public PipelineStepMulti
{
private:
    int activeChoice;
public:
    PipelineStepChoiceBase(std::string configName, std::string displayName) : PipelineStepMulti(configName, displayName)
    {
        registerParameter("type", "Type", &activeChoice, 0);
    }

    void setActiveChoice(int choice)
    {
        activeChoice = choice;
    }

    virtual bool stopAfterRender()
    {
        return getSteps().at(activeChoice)->stopAfterRender();
    }

    int getActiveChoice()
    {
        return activeChoice;
    }

    virtual bool getConfigureBeforeProcessing()
    {
        return getSteps().at(activeChoice)->getConfigureBeforeProcessing();
    }

    bool process()
    {
        return getSteps().at(activeChoice)->process();
    }

    virtual bool startConfig(output::Renderer::Ptr renderer)
    {
        getSteps().at(activeChoice)->startConfig(renderer);
    }

    virtual std::vector<ParameterBase*> getParameters()
    {
        return getSteps().at(activeChoice)->getParameters();
    }
};

template<class T>
class PipelineStepChoice : public PipelineStepChoiceBase
{
private:
    Result<T> *result;
public:
    PipelineStepChoice(std::string configName, std::string displayName, Result<T> *result) : PipelineStepChoiceBase(configName, displayName)
    {
        this->result = result;
    }

    Result<T> *getResult()
    {
        return result;
    }

};

class Pipeline
{
public:
    typedef boost::shared_ptr<Pipeline> Ptr;
private:
    EventManager::State state;
    int currentStep;
    std::vector<PipelineStep*> modules;
    Config::Ptr config;
    SequenceInfo sequence_info;

    PipelineStepMulti *activeSubPipeline;
    output::Renderer::Ptr renderer;
public:
    Pipeline(output::Renderer::Ptr renderer, Config::Ptr config) : config(config)
    {
        this->state = EventManager::PAUSED;
        currentStep = 0;
        activeSubPipeline = NULL;
        this->renderer = renderer;
    }

    EventManager::State getState()
    {
        return state;
    }

    std::string getActiveStepName()
    {
        return modules[currentStep]->getName();
    }

    void setState(EventManager::State state, std::string activeStepName="");

    template<class T>
    ResultPointer<T> beginChoice(std::string configName, std::string displayName)
    {
        Result<T> *result = new Result<T>();
        activeSubPipeline = new PipelineStepChoice<T>(configName, displayName, result);
        add(activeSubPipeline);
        return ResultPointer<T>(result);
    }

    void beginOptional(std::string configName, std::string displayName)
    {
        activeSubPipeline = new PipelineStepOptional(configName, displayName);
        add(activeSubPipeline);
    }

    template<class T>
    Result<T> *getActiveResult()
    {
        if (activeSubPipeline != NULL)
        {
            PipelineStepChoice<T> *choice = dynamic_cast<PipelineStepChoice<T> *>(activeSubPipeline);

            if (choice != NULL)
            {
                return choice->getResult();
            }
        }

        return new Result<T>();
    }

    void endChoice()
    {
        activeSubPipeline = NULL;
    }

    void endOptional()
    {
        activeSubPipeline = NULL;
    }

    std::vector<PipelineStep*> getSteps()
    {
        return modules;
    }

    void add(PipelineStep *step)
    {
        if (activeSubPipeline == NULL || activeSubPipeline == step)
        {
            modules.push_back(step);
        }
        else
        {
            activeSubPipeline->add(step);
        }
    }

    template<class T>
    ResultPointer<T> addIn(InModule<T> *inModule)
    {
        Result<T> *result = getActiveResult<T>();
        PipelineStepIn<T> *step = new PipelineStepIn<T>(renderer, inModule, result);
        add(step);
        return ResultPointer<T>(result);
    }

    template<class T>
    void addOut(OutModule<T> *outModule, ResultPointer<T> param)
    {
        PipelineStepOut<T> *step = new PipelineStepOut<T>(outModule, param.get());
        add(step);
    }

    template<class ARG1, class ARG2>
    void addOut(OutModule<boost::tuples::tuple<ARG1, ARG2> > *outModule, ResultPointer<ARG1> arg1, ResultPointer<ARG2> arg2)
    {
        boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>& > *param
                = new boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>& >(boost::ref(*arg1.get()), boost::ref(*arg2.get()));

        ResultAggregator<boost::tuples::tuple<ARG1, ARG2>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2> &> > *wrappedParam = new ResultAggregator<boost::tuples::tuple<ARG1, ARG2>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2> &> >(param);
        PipelineStepOut<boost::tuples::tuple<ARG1, ARG2> > *step = new PipelineStepOut<boost::tuples::tuple<ARG1, ARG2> >(outModule, wrappedParam);
        add(step);
    }

    template<class ARG1, class ARG2, class ARG3>
    void addOut(OutModule<boost::tuples::tuple<ARG1, ARG2, ARG3> > *outModule, ResultPointer<ARG1> arg1, ResultPointer<ARG2> arg2, ResultPointer<ARG3> arg3)
    {
        boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>&, ResultBase<ARG3>& > *param
                = new boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>&, ResultBase<ARG3>& >(boost::ref(*arg1.get()), boost::ref(*arg2.get()), boost::ref(*arg3.get()));

        ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2> &, ResultBase<ARG3> &> > *wrappedParam = new ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2> &, ResultBase<ARG3> &> >(param);
        PipelineStepOut<boost::tuples::tuple<ARG1, ARG2, ARG3> > *step = new PipelineStepOut<boost::tuples::tuple<ARG1, ARG2, ARG3> >(outModule, wrappedParam);
        add(step);
    }

    template<class T, class U>
    ResultPointer<T> addInOut(InOutModule<T, U> *inOutModule, ResultPointer<U> param)
    {
        Result<T> *result = getActiveResult<T>();
        PipelineStepInOut<T, U> *step = new PipelineStepInOut<T, U>(renderer, inOutModule, result, param.get());
        add(step);
        return ResultPointer<T>(result);
    }

    template<class T, class ARG1, class ARG2>
    ResultPointer<T> addInOut(InOutModule<T, boost::tuples::tuple<ARG1, ARG2> > *inOutModule, ResultPointer<ARG1> arg1, ResultPointer<ARG2> arg2)
    {
        Result<T> *result = getActiveResult<T>();

        boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>& > *param =
                new boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>& >(boost::ref(*arg1.get()), boost::ref(*arg2.get()));

        ResultAggregator<boost::tuples::tuple<ARG1, ARG2>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>& > > *wrappedParam = new ResultAggregator<boost::tuples::tuple<ARG1, ARG2>, boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>& > >(param);

        PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2> > *step = new PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2> >(renderer, inOutModule, result, wrappedParam);
        add(step);
        return ResultPointer<T>(result);
    }

    template<class T, class ARG1, class ARG2, class ARG3>
    ResultPointer<T> addInOut(InOutModule<T, boost::tuples::tuple<ARG1, ARG2, ARG3> > *inOutModule, ResultPointer<ARG1> arg1, ResultPointer<ARG2> arg2, ResultPointer<ARG3> arg3)
    {
        Result<T> *result = getActiveResult<T>();

        boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>& > *param =
                new boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>& >(boost::ref(*arg1.get()), boost::ref(*arg2.get()), boost::ref(*arg3.get()));

        ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>& > > *wrappedParam = new ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3>, boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>&, ResultBase<ARG3>& > >(param);

        PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2, ARG3> > *step = new PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2, ARG3> >(renderer, inOutModule, result, wrappedParam);
        add(step);
        return ResultPointer<T>(result);
    }

    template<class T, class ARG1, class ARG2, class ARG3, class ARG4>
    ResultPointer<T> addInOut(InOutModule<T, boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4> > *inOutModule, ResultPointer<ARG1> &arg1, ResultPointer<ARG2> arg2, ResultPointer<ARG3> arg3, ResultPointer<ARG4> arg4)
    {
        Result<T> *result = getActiveResult<T>();

        boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>& , ResultBase<ARG4>&> *param =
                new boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>&, ResultBase<ARG4>&>(boost::ref(*arg1.get()), boost::ref(*arg2.get()), boost::ref(*arg3.get()), boost::ref(*arg4.get()));

        ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>&, ResultBase<ARG4>& > > *wrappedParam = new ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4>, boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>&, ResultBase<ARG3>&, ResultBase<ARG4>&> >(param);

        PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4> > *step = new PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4> >(renderer, inOutModule, result, wrappedParam);
        add(step);
        return ResultPointer<T>(result);
    }

    template<class T, class ARG1, class ARG2, class ARG3, class ARG4, class ARG5>
    ResultPointer<T> addInOut(InOutModule<T, boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4, ARG5> > *inOutModule, ResultPointer<ARG1> arg1, ResultPointer<ARG2> arg2, ResultPointer<ARG3> arg3, ResultPointer<ARG4> arg4, ResultPointer<ARG5> arg5)
    {
        Result<T> *result = getActiveResult<T>();

        boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>& , ResultBase<ARG4>& , ResultBase<ARG5>& > *param =
                new boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>&, ResultBase<ARG4>&, ResultBase<ARG5>& >(boost::ref(*arg1.get()), boost::ref(*arg2.get()), boost::ref(*arg3.get()), boost::ref(*arg4.get()), boost::ref(*arg5.get()));

        ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4, ARG5>, boost::tuples::tuple<ResultBase<ARG1> &, ResultBase<ARG2>&, ResultBase<ARG3>&, ResultBase<ARG4>&, ResultBase<ARG5>& > > *wrappedParam = new ResultAggregator<boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4, ARG5>, boost::tuples::tuple<ResultBase<ARG1>& , ResultBase<ARG2>&, ResultBase<ARG3>&, ResultBase<ARG4>&, ResultBase<ARG5>& > >(param);

        PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4, ARG5> > *step = new PipelineStepInOut<T,boost::tuples::tuple<ARG1, ARG2, ARG3, ARG4, ARG5> >(renderer, inOutModule, result, wrappedParam);
        add(step);
        return ResultPointer<T>(result);
    }

    void process(bool autorun=true);
    void step();
};

}
