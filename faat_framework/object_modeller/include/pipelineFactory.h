#pragma once

#include <map>
#include <string>
#include <vector>

#include "pipeline.h"

namespace object_modeller
{

    class PipelineFactory
    {
        typedef void (PipelineFactory::*MethodPointer)(Pipeline::Ptr, output::Renderer::Ptr);
    private:
        std::map<std::string, MethodPointer> functions;
    public:
        PipelineFactory()
        {
            //functions.insert( std::make_pair( "Texturing Pipeline", &PipelineFactory::initTexturing ));
            functions.insert( std::make_pair( "Standard Pipeline", &PipelineFactory::initStandard ));
        }

        std::vector<std::string> getPipelines()
        {
            std::vector<std::string> result;

            for(std::map<std::string, MethodPointer>::iterator it = functions.begin(); it != functions.end(); ++it)
            {
              result.push_back(it->first);
            }

            return result;
        }

        Pipeline::Ptr create(std::string name, Config::Ptr config, output::Renderer::Ptr renderer);

    private:
        void initStandard(Pipeline::Ptr p, output::Renderer::Ptr renderer);
        void initTexturing(Pipeline::Ptr pipeline, output::Renderer::Ptr renderer);
    };

}
