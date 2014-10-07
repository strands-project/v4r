#pragma once

#include <QSettings>
#include <QStringList>

#include <map>
#include "config.h"
#include "pipeline.h"

class AutoSaveConfig : public object_modeller::Config
{
public:
    typedef boost::shared_ptr<AutoSaveConfig> Ptr;

    AutoSaveConfig()
    {
        load();
    }

    void load()
    {
        std::cout << "load settings" << std::endl;
        QSettings s(QString("TU Wien"), QString("Object Modeller"));

        s.beginGroup("Settings");

        QStringList keys = s.allKeys();

        for (int i=0;i<keys.size();i++)
        {
            std::cout << "load key " << keys.at(i).toStdString() << " with value " << s.value(keys.at(i).toStdString().c_str()).toString().toStdString() << std::endl;
            parameters[keys.at(i).toStdString()] = s.value(keys.at(i).toStdString().c_str()).toString().toStdString();
        }

        s.endGroup();
    }

    void save()
    {
        QSettings s(QString("TU Wien"), QString("Object Modeller"));

        s.clear();

        s.beginGroup("Settings");

        for(std::map<std::string, std::string>::iterator iter = parameters.begin(); iter != parameters.end(); ++iter)
        {
            QString key = QString::fromStdString(iter->first);
            QString value = QString::fromStdString(iter->second);
            s.setValue(key, QVariant::fromValue(value));
        }

        s.endGroup();
    }
};

class Manager
{
private:
    AutoSaveConfig::Ptr config;
    object_modeller::Pipeline::Ptr pipeline;
    object_modeller::output::Renderer::Ptr renderer;


public:
    Manager();

    void setupPipeline(object_modeller::output::Renderer::Ptr renderer);

    object_modeller::Pipeline::Ptr getPipeline();

    AutoSaveConfig::Ptr getConfig()
    {
        return config;
    }

    void applyConfig(object_modeller::Config::Ptr config)
    {
        for (int i=0;i<getPipeline()->getSteps().size();i++)
        {
            getPipeline()->getSteps().at(i)->applyConfig(config);
        }
    }

    void applyParametersToConfig(object_modeller::Config::Ptr config)
    {
        config->clear();

        for (int i=0;i<getPipeline()->getSteps().size();i++)
        {
            object_modeller::PipelineStep *step = getPipeline()->getSteps().at(i);
            step->applyParametersToConfig(config);
        }
    }

};
