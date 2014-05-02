#include "config.h"

#include <stdlib.h>
#include <fstream>
#include <iostream>

#include <boost/algorithm/string.hpp>

namespace object_modeller
{

Config::Config(std::string path)
{
    std::ifstream in(path.c_str());
    std::string line;

    if (in.is_open())
    {
        //std::cout << "Configuration: " << std::endl << std::endl;

        while ( getline (in,line) )
        {
            // std::cout << "read config line " << line << std::endl;

            boost::algorithm::trim(line);
            if (line.length() > 0 && line.at(0) != '#')
            {
                int index = line.find_first_of('=');
                std::string key = line.substr(0, index);
                std::string value = line.substr(index + 1, line.length() - 1);

                boost::algorithm::trim(key);
                boost::algorithm::trim(value);

                //std::cout << "key=" << key << std::endl;
                //std::cout << "value=" << value << std::endl;

                //std::cout << key << " = " << value << std::endl;

                parameters[key] = value;
            }
        }
        in.close();
    }
    else
    {
        std::cout << "Could not open config file" << std::endl;
    }
}

void Config::printConfig()
{
    std::cout << "Configuration: " << std::endl << std::endl;

    std::map<std::string, std::string>::iterator iter;

    for (iter = parameters.begin(); iter != parameters.end(); ++iter)
    {
        std::cout << iter->first << " = " << iter->second << std::endl;
    }
}

void Config::overrideParameter(std::string key, std::string value)
{
    parameters[key] = value;
}

std::string Config::buildKey(std::string base_path, std::string key)
{
    std::stringstream config_key;
    config_key << base_path << "." << key;
    return config_key.str();
}

std::string Config::getString(std::string base_path, std::string key, std::string defaultValue)
{
    std::string config_key = buildKey(base_path, key);

    if (parameters.count(config_key))
    {
        return parameters[config_key];
    }

    return defaultValue;
}

int Config::getInt(std::string base_path, std::string key, int defaultValue)
{
    std::string config_key = buildKey(base_path, key);

    if (parameters.count(config_key))
    {
        return atoi(parameters[config_key].c_str());
    }

    return defaultValue;
}

float Config::getFloat(std::string base_path, std::string key, float defaultValue)
{
    std::string config_key = buildKey(base_path, key);

    if (parameters.count(config_key))
    {
        return atof(parameters[config_key].c_str());
    }

    return defaultValue;
}

bool Config::getBool(std::string base_path, std::string key, bool defaultValue)
{
    std::string config_key = buildKey(base_path, key);

    if (parameters.count(config_key))
    {
        return parameters[config_key] == "true";
    }

    return defaultValue;
}

std::vector<cv::Size> Config::getCvSizeList(std::string base_path, std::string key2)
{
    std::string config_key = buildKey(base_path, key2);

    std::vector<cv::Size> result;

    if (parameters.count(config_key))
    {
        std::vector<std::string> sizes;
        boost::algorithm::split(sizes, parameters[config_key], boost::algorithm::is_any_of(";"));

        for (unsigned int i=0;i<sizes.size();i++)
        {
            boost::algorithm::trim(sizes[i]);
            int index = sizes[i].find(",");
            std::string x = sizes[i].substr(0, index);
            std::string y = sizes[i].substr(index + 1, sizes[i].length() - 1);

            cv::Size size(atoi(x.c_str()), atoi(y.c_str()));
            result.push_back(size);
        }
    }

    return result;
}

}
