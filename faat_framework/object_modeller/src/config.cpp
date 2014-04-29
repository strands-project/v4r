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

std::string Config::getString(std::string key, std::string defaultValue)
{
    if (parameters.count(key))
    {
        return parameters[key];
    }

    return defaultValue;
}

int Config::getInt(std::string key, int defaultValue)
{
    if (parameters.count(key))
    {
        return atoi(parameters[key].c_str());
    }

    return defaultValue;
}

float Config::getFloat(std::string key, float defaultValue)
{
    if (parameters.count(key))
    {
        return atof(parameters[key].c_str());
    }

    return defaultValue;
}

bool Config::getBool(std::string key, bool defaultValue)
{
    if (parameters.count(key))
    {
        return parameters[key] == "true";
    }

    return defaultValue;
}

std::vector<cv::Size> Config::getCvSizeList(std::string key)
{
    std::vector<cv::Size> result;

    if (parameters.count(key))
    {
        std::vector<std::string> sizes;
        boost::algorithm::split(sizes, parameters[key], boost::algorithm::is_any_of(";"));

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
