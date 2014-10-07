#pragma once

#include <string>
#include <iomanip>
#include <map>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

//#include <cv.h>
#include <opencv2/opencv.hpp>

#include <pcl/common/common.h>

namespace object_modeller
{

class Config
{
public:
    typedef boost::shared_ptr<Config> Ptr;
protected:
    std::map<std::string, std::string> parameters;

    std::string buildKey(std::string base_path, std::string key);

public:
    Config() {}

    void loadFromFile(std::string path);
    void saveToFile(std::string path);

    void clear();

    bool hasParameter(std::string base_path, std::string key);

    void printConfig();

    void overrideParameter(std::string key, std::string value);

    void applyValues(Config *config);

    std::string get(std::string base_path, std::string key);

    std::string getString(std::string base_path, std::string key, std::string defaultValue);
    int getInt(std::string base_path, std::string key, int defaultValue = 0);
    float getFloat(std::string base_path, std::string key, float defaultValue = 0.0f);
    bool getBool(std::string base_path, std::string key, bool defaultValue=false);
    std::vector<cv::Size> getCvSizeList(std::string base_path, std::string key);
};

template<class T>
class ParameterReader
{
public:
    virtual T fromString(std::string) = 0;
    virtual std::string toString(T value) = 0;
};

template<>
class ParameterReader<int>
{
public:
    int fromString(std::string s)
    {
        return atoi(s.c_str());
    }

    std::string toString(int v)
    {
        return boost::lexical_cast<std::string>(v);
    }
};

template<>
class ParameterReader<float>
{
public:
    float fromString(std::string s)
    {
        std::cout << "from string " << s << std::endl;
        return atof(s.c_str());
    }

    std::string toString(float v)
    {
        std::cout << "to string " << v << std::endl;
        /*
        std::stringstream ss;
        ss << std::setprecision(9);
        ss << v;
        return ss.str();
        */
        //return str( boost::format("%d") % v );
        return boost::lexical_cast<std::string>(v);
    }
};

template<>
class ParameterReader<std::string>
{
public:
    std::string fromString(std::string s)
    {
        return s;
    }

    std::string toString(std::string v)
    {
        return v;
    }
};

template<>
class ParameterReader<Eigen::Vector3f>
{
public:
    Eigen::Vector3f fromString(std::string s)
    {
        std::vector<std::string> v;
        boost::algorithm::split(v, s, boost::algorithm::is_any_of(";"));

        return Eigen::Vector3f(atof(v[0].c_str()), atof(v[1].c_str()), atof(v[2].c_str()));
    }

    std::string toString(Eigen::Vector3f v)
    {
        std::stringstream ss;

        ss << boost::lexical_cast<std::string>(v[0]);
        ss << ";";
        ss << boost::lexical_cast<std::string>(v[1]);
        ss << ";";
        ss << boost::lexical_cast<std::string>(v[2]);

        return ss.str();
    }
};

template<>
class ParameterReader<Eigen::Quaternionf>
{
public:
    Eigen::Quaternionf fromString(std::string s)
    {
        std::vector<std::string> v;
        boost::algorithm::split(v, s, boost::algorithm::is_any_of(";"));

        return Eigen::Quaternionf(atof(v[3].c_str()), atof(v[0].c_str()), atof(v[1].c_str()), atof(v[2].c_str()));
    }

    std::string toString(Eigen::Quaternionf v)
    {
        std::stringstream ss;

        ss << boost::lexical_cast<std::string>(v.x());
        ss << ";";
        ss << boost::lexical_cast<std::string>(v.y());
        ss << ";";
        ss << boost::lexical_cast<std::string>(v.z());
        ss << ";";
        ss << boost::lexical_cast<std::string>(v.w());

        return ss.str();
    }
};

template<>
class ParameterReader<bool>
{
public:
    bool fromString(std::string s)
    {
        return s == "true";
    }

    std::string toString(bool b)
    {
        if (b)
        {
            return "true";
        }

        return "false";
    }
};

template<>
class ParameterReader<std::vector<cv::Size> >
{
public:
    std::vector<cv::Size> fromString(std::string s)
    {
        std::vector<cv::Size> result;

        std::vector<std::string> sizes;
        boost::algorithm::split(sizes, s, boost::algorithm::is_any_of(";"));

        for (unsigned int i=0;i<sizes.size();i++)
        {
            boost::algorithm::trim(sizes[i]);
            int index = sizes[i].find(",");
            std::string x = sizes[i].substr(0, index);
            std::string y = sizes[i].substr(index + 1, sizes[i].length() - 1);

            cv::Size size(atoi(x.c_str()), atoi(y.c_str()));
            result.push_back(size);
        }

        return result;
    }

    std::string toString(std::vector<cv::Size> sizes)
    {
        std::stringstream ss;

        for (int i=0;i<sizes.size();i++)
        {
            if (i > 0)
            {
                ss << ";";
            }

            ss << sizes[i].width;
            ss << ",";
            ss << sizes[i].height;
        }

        return ss.str();
    }
};

class ParameterBase
{
public:
    enum ValueType
    {
        STRING,
        BOOL,
        INT,
        FLOAT,
        FOLDER,
        FILE,
        GUESS
    };

protected:
    ValueType type;
    std::string name;
    std::string base_path;
    std::string key;
public:

    ParameterBase()
    {
        type = STRING;
    }

    virtual void read(Config::Ptr config) = 0;

    virtual void fromString(std::string) = 0;
    virtual std::string toString() = 0;

    ValueType getType()
    {
        return type;
    }

    void setType(ValueType type)
    {
        this->type = type;
    }

    std::string getName()
    {
        return name;
    }

    std::string getFullKey()
    {
        std::stringstream config_key;
        config_key << base_path << "." << key;
        return config_key.str();
    }
};

template<class T>
class Parameter : public ParameterBase
{
private:
    T *value;
    T defaultValue;
public:
    Parameter(std::string base_path, std::string key, std::string name, T *value, T defaultValue, ValueType type = GUESS)
    {
        this->base_path = base_path;
        this->key = key;
        this->name = name;
        this->value = value;
        *(this->value) = defaultValue;
        this->defaultValue = defaultValue;

        if (type == GUESS)
        {
            this->type = guessType();
        } else {
            this->type = type;
        }
    }

    ValueType guessType()
    {
        return ParameterBase::STRING;
    }

    T *getValue()
    {
        return value;
    }

    void fromString(std::string s)
    {
        ParameterReader<T> r;
        *value = r.fromString(s);
    }

    std::string toString()
    {
        ParameterReader<T> r;
        return r.toString(*value);
    }

    void read(Config::Ptr config)
    {
        if (config->hasParameter(base_path, key))
        {
            std::string value_string = config->get(base_path, key);
            fromString(value_string);
        } else {
            *value = defaultValue;
        }
    }
};

template <>
inline ParameterBase::ValueType Parameter<bool>::guessType()
{
    return ParameterBase::BOOL;
}

template <>
inline ParameterBase::ValueType Parameter<int>::guessType()
{
    return ParameterBase::INT;
}

template <>
inline ParameterBase::ValueType Parameter<float>::guessType()
{
    return ParameterBase::FLOAT;
}



class ConfigItem
{
private:
    std::string config_name;
    std::vector<ParameterBase*> parameters;
public:
    ConfigItem(std::string config_name)
    {
        this->config_name = config_name;
    }

    std::vector<ParameterBase*> getParameters()
    {
        return parameters;
    }

    template<class T>
    void registerParameter(std::string key, std::string name, T *value, T defaultValue)
    {
        Parameter<T> *p = new Parameter<T>(config_name, key, name, value, defaultValue);
        parameters.push_back(p);
    }

    template<class T>
    void registerParameter(ParameterBase::ValueType type, std::string key, std::string name, T *value, T defaultValue)
    {
        Parameter<T> *p = new Parameter<T>(config_name, key, name, value, defaultValue, type);
        parameters.push_back(p);
    }

    virtual void applyParametersToConfig(Config::Ptr config)
    {
        for (int j=0;j<parameters.size();j++)
        {
            object_modeller::ParameterBase *param = parameters.at(j);

            config->overrideParameter(param->getFullKey(), param->toString());
        }
    }

    virtual void applyConfig(Config::Ptr config)
    {
        for (int i=0;i<parameters.size();i++)
        {
            parameters[i]->read(config);
        }
    }

    std::string getConfigName()
    {
        return config_name;
    }
};

}
