#pragma once

#include <string>
#include <map>
#include <vector>

//#include <cv.h>
#include <opencv2/opencv.hpp>

namespace object_modeller
{

class Config
{
private:
    std::map<std::string, std::string> parameters;
public:
    Config(std::string path);

    void printConfig();

    void overrideParameter(std::string key, std::string value);

    std::string getString(std::string key, std::string defaultValue);
    int getInt(std::string key, int defaultValue = 0);
    float getFloat(std::string key, float defaultValue = 0.0f);
    bool getBool(std::string key, bool defaultValue=false);
    std::vector<cv::Size> getCvSizeList(std::string key);
};

}
