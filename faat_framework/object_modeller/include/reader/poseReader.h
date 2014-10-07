#pragma once

#include "inputModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace reader
{

class PoseReader : public InModule<std::vector<Eigen::Matrix4f> >
{
private:
    std::string sequencePrefix;
    std::string pattern;
    std::string inputPath;
    int step;
    int nrSequences;

public:
    PoseReader(std::string config_name="posereader");

    virtual std::vector<Eigen::Matrix4f> process();

    Eigen::Matrix4f readPose(std::string filename);

    std::string getSequenceFolder();

    virtual int getNrOutputSequences()
    {
        return nrSequences;
    }

    std::string getName()
    {
        return "Pose Reader";
    }
};

}
}
