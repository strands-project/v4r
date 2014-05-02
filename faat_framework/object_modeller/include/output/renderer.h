#pragma once

#include "outputModule.h"
#include "module.h"
#include "rendererArgs.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace object_modeller
{
namespace output
{

class Renderer
{
public:

    virtual void applyConfig(Config &config, std::string base_path) = 0;

    virtual std::string getName() = 0;

    virtual void renderPointClouds(std::vector<typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr> point_clouds, std::string name, bool step) = 0;
    virtual void renderPointClouds(std::vector<typename pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> point_clouds, std::string name, bool step) = 0;

    virtual void renderMesh(pcl::PolygonMesh::Ptr, std::string, bool) = 0;
};

}
}
