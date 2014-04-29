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

class Renderer : public OutModule<boost::tuples::tuple<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>, std::string, bool> >
{
public:
    virtual void renderMesh(pcl::PolygonMesh::Ptr, std::string, bool) = 0;
};

}
}
