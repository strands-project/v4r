#pragma once

#include "outputModule.h"
#include "module.h"
#include "rendererArgs.h"
#include "renderer.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace object_modeller
{
namespace output
{

template<class TPointType>
class PointCloudRenderer : public OutModule<boost::tuples::tuple<std::vector<typename pcl::PointCloud<TPointType>::Ptr>, std::string, bool> >
{
private:
    boost::shared_ptr<Renderer> baseRenderer;

public:

    PointCloudRenderer(boost::shared_ptr<Renderer> baseRenderer, std::string config_name="renderer")
        : OutModule<boost::tuples::tuple<std::vector<typename pcl::PointCloud<TPointType>::Ptr>, std::string, bool> >(config_name)
    {
        this->baseRenderer = baseRenderer;
    }

    void applyConfig(Config &config)
    {
    }

    void process(boost::tuples::tuple<std::vector<typename pcl::PointCloud<TPointType>::Ptr>, std::string, bool> input)
    {
        std::vector<typename pcl::PointCloud<TPointType>::Ptr> pointClouds = boost::tuples::get<0>(input);
        std::string text = boost::tuples::get<1>(input);
        bool step = boost::tuples::get<2>(input);

        std::cout << "Render; step= " << step << std::endl;

        baseRenderer->renderPointClouds(pointClouds, text, step);
    }

    std::string getName()
    {
        return "Point cloud Renderer";
    }
};

}
}
