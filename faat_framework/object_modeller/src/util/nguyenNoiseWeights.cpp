
#include "util/nguyenNoiseWeights.h"

#include <faat_pcl/utils/noise_models.h>

namespace object_modeller
{
namespace util
{

void NguyenNoiseWeights::applyConfig(Config &config)
{
    depth_edges = true;
    max_angle = 60.f;
    lateral_sigma = 0.002f;
}

std::vector<std::vector<float> > NguyenNoiseWeights::process(boost::tuples::tuple<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> , std::vector<pcl::PointCloud<pcl::Normal>::Ptr> > input)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds = input.get<0>();
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normal_clouds = input.get<1>();

    std::vector<std::vector<float> > weightsList;

    for(size_t i=0; i < pointClouds.size(); i++)
    {
        // calculate weights
        faat_pcl::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
        nm.setInputCloud(pointClouds[i]);
        nm.setInputNormals(normal_clouds[i]);
        nm.setLateralSigma(lateral_sigma);
        nm.setMaxAngle(max_angle);
        nm.setUseDepthEdges(depth_edges);
        nm.compute();
        std::vector<float> weights;
        nm.getWeights(weights);

        weightsList.push_back(weights);
    }

    return weightsList;
}

}
}
