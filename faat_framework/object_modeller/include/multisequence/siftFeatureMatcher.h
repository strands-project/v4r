#pragma once

#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <faat_pcl/3d_rec_framework/defines/faat_3d_rec_framework_defines.h>

namespace object_modeller
{
namespace multisequence
{


class SiftFeatureMatcher :
        public InOutModule<Eigen::Matrix4f,
                                boost::tuples::tuple<
                                    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                                    std::vector<Eigen::Matrix4f>,
                                    std::vector<std::vector<int> >,
                                    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>
                                > >
{
private:
    std::vector<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > inputSequences;
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> models;
    std::vector<std::vector<Eigen::Matrix4f> > poses;
    std::vector<std::vector<std::vector<int> > > indices;

public:
    Eigen::Matrix4f process(boost::tuples::tuple<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>, std::vector<Eigen::Matrix4f>, std::vector<std::vector<int> >, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> > input);

    SiftFeatureMatcher(std::string config_name="siftFeatureMatcher");

    std::string getName()
    {
        return "Sift feature matcher";
    }
};

}
}
