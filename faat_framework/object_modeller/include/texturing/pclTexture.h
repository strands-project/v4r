#pragma once

#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/surface/texture_mapping.h>

#include <pcl/range_image/range_image_planar.h>

namespace object_modeller
{
namespace texturing
{

class PclTexture :
        public InOutModule<output::TexturedMesh::Ptr,
                            boost::tuples::tuple<
                                pcl::PolygonMesh::Ptr,
                                std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                                std::vector<Eigen::Matrix4f> >
                            >
{
private:

    int angleThreshDegree;
    bool projectNormals;
    double angleThresh;

public:
    PclTexture(std::string config_name="pclTexture");

    output::TexturedMesh::Ptr process(boost::tuples::tuple<pcl::PolygonMesh::Ptr, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>, std::vector<Eigen::Matrix4f> > input);

    void addTexture(pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud, output::TexturedMesh::Ptr result, pcl::PointCloud<pcl::PointXYZRGB>::Ptr image, Eigen::Matrix4f pose, int imageIndex, std::vector<int> bestImages);

    Eigen::Vector3f calculateNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud, pcl::Vertices vertices, Eigen::Matrix4f pose);

    pcl::RangeImagePlanar getMeshProjection(pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Eigen::Matrix4f pose);

    std::string getName()
    {
        return "PCL Texture";
    }
};

}
}
