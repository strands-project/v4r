
#include "texturing/shadingTexture.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/io/io.h>

#include <pcl/search/kdtree.h>

namespace object_modeller
{
namespace texturing
{

ShadingTexture::ShadingTexture(std::string config_name) : InOutModule(config_name)
{
}

output::TexturedMesh::Ptr ShadingTexture::process(boost::tuples::tuple<pcl::PolygonMesh::Ptr, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>, std::vector<Eigen::Matrix4f> > input)
{
    pcl::PolygonMesh::Ptr mesh = boost::tuples::get<0>(input);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud = boost::tuples::get<1>(input).at(0);

    pcl::PointCloud<pcl::PointXYZ> c;
    pcl::PointCloud<pcl::PointXYZRGBNormal> colored;
    pcl::fromPCLPointCloud2(mesh->cloud, c);
    pcl::copyPointCloud(c, colored);

    pcl::KdTree<pcl::PointXYZRGBNormal>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZRGBNormal>);
    tree->setInputCloud(cloud);

    for (int i=0;i<mesh->polygons.size();i++)
    {
        for (int j=0;j<mesh->polygons[i].vertices.size();j++)
        {
            uint32_t index = mesh->polygons[i].vertices[j];

            std::vector< int > ind;
            std::vector<float> dist;

            tree->nearestKSearch(colored[index], 1, ind, dist);

            colored[index].r = (*cloud)[ind.at(0)].r;
            colored[index].g = (*cloud)[ind.at(0)].g;
            colored[index].b = (*cloud)[ind.at(0)].b;
        }
    }

    pcl::toPCLPointCloud2(colored, mesh->cloud);

    output::TexturedMesh::Ptr result(new output::TexturedMesh());
    result->mesh = mesh;

    return result;
}

}
}
