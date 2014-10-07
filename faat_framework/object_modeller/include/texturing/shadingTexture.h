
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/PolygonMesh.h>



namespace object_modeller
{
namespace texturing
{

class ShadingTexture :
        public InOutModule<output::TexturedMesh::Ptr,
                            boost::tuples::tuple<
                                pcl::PolygonMesh::Ptr,
                                std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>,
                                std::vector<Eigen::Matrix4f> >
                            >
{
public:
    ShadingTexture(std::string config_name="shadingTexture");

    output::TexturedMesh::Ptr process(boost::tuples::tuple<pcl::PolygonMesh::Ptr, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>, std::vector<Eigen::Matrix4f> > input);

    std::string getName()
    {
        return "Shading Texture";
    }
};

}
}
