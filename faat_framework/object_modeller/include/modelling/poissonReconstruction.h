
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/surface/impl/mls.hpp>
#include <pcl/surface/poisson.h>
#include <pcl/surface/mls.h>
#include <pcl/impl/instantiate.hpp>

PCL_INSTANTIATE_PRODUCT (MovingLeastSquares, ((pcl::PointXYZRGBNormal))((pcl::PointXYZRGBNormal)))

namespace object_modeller
{
namespace modelling
{

class PoissonReconstruction :
        public InOutModule<pcl::PolygonMesh::Ptr,
                                std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> >
{

public:
    pcl::PolygonMesh::Ptr process(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> input);

    PoissonReconstruction(std::string config_name="poissonReconstruction");

    virtual void applyConfig(Config &config);

    std::string getName()
    {
        return "Poisson Reconstruction";
    }
};

}
}
