
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>

namespace object_modeller
{
namespace modelling
{

class PoissonReconstruction :
        public InOutModule<pcl::PolygonMesh::Ptr,
                                std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{

public:
    pcl::PolygonMesh::Ptr process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> input);

    PoissonReconstruction();

    virtual void applyConfig(Config &config);

    std::string getName()
    {
        return "Poisson Reconstruction";
    }
};

}
}
