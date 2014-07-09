
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace segmentation
{

class NotNaNSegmentation :
        public InOutModule<std::vector<std::vector<int> >,
                           std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{
public:
    NotNaNSegmentation(std::string config_name="notNaNSegmentation");

    std::vector<std::vector<int> > process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> input);

    virtual void applyConfig(Config &config);

    std::string getName()
    {
        return "Not NaN segmentation";
    }
};

}
}
