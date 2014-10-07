
#include "inputModule.h"
#include "outputModule.h"
#include "ioModule.h"
#include "module.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "visualConfig.h"

namespace object_modeller
{
namespace util
{

class BoxFilter;

class BoxConfig : public VisualConfigBase
{
private:
    BoxFilter *filter;
public:
    BoxConfig(BoxFilter *module);

    void startConfig(output::Renderer::Ptr renderer);
};

class BoxFilter :
        public InOutModule<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{
    friend class BoxConfig;
private:
    Eigen::Vector3f dim;
    Eigen::Quaternionf rotation;
    Eigen::Vector3f translation;

public:
    BoxFilter(std::string config_name="boxFilter");

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> input);

    std::string getName()
    {
        return "Box Filter";
    }
};

}
}
