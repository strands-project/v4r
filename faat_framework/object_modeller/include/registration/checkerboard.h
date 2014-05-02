
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv/cv.h>

namespace object_modeller
{
namespace registration
{

class CheckerboardRegistration :
        public InOutModule<std::vector<Eigen::Matrix4f>,
                        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{
private:

    std::vector<cv::Size> boardSizes;

public:
    CheckerboardRegistration(std::string config_name="checkerboardRegistration") : InOutModule(config_name)
    {}

    virtual void applyConfig(Config &config);

    std::vector<Eigen::Matrix4f> process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> input);

    std::string getName()
    {
        return "Checkerboard registration";
    }
};

}
}
