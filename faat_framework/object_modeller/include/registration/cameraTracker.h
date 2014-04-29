
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#define KP_NO_CERES_AVAILABLE // Question: What is this?

#include "v4r/KeypointCameraTrackerPCL/CameraTrackerRGBDPCL.hh"

namespace object_modeller
{
namespace registration
{

class CameraTracker :
        public InOutModule<std::vector<Eigen::Matrix4f>,
                           std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{
private:
    kp::CameraTrackerRGBDPCL::Ptr camtracker;
    bool keyframesOnly;
public:

    virtual void applyConfig(Config &config);

    std::vector<Eigen::Matrix4f> process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds);

    std::string getName()
    {
        return "Keypoint based Camera Tracker";
    }
};

}
}
