#pragma once

#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

//#define KP_NO_CERES_AVAILABLE // Question: What is this?
// this should not be define in user code, otherwise if v4r gets compiled with CERES, then
// the size of the classes do not match and weird memory access occur

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
    kp::KeypointTracker::Parameter kt_param;
public:
    CameraTracker(std::string config_name="cameraTracker");

    virtual void applyConfig(Config::Ptr config);

    std::vector<Eigen::Matrix4f> process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds);

    bool trackSingle(pcl::PointCloud<pcl::PointXYZRGB>::Ptr keyframe, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, bool &is_keyframe);

    std::string getName()
    {
        return "Keypoint based Camera Tracker";
    }

    void trackSingleFrame(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud, Eigen::Matrix4f & pose, bool & is_key_frame);

};

}
}
