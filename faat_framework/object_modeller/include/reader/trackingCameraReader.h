#pragma once

#include "inputModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/io/openni_grabber.h>

#include "output/pointCloudRenderer.h"

#include "pipeline.h"

#include "registration/cameraTracker.h"

namespace object_modeller
{
namespace reader
{

class TrackingCameraReader : public InModule<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{
protected:
    pcl::Grabber* grabber;
    boost::shared_ptr<output::Renderer> renderer;

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> result;
    bool copyFrame;

    registration::CameraTracker *tracker;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr last_keyframe;
    bool m_trackerEnabled;
    int nrSequences;
public:
    TrackingCameraReader(registration::CameraTracker *tracker, output::Renderer::Ptr renderer, std::string config_name="trackingcamreader") : InModule(config_name), renderer(renderer)
    {
        this->tracker = tracker;
        grabber = NULL;
        m_trackerEnabled = false;
        nrSequences = 1;
    }

    virtual void applyConfig(Config::Ptr config)
    {
        ConfigItem::applyConfig(config);

        tracker->applyConfig(config);
    }

    virtual int getNrOutputSequences()
    {
        return nrSequences;
    }

    virtual std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> process();

    void grabberCallback(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud);
    void image_callback(const boost::shared_ptr<openni_wrapper::Image>& image);

    std::string getName()
    {
        return "Tracking Camera Reader";
    }
};

}
}
